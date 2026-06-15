#!/usr/bin/env python3
"""Shared 7:3 patient-level training pipeline for private 3D MRI classifiers."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("Please install nibabel first: pip install nibabel") from exc

try:
    from scipy.ndimage import zoom
except ImportError as exc:
    raise ImportError("Please install scipy first: pip install scipy") from exc


@dataclass
class TrainConfig:
    img_root: str = os.environ.get("IMG_ROOT", "")
    excel_path: str = os.environ.get("EXCEL_PATH", "")
    id_col: str = os.environ.get("ID_COL", "case_id")
    label_col: str = os.environ.get("LABEL_COL", "label")

    output_root: str = "Model_7_3_Output"
    input_size: Tuple[int, int, int] = (64, 64, 64)
    modalities: Tuple[str, str, str] = ("T1WI", "T1WI+C", "T2WI")

    random_state: int = 2024
    test_size: float = 0.30
    batch_size: int = 2
    epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    threshold: float = 0.5
    use_amp: bool = True
    augment: bool = True

    # Transfer-learning controls. A run is transfer learning only when
    # pretrained_path points to an existing checkpoint and weights are loaded.
    pretrained_path: str = os.environ.get("PRETRAINED_PATH", "")
    strict_load: bool = False
    freeze_backbone: bool = False
    freeze_epochs: int = 0
    head_lr_mult: float = 5.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def clean_patient_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def infer_id_column(df: pd.DataFrame, img_root: Path) -> str:
    folder_names = {p.name for p in img_root.iterdir() if p.is_dir()} if img_root.exists() else set()
    best_col = ""
    best_hits = -1
    for col in df.columns:
        values = df[col].dropna().map(clean_patient_id).astype(str).head(500)
        hits = sum(v in folder_names for v in values) if folder_names else values.nunique()
        if hits > best_hits:
            best_hits = hits
            best_col = str(col)
    if not best_col:
        raise ValueError("Could not infer patient ID column. Please pass --id-col.")
    return best_col


def infer_label_column(df: pd.DataFrame, exclude: str) -> str:
    candidates: List[Tuple[int, str]] = []
    for col in df.columns:
        if str(col) == exclude:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        unique = set(numeric.round(6).unique().tolist())
        if unique.issubset({0.0, 1.0}) and len(unique) == 2:
            candidates.append((len(numeric), str(col)))
    if not candidates:
        raise ValueError("Could not infer binary label column. Please pass --label-col.")
    return sorted(candidates, reverse=True)[0][1]


def load_label_table(config: TrainConfig) -> pd.DataFrame:
    excel_path = Path(config.excel_path)
    img_root = Path(config.img_root)
    if not excel_path.exists():
        raise FileNotFoundError(f"Label file not found: {excel_path}")
    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")

    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = config.id_col.strip() or infer_id_column(df, img_root)
    label_col = config.label_col.strip() or infer_label_column(df, id_col)

    work = df[[id_col, label_col]].copy()
    work.columns = ["patient_id", "label"]
    work["patient_id"] = work["patient_id"].map(clean_patient_id)
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work = work.dropna(subset=["patient_id", "label"])
    work["label"] = work["label"].astype(int)
    work = work[work["label"].isin([0, 1])]
    work = work.drop_duplicates(subset=["patient_id"], keep="first")

    valid_mask = work["patient_id"].map(lambda pid: (img_root / pid).is_dir())
    missing_count = int((~valid_mask).sum())
    if missing_count:
        print(f"[WARN] {missing_count} patients have no folder under img_root and will be skipped.")
    work = work[valid_mask].reset_index(drop=True)

    if work["label"].nunique() < 2:
        raise ValueError("Only one class remained after filtering. Please check labels and paths.")

    print(f"[INFO] ID column: {id_col}")
    print(f"[INFO] Label column: {label_col}")
    print(f"[INFO] Valid patients: {len(work)}")
    print(f"[INFO] Class distribution: {work['label'].value_counts().to_dict()}")
    return work


def modality_aliases(modality: str) -> Sequence[str]:
    aliases = {
        "T1WI": ("T1WI", "T1", "t1wi", "t1"),
        "T1WI+C": ("T1WI+C", "T1WI_C", "T1C", "T1CE", "T1WI-CE", "t1wi+c", "t1c"),
        "T2WI": ("T2WI", "T2", "t2wi", "t2"),
    }
    return aliases.get(modality, (modality,))


def find_modality_file(img_root: Path, patient_id: str, modality: str) -> Optional[Path]:
    patient_dir = img_root / patient_id
    for alias in modality_aliases(modality):
        modality_dir = patient_dir / alias
        if modality_dir.is_dir():
            for item in [modality_dir / f"{patient_id}.nii.gz", modality_dir / f"{patient_id}.nii"]:
                if item.exists():
                    return item
            files = sorted(list(modality_dir.glob("*.nii.gz")) + list(modality_dir.glob("*.nii")))
            if files:
                return files[0]

    normalized_mod = modality.replace("+", "").replace("_", "").replace("-", "").lower()
    files = sorted(
        p
        for p in patient_dir.rglob("*")
        if p.is_file()
        and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        and normalized_mod in p.as_posix().replace("+", "").replace("_", "").replace("-", "").lower()
    )
    return files[0] if files else None


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(volume, [1, 99])
    if hi > lo:
        volume = np.clip(volume, lo, hi)
    std = float(volume.std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - float(volume.mean())) / std).astype(np.float32)


def load_patient_tensor(img_root: Path, patient_id: str, config: TrainConfig) -> np.ndarray:
    channels: List[np.ndarray] = []
    missing: List[str] = []
    for modality in config.modalities:
        nii_path = find_modality_file(img_root, patient_id, modality)
        if nii_path is None:
            missing.append(modality)
            channels.append(np.zeros(config.input_size, dtype=np.float32))
            continue
        volume = nib.load(str(nii_path)).get_fdata()
        volume = resize_volume(volume, config.input_size)
        channels.append(normalize_volume(volume))
    if missing:
        print(f"[WARN] Patient {patient_id}: missing {missing}; zero-filled these channels.")
    return np.stack(channels, axis=0).astype(np.float32)


class Patient3DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, config: TrainConfig, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if not self.train or not self.config.augment:
            return image
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
        if random.random() < 0.5:
            image = np.flip(image, axis=3)
        if random.random() < 0.25:
            image = image + np.random.normal(0.0, 0.03, size=image.shape).astype(np.float32)
        return np.ascontiguousarray(image)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        patient_id = str(row["patient_id"])
        image = self._augment(load_patient_tensor(self.img_root, patient_id, self.config))
        label = float(row["label"])
        return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.float32), patient_id


class Safe3DClassifier(nn.Module):
    """Fallback classifier used only when a supplied model has unavailable dependencies."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class SegmentationOutputToLogit(nn.Module):
    """Adapts segmentation-style [B, C, D, H, W] output to one patient-level logit."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            out = out.mean(dim=tuple(range(2, out.ndim)))
        if out.ndim == 1:
            out = out.unsqueeze(1)
        if out.shape[1] > 1:
            out = out[:, :1]
        return out


def output_to_logits(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim > 2:
        output = output.mean(dim=tuple(range(2, output.ndim)))
    if output.ndim == 1:
        output = output.unsqueeze(1)
    if output.shape[1] > 1:
        output = output[:, :1]
    return output.view(-1)


def create_split(df: pd.DataFrame, config: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("[INFO] Split mode: train+validation merged, test held out")
    print(f"[INFO] Train patients: {len(train_df)}")
    print(f"[INFO] Test patients:  {len(test_df)}")
    print(f"[INFO] Train class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"[INFO] Test class distribution:  {test_df['label'].value_counts().to_dict()}")
    return train_df, test_df


def make_output_dir(config: TrainConfig, model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(" ", "_")
    out_dir = Path(config.output_root) / f"{safe_name}_7_3_{stamp}"
    for sub in ["checkpoints", "metrics", "predictions"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    return out_dir


def _strip_state_dict_prefix(key: str) -> str:
    prefixes = ("module.", "model.", "net.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Could not find a model state_dict in the checkpoint.")


def load_pretrained_weights(model: nn.Module, pretrained_path: str, strict: bool = False) -> Dict[str, object]:
    if not pretrained_path:
        print("[INFO] No --pretrained provided. This run is scratch training, not transfer learning.")
        return {"loaded": False, "path": "", "matched": 0, "skipped": 0}

    ckpt_path = Path(pretrained_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    raw_state = extract_state_dict(checkpoint)
    model_state = model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in raw_state.items():
        clean_key = _strip_state_dict_prefix(str(key))
        if clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
            matched[clean_key] = value
        else:
            skipped.append(str(key))

    if strict:
        model.load_state_dict(raw_state, strict=True)  # type: ignore[arg-type]
        matched_count = len(raw_state)
        skipped_count = 0
    else:
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        matched_count = len(matched)
        skipped_count = len(skipped)

    print(f"[INFO] Transfer learning checkpoint loaded: {ckpt_path}")
    print(f"[INFO] Matched parameters: {matched_count}; skipped parameters: {skipped_count}")
    if not strict and skipped[:10]:
        print(f"[INFO] First skipped keys: {skipped[:10]}")
    return {
        "loaded": True,
        "path": str(ckpt_path),
        "matched": matched_count,
        "skipped": skipped_count,
        "strict": strict,
    }


def is_head_parameter(name: str) -> bool:
    head_keywords = (
        "fc",
        "classifier",
        "head",
        "task_head",
        "out.",
        "decoder.task_head",
    )
    return any(keyword in name for keyword in head_keywords)


def set_finetune_trainable(model: nn.Module, freeze_backbone: bool) -> None:
    if not freeze_backbone:
        for param in model.parameters():
            param.requires_grad = True
        return

    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = is_head_parameter(name)
        trainable += int(param.requires_grad)

    if trainable == 0:
        # Safety fallback: if the model uses unusual names, train the last few tensors.
        params = list(model.parameters())
        for param in params:
            param.requires_grad = False
        for param in params[-4:]:
            param.requires_grad = True


def build_optimizer(model: nn.Module, config: TrainConfig) -> optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_head_parameter(name):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": config.learning_rate})
    if head_params:
        groups.append({"params": head_params, "lr": config.learning_rate * config.head_lr_mult})
    if not groups:
        raise RuntimeError("No trainable parameters found after applying freeze settings.")
    return optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, object]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr: List[float] = []
    tpr: List[float] = []
    thresholds: List[float] = []
    if len(set(y_true)) == 2:
        fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_prob)
        fpr, tpr, thresholds = fpr_arr.tolist(), tpr_arr.tolist(), thr_arr.tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }


def save_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_scaler(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def autocast_ctx(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if not enabled:
        yield
        return
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast("cuda", enabled=True):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=True):
            yield


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_prob: List[float] = []
    for images, labels, _patient_ids in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            logits = output_to_logits(model(images))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        y_prob.extend(torch.sigmoid(logits.detach()).cpu().numpy().astype(float).tolist())
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        losses.append(float(loss.item()))
    return float(np.mean(losses)), safe_auc(y_true, y_prob)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    patient_ids_all: List[str] = []
    for images, labels, patient_ids in tqdm(loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        logits = output_to_logits(model(images))
        probs = torch.sigmoid(logits).cpu().numpy().astype(float).tolist()
        y_prob.extend(probs)
        y_true.extend(labels.numpy().astype(int).tolist())
        patient_ids_all.extend([str(pid) for pid in patient_ids])
    metrics = compute_metrics(y_true, y_prob, threshold)
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    predictions = [
        {
            "patient_id": pid,
            "true_label": int(t),
            "predicted_label": int(pred),
            "probability": float(prob),
            "confidence": float(max(prob, 1.0 - prob)),
            "correct": int(t == pred),
        }
        for pid, t, pred, prob in zip(patient_ids_all, y_true, y_pred, y_prob)
    ]
    return metrics, predictions


def parse_common_args(default_config: Optional[TrainConfig] = None) -> TrainConfig:
    defaults = default_config or TrainConfig()
    parser = argparse.ArgumentParser(description="Train a 3D classifier with a 7:3 patient-level split.")
    parser.add_argument("--img-root", default=defaults.img_root)
    parser.add_argument("--excel-path", default=defaults.excel_path)
    parser.add_argument("--id-col", default=defaults.id_col)
    parser.add_argument("--label-col", default=defaults.label_col)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--input-size", default=",".join(map(str, defaults.input_size)))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", default=defaults.pretrained_path, help="Checkpoint path for transfer learning.")
    parser.add_argument("--strict-load", action="store_true", help="Require exact checkpoint key matching.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze non-classifier layers.")
    parser.add_argument("--freeze-epochs", type=int, default=defaults.freeze_epochs, help="Freeze backbone for N warm-up epochs, then unfreeze.")
    parser.add_argument("--head-lr-mult", type=float, default=defaults.head_lr_mult, help="Learning-rate multiplier for classifier/head parameters.")
    args = parser.parse_args()
    dims = tuple(int(x.strip()) for x in args.input_size.split(","))
    if len(dims) != 3:
        raise ValueError("--input-size must be D,H,W, for example 64,64,64.")
    return TrainConfig(
        img_root=args.img_root,
        excel_path=args.excel_path,
        id_col=args.id_col,
        label_col=args.label_col,
        output_root=args.output_root,
        input_size=dims,  # type: ignore[arg-type]
        random_state=args.random_state,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        augment=not args.no_augment,
        pretrained_path=args.pretrained,
        strict_load=args.strict_load,
        freeze_backbone=args.freeze_backbone,
        freeze_epochs=args.freeze_epochs,
        head_lr_mult=args.head_lr_mult,
    )


def run_training(model_name: str, build_model: Callable[[TrainConfig], nn.Module], default_config: Optional[TrainConfig] = None) -> None:
    config = parse_common_args(default_config)
    if not config.img_root or not config.excel_path:
        raise ValueError("Please set --img-root and --excel-path.")

    seed_everything(config.random_state)
    out_dir = make_output_dir(config, model_name)
    save_json(out_dir / "metrics" / "config.json", asdict(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {out_dir.resolve()}")

    df = load_label_table(config)
    train_df, test_df = create_split(df, config)
    save_json(
        out_dir / "metrics" / "split_summary.json",
        {
            "split": "train_test_7_3",
            "train_patients": train_df["patient_id"].tolist(),
            "test_patients": test_df["patient_id"].tolist(),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_distribution": train_df["label"].value_counts().to_dict(),
            "test_distribution": test_df["label"].value_counts().to_dict(),
        },
    )

    train_loader = DataLoader(
        Patient3DDataset(train_df, config.img_root, config, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Patient3DDataset(test_df, config.img_root, config, train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    transfer_info = load_pretrained_weights(model, config.pretrained_path, config.strict_load)
    freeze_now = config.freeze_backbone or config.freeze_epochs > 0
    set_finetune_trainable(model, freeze_now)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,}")
    with torch.no_grad():
        dummy = torch.zeros(1, len(config.modalities), *config.input_size, device=device)
        dummy_logits = output_to_logits(model(dummy))
        if dummy_logits.shape[0] != 1:
            raise RuntimeError(f"{model_name} dry-run output shape is invalid: {tuple(dummy_logits.shape)}")
    print(f"[INFO] Dry-run OK. Output shape after adapter: {tuple(dummy_logits.shape)}")

    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device))
    optimizer = build_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = make_scaler(device, config.use_amp)

    history: Dict[str, List[float]] = {"train_loss": [], "train_auc": [], "learning_rate": []}
    best_train_loss = math.inf
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, config.use_amp)
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["learning_rate"].append(lr_now)
        print(f"Epoch {epoch:03d}/{config.epochs} | train_loss={train_loss:.4f} | train_auc={train_auc:.4f} | lr={lr_now:.6g}")
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "best_train_loss": best_train_loss,
                    "model_name": model_name,
                },
                out_dir / "checkpoints" / "best_train_loss.pth",
            )
        save_json(out_dir / "metrics" / "training_history.json", history)

        if config.freeze_epochs > 0 and epoch == config.freeze_epochs:
            print(f"[INFO] Freeze warm-up finished at epoch {epoch}. Unfreezing all layers for fine-tuning.")
            set_finetune_trainable(model, freeze_backbone=False)
            optimizer = build_optimizer(model, config)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - epoch, 1),
            )

    checkpoint = torch.load(out_dir / "checkpoints" / "best_train_loss.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, predictions = evaluate(model, test_loader, device, config.threshold)

    final_results = {
        "experiment_info": {
            "experiment_name": out_dir.name,
            "model_name": model_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "training_seconds": time.time() - start,
        },
        "data_info": {
            "total_valid_patients": len(df),
            "train_patients": len(train_df),
            "test_patients": len(test_df),
            "split": "7:3 train/test; validation merged into training",
            "input_size": list(config.input_size),
            "modalities": list(config.modalities),
            "patient_level_prediction": True,
        },
        "training_history": history,
        "transfer_learning": transfer_info,
        "finetuning": {
            "freeze_backbone": config.freeze_backbone,
            "freeze_epochs": config.freeze_epochs,
            "head_lr_mult": config.head_lr_mult,
        },
        "best_train_loss": best_train_loss,
        "test_results": test_metrics,
    }
    save_json(out_dir / "metrics" / "final_results.json", final_results)
    save_json(out_dir / "predictions" / "test_predictions.json", predictions)

    print("\n[RESULT] Test metrics")
    for key in ["accuracy", "auc", "precision", "recall", "specificity", "f1_score", "mcc"]:
        value = test_metrics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"confusion_matrix: {test_metrics['confusion_matrix']}")
    print(f"[INFO] Saved predictions to: {(out_dir / 'predictions' / 'test_predictions.json').resolve()}")
