#!/usr/bin/env python3
"""
Train AMSNet on the same private 3D medical-image dataset pipeline.

Key choices for this integration:
1. Patient-level prediction: one 3D tensor per patient, one prediction per patient.
2. Split: train/test = 7/3. The previous train and validation sets are merged.
3. Input tensor shape: [B, 3, D, H, W], matching the AMSNet implementation.
4. Robust modality loading: T1WI, T1WI+C, and T2WI are resized before stacking.

Run example:
python AMSNet_7_3_private_dataset.py --img-root "D:/path/to/images" --excel-path "D:/path/to/labels.xlsx"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class Config:
    # Set these paths by command line, environment variables, or edit them here.
    img_root: str = os.environ.get("IMG_ROOT", "data/private_mri")
    excel_path: str = os.environ.get(
        "EXCEL_PATH",
        "metadata/private_labels_anonymous.xlsx",
    )
    id_col: str = os.environ.get("ID_COL", "case_id")
    label_col: str = os.environ.get("LABEL_COL", "label")

    output_root: str = "AMSNet_7_3_Output"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class MSIB(nn.Module):
    """Multi-scale information block from the supplied AMSNet code."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inter_channels = max(1, out_channels // 4)

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(inter_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fusion(torch.cat([b1, b2, b3], dim=1))
        return fused + self.residual(x)


class AMSNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout_rate: float = 0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            MSIB(64, 64),
            ChannelAttention3D(64),
            MSIB(64, 64),
        )

        self.layer2 = nn.Sequential(
            MSIB(64, 128),
            ChannelAttention3D(128),
            nn.MaxPool3d(2),
        )

        self.layer3 = nn.Sequential(
            MSIB(128, 256),
            ChannelAttention3D(256),
            nn.MaxPool3d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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
        if folder_names:
            hits = sum(v in folder_names for v in values)
        else:
            hits = values.nunique()
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
        rounded = numeric.round(6)
        unique = set(rounded.unique().tolist())
        if unique.issubset({0.0, 1.0}) and len(unique) == 2:
            candidates.append((len(numeric), str(col)))

    if not candidates:
        raise ValueError("Could not infer binary label column. Please pass --label-col.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_label_table(config: Config) -> pd.DataFrame:
    excel_path = Path(config.excel_path)
    img_root = Path(config.img_root)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {excel_path}. Set --excel-path or EXCEL_PATH."
        )
    if not img_root.exists():
        raise FileNotFoundError(
            f"Image root not found: {img_root}. Set --img-root or IMG_ROOT."
        )

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
            direct = [
                modality_dir / f"{patient_id}.nii.gz",
                modality_dir / f"{patient_id}.nii",
            ]
            for item in direct:
                if item.exists():
                    return item
            files = sorted(list(modality_dir.glob("*.nii.gz")) + list(modality_dir.glob("*.nii")))
            if files:
                return files[0]

    files = sorted(
        [
            p
            for p in patient_dir.rglob("*")
            if p.is_file()
            and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
            and modality.replace("+", "").replace("_", "").lower()
            in p.as_posix().replace("+", "").replace("_", "").lower()
        ]
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
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - mean) / std).astype(np.float32)


def load_patient_tensor(img_root: Path, patient_id: str, config: Config) -> np.ndarray:
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
        volume = normalize_volume(volume)
        channels.append(volume)

    if missing:
        print(f"[WARN] Patient {patient_id}: missing {missing}; zero-filled these channels.")
    return np.stack(channels, axis=0).astype(np.float32)


class Patient3DDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_root: str,
        config: Config,
        train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if not self.train or not self.config.augment:
            return image
        # image shape: [C, D, H, W]
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
        label = float(row["label"])
        image = load_patient_tensor(self.img_root, patient_id, self.config)
        image = self._augment(image)
        return (
            torch.from_numpy(image).float(),
            torch.tensor(label, dtype=torch.float32),
            patient_id,
        )


def create_split(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def make_output_dir(config: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_root) / f"AMSNet_7_3_{stamp}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, object]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc_value = safe_auc(y_true, y_prob)

    fpr, tpr, thresholds = [], [], []
    if len(set(y_true)) == 2:
        fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_prob)
        fpr = fpr_arr.tolist()
        tpr = tpr_arr.tolist()
        thresholds = thr_arr.tolist()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc_value,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        },
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
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
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits.detach()).cpu().numpy().tolist()
        y_prob.extend(float(p) for p in probs)
        y_true.extend(int(v) for v in labels.detach().cpu().numpy().tolist())
        losses.append(float(loss.item()))

    return float(np.mean(losses)), safe_auc(y_true, y_prob)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    patient_ids_all: List[str] = []

    for images, labels, patient_ids in tqdm(loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

        y_prob.extend(float(p) for p in probs)
        y_true.extend(int(v) for v in labels.numpy().tolist())
        patient_ids_all.extend([str(pid) for pid in patient_ids])

    metrics = compute_metrics(y_true, y_prob, threshold)
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    predictions = [
        {
            "patient_id": pid,
            "true_label": int(t),
            "predicted_label": int(pred),
            "probability": float(prob),
            "correct": int(t == pred),
        }
        for pid, t, pred, prob in zip(patient_ids_all, y_true, y_pred, y_prob)
    ]
    return metrics, predictions


def save_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train AMSNet with a 7:3 patient-level split.")
    parser.add_argument("--img-root", default=Config.img_root, help="Root folder containing patient folders.")
    parser.add_argument("--excel-path", default=Config.excel_path, help="Excel file with labels.")
    parser.add_argument("--id-col", default=Config.id_col, help="Patient ID column name. Optional.")
    parser.add_argument("--label-col", default=Config.label_col, help="Binary label column name. Optional.")
    parser.add_argument("--output-root", default=Config.output_root)
    parser.add_argument("--input-size", default="64,64,64", help="D,H,W. Example: 64,64,64 or 128,128,128.")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--test-size", type=float, default=Config.test_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    dims = tuple(int(x.strip()) for x in args.input_size.split(","))
    if len(dims) != 3:
        raise ValueError("--input-size must contain three comma-separated integers.")

    return Config(
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
    )


def main() -> None:
    config = parse_args()
    seed_everything(config.random_state)

    if not config.img_root or not config.excel_path:
        raise ValueError(
            "Please set --img-root and --excel-path, or define IMG_ROOT and EXCEL_PATH environment variables."
        )

    out_dir = make_output_dir(config)
    save_json(out_dir / "metrics" / "config.json", asdict(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_dataset = Patient3DDataset(train_df, config.img_root, config, train=True)
    test_dataset = Patient3DDataset(test_df, config.img_root, config, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = AMSNet(in_channels=3, num_classes=1).to(device)

    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())
    pos_weight_value = neg / max(pos, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

    history: Dict[str, List[float]] = {"train_loss": [], "train_auc": [], "learning_rate": []}
    best_train_loss = math.inf
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=config.use_amp,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["learning_rate"].append(float(lr_now))

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | train_auc={train_auc:.4f} | lr={lr_now:.6g}"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "best_train_loss": best_train_loss,
                },
                out_dir / "checkpoints" / "best_train_loss.pth",
            )

        save_json(out_dir / "metrics" / "training_history.json", history)

    elapsed = time.time() - start

    checkpoint = torch.load(out_dir / "checkpoints" / "best_train_loss.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, predictions = evaluate(model, test_loader, device, config.threshold)

    final_results = {
        "experiment_info": {
            "experiment_name": out_dir.name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "training_seconds": elapsed,
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
        "model_info": {
            "name": "AMSNet",
            "input_shape": "[B, 3, D, H, W]",
            "output": "single binary logit per patient",
        },
        "training_history": history,
        "best_train_loss": best_train_loss,
        "test_results": test_metrics,
    }
    save_json(out_dir / "metrics" / "final_results.json", final_results)
    save_json(out_dir / "predictions" / "test_predictions.json", predictions)

    print("\n[RESULT] Test metrics")
    for key in ["accuracy", "auc", "precision", "recall", "specificity", "f1_score", "mcc"]:
        value = test_metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"confusion_matrix: {test_metrics['confusion_matrix']}")
    print(f"[INFO] Saved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
