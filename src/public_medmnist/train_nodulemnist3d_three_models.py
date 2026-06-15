#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train M3D-MILNet, MedPipe-style and MedViT-3D on NoduleMNIST3D.

The goal is to create fair checkpoints before heatmap comparison. NoduleMNIST3D
is single-channel, so each volume is repeated into three channels to match the
models originally written for three MRI modalities.

MedPipe is implemented as a paper-inspired fixed instantiation of the MedPipe
search space: a 3D convolutional stem followed by six 3D MBConv-style blocks
with output channels 32, 48, 64, 96, 160 and 320. The original paper performs
joint NAS over augmentation and architecture; this script trains a deterministic
candidate from the same operation family so it can be compared and visualized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def extract_python_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#!") or line.startswith("import ") or line.startswith("from "):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if line.startswith("cat >") and "<< 'EOF'" in line:
                start = i + 1
                break
    if start is None:
        start = 0

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].strip() == "EOF":
            end = i
            break
    return "\n".join(lines[start:end]) + "\n"


def load_module_from_file(path: Path, module_name: str):
    path = path.expanduser().resolve()
    if path.suffix.lower() in {".txt", ".sh"}:
        code = extract_python_source(path)
        module_path = Path(tempfile.gettempdir()) / f"{module_name}_{abs(hash(str(path)))}.py"
        module_path.write_text(code, encoding="utf-8")
        path = module_path

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dhw(volume: object) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")
    return arr.astype(np.float32)


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32)
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32))
    lo, hi = np.percentile(volume, [1, 99])
    if hi > lo:
        volume = np.clip(volume, lo, hi)
    std = float(volume.std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - float(volume.mean())) / std).astype(np.float32)


class NoduleTensorDataset(Dataset):
    def __init__(self, split: str, root: str, size: int, download: bool, augment: bool = False) -> None:
        try:
            from medmnist import NoduleMNIST3D
        except ImportError as exc:
            raise ImportError("Please install medmnist first: pip install medmnist") from exc

        self.base = NoduleMNIST3D(split=split, size=size, download=download, root=root)
        self.split = split
        self.size = int(size)
        self.target_shape = (self.size, self.size, self.size)
        self.augment = bool(augment)
        self.labels: List[int] = []
        for i in range(len(self.base)):
            _x, y = self.base[i]
            self.labels.append(int(np.asarray(y).reshape(-1)[0]))

        unique = sorted(set(self.labels))
        if unique != [0, 1]:
            raise ValueError(f"This script expects binary labels [0, 1], got {unique}.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        volume = ensure_dhw(x)
        volume = resize_volume(volume, self.target_shape)
        volume = normalize_volume(volume)

        if self.augment:
            for axis in range(3):
                if random.random() < 0.5:
                    volume = np.flip(volume, axis=axis)
            if random.random() < 0.35:
                volume = volume + np.random.normal(0, 0.03, size=volume.shape).astype(np.float32)

        channels = np.repeat(volume[None, ...], 3, axis=0).copy().astype(np.float32)
        label = int(np.asarray(y).reshape(-1)[0])
        return torch.from_numpy(channels), torch.tensor(label, dtype=torch.long), f"{self.split}_{index:04d}"


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MBConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        hidden = in_channels * expand_ratio
        padding = kernel_size // 2
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm3d(hidden),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv3d(hidden, hidden, kernel_size, stride=stride, padding=padding, groups=hidden, bias=False),
            nn.BatchNorm3d(hidden),
            nn.SiLU(inplace=True),
            SqueezeExcite3D(hidden),
            nn.Conv3d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
        ]
        self.block = nn.Sequential(*layers)
        self.drop = nn.Dropout3d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            y = self.drop(y)
            return x + y
        return y


class MedPipe3D(nn.Module):
    """A deterministic 3D MBConv candidate from the MedPipe search family."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )
        channels = [32, 48, 64, 96, 160, 320]
        kernels = [3, 5, 3, 5, 3, 5]
        expands = [1, 4, 4, 4, 6, 6]
        repeats = [1, 1, 2, 2, 2, 1]
        strides = [1, 2, 2, 2, 2, 1]
        blocks: List[nn.Module] = []
        c_in = 32
        for c_out, k, e, r, s in zip(channels, kernels, expands, repeats, strides):
            for j in range(r):
                blocks.append(MBConv3D(c_in, c_out, k, stride=s if j == 0 else 1, expand_ratio=e, drop_rate=0.05))
                c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv3d(c_in, 384, 1, bias=False),
            nn.BatchNorm3d(384),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


@dataclass
class TrainResult:
    model_name: str
    out_dir: str
    best_ckpt: str
    best_epoch: int
    best_val_auc: float
    best_val_acc: float
    test_auc: float
    test_acc: float


def output_to_logits(output) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("logits", output)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.reshape(output.shape[0], -1)


def binary_probs_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    logits = logits.reshape(logits.shape[0], -1)
    if loss_type == "ce":
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits[:, 0])


def make_loss_fn(loss_type: str, train_labels: Sequence[int], device: torch.device):
    labels = np.asarray(train_labels, dtype=np.int64)
    neg = int((labels == 0).sum())
    pos = int((labels == 1).sum())
    if neg == 0 or pos == 0:
        raise ValueError(f"Both classes are required, got neg={neg}, pos={pos}")
    if loss_type == "ce":
        weights = torch.tensor(
            [len(labels) / (2.0 * neg), len(labels) / (2.0 * pos)],
            dtype=torch.float32,
            device=device,
        )
        return lambda logits, y: F.cross_entropy(logits, y.long(), weight=weights)
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    return lambda logits, y: F.binary_cross_entropy_with_logits(logits[:, 0], y.float(), pos_weight=pos_weight)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_type: str, device: torch.device) -> Dict[str, object]:
    model.eval()
    probs: List[float] = []
    preds: List[int] = []
    labels: List[int] = []
    sample_ids: List[str] = []
    for x, y, ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = output_to_logits(model(x))
        prob = binary_probs_from_logits(logits, loss_type)
        pred = (prob >= 0.5).long()
        probs.extend(prob.detach().cpu().numpy().astype(float).tolist())
        preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        sample_ids.extend(list(ids))

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float("nan"),
        "predictions": [
            {
                "sample_id": sid,
                "true_label": int(label),
                "pred_label": int(pred),
                "prob_class1": float(prob),
                "correct": int(label == pred),
            }
            for sid, label, pred, prob in zip(sample_ids, labels, preds, probs)
        ],
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        pass
    return metrics


def save_predictions_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_state_flexible(model: nn.Module, ckpt_path: Path) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, dict):
        raise RuntimeError(f"Cannot parse checkpoint: {ckpt_path}")
    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        clean = key
        for prefix in ("module.", "model.", "net.", "backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if clean in model_state and tuple(model_state[clean].shape) == tuple(value.shape):
            filtered[clean] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[INFO] Loaded {ckpt_path.name}: matched={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}")


def build_final_model(script_path: Path, size: int, device: torch.device) -> nn.Module:
    module = load_module_from_file(script_path, "final_nodule_train")
    config = module.EnhancedConfig(None)
    config.DETECTOR_IMG_SIZE = (size, size, size)
    config.IN_CHANNELS = 3
    config.NUM_CLASSES = 2
    config.ROI_SIZE = min(int(config.ROI_SIZE), size)
    config.ROI_SUPPRESS_RADIUS = max(1, int(config.ROI_SIZE) // 2)
    config.device = device
    model = module.MacroToMicroDensityNet(module.EnhancedUnsupervisedTransformerDetector(config), config)
    return model.to(device)


def build_medpipe_model(device: torch.device) -> nn.Module:
    return MedPipe3D(in_channels=3, num_classes=1).to(device)


def build_medvit_model(script_path: Path, size: int, device: torch.device) -> nn.Module:
    module = load_module_from_file(script_path, "medvit3d_nodule_train")
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{script_path} must expose build_model(config).")
    config = module.TrainConfig(img_root="", excel_path="", input_size=(size, size, size))
    if hasattr(config, "modalities"):
        config.modalities = ("T1WI", "T1WI+C", "T2WI")
    model = module.build_model(config)
    return model.to(device)


def train_one_model(
    model_name: str,
    model: nn.Module,
    loss_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_labels: Sequence[int],
    out_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainResult:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{model_name}_NoduleMNIST3D_{stamp}"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    pred_dir = out_dir / "predictions"
    for d in (ckpt_dir, metrics_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    criterion = make_loss_fn(loss_type, train_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    config_payload = vars(args).copy()
    config_payload.update({"model_name": model_name, "loss_type": loss_type, "created_at": stamp})
    (metrics_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_auc = -math.inf
    best_acc = -math.inf
    best_epoch = -1
    best_ckpt = ckpt_dir / "best_auc.pth"
    history: List[Dict[str, float]] = []
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch:03d}", leave=False)
        for x, y, _ids in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = output_to_logits(model(x))
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            bs = int(x.shape[0])
            running_loss += float(loss.detach().item()) * bs
            n_seen += bs
            progress.set_postfix(loss=running_loss / max(1, n_seen))

        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_type, device)
        val_auc = float(val_metrics["auc"])
        val_acc = float(val_metrics["accuracy"])
        train_loss = running_loss / max(1, n_seen)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "val_acc": val_acc})
        print(f"[{model_name}] epoch={epoch:03d} loss={train_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")

        score_auc = val_auc if not math.isnan(val_auc) else -math.inf
        improved = score_auc > best_auc + 1e-6 or (abs(score_auc - best_auc) <= 1e-6 and val_acc > best_acc)
        if improved:
            best_auc = score_auc
            best_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "args": config_payload,
                },
                best_ckpt,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[{model_name}] early stop at epoch {epoch}")
                break

        (metrics_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best_ckpt.exists():
        load_state_flexible(model, best_ckpt)
    test_metrics = evaluate(model, test_loader, loss_type, device)
    save_predictions_csv(test_metrics["predictions"], pred_dir / "test_predictions.csv")
    (metrics_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "model_name": model_name, "loss_type": loss_type}, ckpt_dir / "last.pth")

    result = TrainResult(
        model_name=model_name,
        out_dir=str(out_dir),
        best_ckpt=str(best_ckpt),
        best_epoch=int(best_epoch),
        best_val_auc=float(best_auc),
        best_val_acc=float(best_acc),
        test_auc=float(test_metrics["auc"]),
        test_acc=float(test_metrics["accuracy"]),
    )
    (metrics_dir / "summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"[DONE] {model_name}: best_epoch={best_epoch}, test_auc={result.test_auc:.4f}, test_acc={result.test_acc:.4f}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodule-root", default="data/public/NoduleMNIST3D")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--models", nargs="+", default=["final", "medpipe", "medvit"], choices=["final", "medpipe", "medvit"])
    parser.add_argument("--final-script", default="final.py")
    parser.add_argument("--medvit-script", default="train_M3T_transfer_7_3_paste.txt")
    parser.add_argument("--resume-final", default="")
    parser.add_argument("--resume-medpipe", default="")
    parser.add_argument("--resume-medvit", default="")
    parser.add_argument("--output-root", default="NoduleMNIST3D_Training_Output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Use light flips/noise for train split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds = NoduleTensorDataset("train", args.nodule_root, args.size, args.download, augment=args.augment)
    val_ds = NoduleTensorDataset("val", args.nodule_root, args.size, args.download, augment=False)
    test_ds = NoduleTensorDataset("test", args.nodule_root, args.size, args.download, augment=False)
    print(f"[INFO] NoduleMNIST3D counts: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    results: List[TrainResult] = []
    if "final" in args.models:
        print("[INFO] Building M3D-MILNet")
        model = build_final_model(Path(args.final_script), args.size, device)
        if args.resume_final:
            load_state_flexible(model, Path(args.resume_final))
        results.append(train_one_model("M3D-MILNet", model, "ce", train_loader, val_loader, test_loader, train_ds.labels, out_root, args, device))

    if "medpipe" in args.models:
        print("[INFO] Building MedPipe-style")
        model = build_medpipe_model(device)
        if args.resume_medpipe:
            load_state_flexible(model, Path(args.resume_medpipe))
        results.append(train_one_model("MedPipe", model, "bce", train_loader, val_loader, test_loader, train_ds.labels, out_root, args, device))

    if "medvit" in args.models:
        print("[INFO] Building MedViT-3D")
        model = build_medvit_model(Path(args.medvit_script), args.size, device)
        if args.resume_medvit:
            load_state_flexible(model, Path(args.resume_medvit))
        results.append(train_one_model("MedViT-3D", model, "bce", train_loader, val_loader, test_loader, train_ds.labels, out_root, args, device))

    summary_path = out_root / f"nodulemnist3d_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"[DONE] Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
