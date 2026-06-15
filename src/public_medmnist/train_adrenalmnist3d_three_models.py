#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train M3D-MILNet, Med3D and AMSNet on AdrenalMNIST3D.

This script is meant to create fair checkpoints before heatmap comparison.
All three models see the same AdrenalMNIST3D train/val/test splits.

AdrenalMNIST3D is single-channel. We repeat the same 3D volume into three
channels so the models originally written for three MRI modalities can run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def load_module_from_file(path: Path, module_name: str):
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


class AdrenalTensorDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: str,
        size: int,
        download: bool,
        augment: bool = False,
    ) -> None:
        try:
            from medmnist import AdrenalMNIST3D
        except ImportError as exc:
            raise ImportError("Please install medmnist first: pip install medmnist") from exc

        self.base = AdrenalMNIST3D(split=split, size=size, download=download, root=root)
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
    return lambda logits, y: F.binary_cross_entropy_with_logits(
        logits[:, 0],
        y.float(),
        pos_weight=pos_weight,
    )


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


def build_final_model(script_path: Path, size: int, device: torch.device) -> nn.Module:
    module = load_module_from_file(script_path, "final_adrenal_train")
    config = module.EnhancedConfig(None)
    config.DETECTOR_IMG_SIZE = (size, size, size)
    config.IN_CHANNELS = 3
    config.NUM_CLASSES = 2
    config.ROI_SIZE = min(int(config.ROI_SIZE), size)
    config.ROI_SUPPRESS_RADIUS = max(1, int(config.ROI_SIZE) // 2)
    config.device = device
    model = module.MacroToMicroDensityNet(
        module.EnhancedUnsupervisedTransformerDetector(config),
        config,
    )
    return model.to(device)


def build_med3d_model(script_path: Path, size: int, device: torch.device) -> nn.Module:
    module = load_module_from_file(script_path, "med3d_adrenal_train")
    config = module.TrainConfig(img_root="", excel_path="", input_size=(size, size, size))
    model = module.build_model(config)
    return model.to(device)


def build_amsnet_model(script_path: Path, device: torch.device) -> nn.Module:
    module = load_module_from_file(script_path, "amsnet_adrenal_train")
    model = module.AMSNet(in_channels=3, num_classes=1)
    return model.to(device)


def train_one_model(
    model_name: str,
    model: nn.Module,
    loss_type: str,
    loaders: Dict[str, DataLoader],
    train_labels: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainResult:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{model_name}_AdrenalMNIST3D_{stamp}"
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    metric_dir = out_dir / "metrics"
    for d in (ckpt_dir, pred_dir, metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    loss_fn = make_loss_fn(loss_type, train_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    best_epoch = 0
    best_metrics: Dict[str, object] = {"auc": float("nan"), "accuracy": 0.0}
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        pbar = tqdm(loaders["train"], desc=f"{model_name} epoch {epoch}/{args.epochs}", leave=False)
        for x, y, _ids in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = output_to_logits(model(x))
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            seen += batch_size
            pbar.set_postfix(loss=total_loss / max(1, seen))

        scheduler.step()
        train_loss = total_loss / max(1, seen)
        val_metrics = evaluate(model, loaders["val"], loss_type, device)
        val_auc = float(val_metrics["auc"])
        val_acc = float(val_metrics["accuracy"])
        score = val_auc if not np.isnan(val_auc) else val_acc
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"loss={train_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "val_metrics": {k: v for k, v in val_metrics.items() if k != "predictions"},
                },
                ckpt_dir / "best_auc.pth",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model_name": model_name,
            "loss_type": loss_type,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / "last.pth",
    )
    (metric_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best = torch.load(ckpt_dir / "best_auc.pth", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], loss_type, device)
    save_predictions_csv(test_metrics["predictions"], pred_dir / "test_predictions.csv")
    metric_payload = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_metrics["auc"]),
        "best_val_acc": float(best_metrics["accuracy"]),
        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
    }
    (metric_dir / "test_metrics.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
    print(
        f"[DONE] {model_name}: best_epoch={best_epoch}, "
        f"test_auc={metric_payload['test_auc']:.4f}, test_acc={metric_payload['test_acc']:.4f}"
    )

    return TrainResult(
        model_name=model_name,
        out_dir=str(out_dir),
        best_ckpt=str(ckpt_dir / "best_auc.pth"),
        best_epoch=best_epoch,
        best_val_auc=float(best_metrics["auc"]),
        best_val_acc=float(best_metrics["accuracy"]),
        test_auc=float(test_metrics["auc"]),
        test_acc=float(test_metrics["accuracy"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three models on AdrenalMNIST3D.")
    parser.add_argument("--adrenal-root", default="data/public/AdrenalMNIST3D")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--models", nargs="*", default=["final", "med3d", "amsnet"], choices=["final", "med3d", "amsnet"])
    parser.add_argument("--output-root", default="AdrenalMNIST3D_Training_Output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--final-script", default="final.py")
    parser.add_argument("--med3d-script", default="train_Med3D_ResNet_standalone_7_3.py")
    parser.add_argument("--amsnet-script", default="AMSNet_7_3_private_dataset.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = AdrenalTensorDataset(
        split="train",
        root=args.adrenal_root,
        size=args.size,
        download=args.download,
        augment=not args.no_augment,
    )
    val_ds = AdrenalTensorDataset(split="val", root=args.adrenal_root, size=args.size, download=args.download)
    test_ds = AdrenalTensorDataset(split="test", root=args.adrenal_root, size=args.size, download=args.download)
    print(
        "[INFO] AdrenalMNIST3D counts: "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    print(f"[INFO] Train class counts: {dict(zip(*np.unique(train_ds.labels, return_counts=True)))}")

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    results: List[TrainResult] = []
    for model_key in args.models:
        if model_key == "final":
            model = build_final_model(Path(args.final_script), args.size, device)
            result = train_one_model("M3D-MILNet", model, "ce", loaders, train_ds.labels, args, device)
        elif model_key == "med3d":
            model = build_med3d_model(Path(args.med3d_script), args.size, device)
            result = train_one_model("Med3D", model, "bce", loaders, train_ds.labels, args, device)
        elif model_key == "amsnet":
            model = build_amsnet_model(Path(args.amsnet_script), device)
            result = train_one_model("AMSNet", model, "bce", loaders, train_ds.labels, args, device)
        else:
            raise ValueError(model_key)
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = Path(args.output_root) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Training summary: {summary_path.resolve()}")
    for item in results:
        print(f"[CKPT] {item.model_name}: {item.best_ckpt}")


if __name__ == "__main__":
    main()
