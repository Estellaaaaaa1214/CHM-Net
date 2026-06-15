#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared stratified 5-fold cross-validation pipeline for private 3D MRI classifiers.

This file is the 5-fold companion of common_3d_train_7_3.py. It reuses the
same dataset, preprocessing, metric functions, model-output adapters, and
checkpoint-loading helpers, but replaces the single 7:3 split with:

    outer StratifiedKFold test split
    + inner validation split for early stopping/model selection

All baseline models should call run_5fold_training(...) so they share exactly
the same patient folds and evaluation protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_3d_train_7_3 import (
    Patient3DDataset,
    Safe3DClassifier,
    SegmentationOutputToLogit,
    TrainConfig,
    autocast_ctx,
    build_optimizer,
    compute_metrics,
    extract_state_dict,
    load_label_table,
    make_scaler,
    output_to_logits,
    seed_everything,
    set_finetune_trainable,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_input_size(text: str) -> Tuple[int, int, int]:
    dims = tuple(int(x.strip()) for x in text.split(","))
    if len(dims) != 3:
        raise ValueError("--input-size must be D,H,W, for example 64,64,64.")
    return dims  # type: ignore[return-value]


def safe_auc_value(metrics: Dict[str, object]) -> float:
    value = metrics.get("auc", float("nan"))
    if value is None:
        return -1.0
    value = float(value)
    if math.isnan(value):
        return -1.0
    return value


def selection_score(metrics: Dict[str, object], val_loss: float, metric_name: str) -> float:
    if metric_name == "loss":
        return -float(val_loss)
    if metric_name == "auc":
        return safe_auc_value(metrics)
    return float(metrics.get(metric_name, 0.0))


def load_pretrained_weights_safe(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
) -> Dict[str, object]:
    """PyTorch-2.6-safe checkpoint loader for optional transfer learning."""

    if not pretrained_path:
        return {"loaded": False, "path": "", "matched": 0, "skipped": 0}

    ckpt_path = Path(pretrained_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_state = extract_state_dict(checkpoint)
    model_state = model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in raw_state.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "net.", "backbone."):
            while clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
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

    print(f"[INFO] Transfer checkpoint loaded: {ckpt_path}")
    print(f"[INFO] Matched parameters: {matched_count}; skipped parameters: {skipped_count}")
    return {
        "loaded": True,
        "path": str(ckpt_path),
        "matched": matched_count,
        "skipped": skipped_count,
        "strict": strict,
    }


@torch.no_grad()
def evaluate_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], float]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    patient_ids_all: List[str] = []
    losses: List[float] = []

    for images, labels, patient_ids in tqdm(loader, desc="Evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = output_to_logits(model(images))
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
        y_prob.extend(probs)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        patient_ids_all.extend([str(pid) for pid in patient_ids])

    metrics = compute_metrics(y_true, y_prob, threshold)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    metrics["loss"] = avg_loss
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
    return metrics, predictions, avg_loss


def train_one_epoch_5fold(
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

    train_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    return float(np.mean(losses)) if losses else 0.0, safe_auc_value(train_metrics)


def make_fold_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        Patient3DDataset(train_df, config.img_root, config, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        Patient3DDataset(val_df, config.img_root, config, train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        Patient3DDataset(test_df, config.img_root, config, train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def make_fold_dir(output_root: Path, run_name: str, model_name: str, fold_idx: int) -> Path:
    safe_model = model_name.replace(" ", "_")
    out_dir = output_root / run_name / safe_model / f"fold_{fold_idx + 1}"
    for sub in ["checkpoints", "metrics", "predictions"]:
        ensure_dir(out_dir / sub)
    return out_dir


def train_fold(
    model_name: str,
    build_model: Callable[[TrainConfig], nn.Module],
    config: TrainConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: Path,
    run_name: str,
    fold_idx: int,
    n_splits: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    selection_metric: str,
) -> Dict[str, object]:
    print("\n" + "=" * 80)
    print(f"[{model_name}] Fold {fold_idx + 1}/{n_splits}")
    print("=" * 80)
    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Val class distribution: {val_df['label'].value_counts().to_dict()}")
    print(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")

    out_dir = make_fold_dir(output_root, run_name, model_name, fold_idx)
    save_json(
        out_dir / "metrics" / "split_summary.json",
        {
            "model_name": model_name,
            "fold": fold_idx + 1,
            "train_patients": train_df["patient_id"].tolist(),
            "val_patients": val_df["patient_id"].tolist(),
            "test_patients": test_df["patient_id"].tolist(),
            "train_distribution": train_df["label"].value_counts().to_dict(),
            "val_distribution": val_df["label"].value_counts().to_dict(),
            "test_distribution": test_df["label"].value_counts().to_dict(),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    transfer_info = load_pretrained_weights_safe(
        model,
        config.pretrained_path,
        config.strict_load,
    )

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
    print(f"[INFO] Dry-run OK: {tuple(dummy_logits.shape)}")

    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    )
    optimizer = build_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = make_scaler(device, config.use_amp)

    train_loader, val_loader, test_loader = make_fold_loaders(train_df, val_df, test_df, config)
    best_score = -float("inf")
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, object]] = None
    best_path = out_dir / "checkpoints" / "best_model.pth"
    early_counter = 0
    history: List[Dict[str, float]] = []
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        if config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
            print(f"[INFO] Freeze warm-up finished at epoch {epoch - 1}. Unfreezing all layers.")
            set_finetune_trainable(model, freeze_backbone=False)
            optimizer = build_optimizer(model, config)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - epoch + 1, 1),
            )

        train_loss, train_auc = train_one_epoch_5fold(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config.use_amp,
        )
        val_metrics, _val_predictions, val_loss = evaluate_with_loss(
            model,
            val_loader,
            criterion,
            device,
            config.threshold,
        )
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        score = selection_score(val_metrics, val_loss, selection_metric)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_auc": float(train_auc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_auc": safe_auc_value(val_metrics),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_specificity": float(val_metrics.get("specificity", 0.0)),
            "val_f1_score": float(val_metrics.get("f1_score", 0.0)),
            "val_mcc": float(val_metrics.get("mcc", 0.0)),
            "selection_score": float(score),
            "learning_rate": lr_now,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics" / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={row['val_auc']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1_score']:.4f}"
        )

        if score > best_score + early_stop_min_delta:
            best_score = score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            early_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "model_name": model_name,
                    "fold": fold_idx + 1,
                    "selection_metric": selection_metric,
                    "selection_score": score,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"[INFO] Saved best fold model: {selection_metric}={score:.4f}")
        else:
            early_counter += 1
            print(f"[INFO] No improvement: {early_counter}/{early_stop_patience}")
            if early_counter >= early_stop_patience:
                print(f"[INFO] Early stopped at epoch {epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_predictions, test_loss = evaluate_with_loss(
        model,
        test_loader,
        criterion,
        device,
        config.threshold,
    )
    test_metrics["loss"] = float(test_loss)
    save_json(out_dir / "predictions" / "test_predictions.json", test_predictions)

    result = {
        "model_name": model_name,
        "fold": fold_idx + 1,
        "output_dir": str(out_dir),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_selection_score": float(best_score),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "transfer_learning": transfer_info,
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "time_seconds": float(time.time() - start),
    }
    save_json(out_dir / "metrics" / "fold_result.json", result)
    print(f"[RESULT] {model_name} fold {fold_idx + 1}: AUC={test_metrics.get('auc', 0):.4f}, ACC={test_metrics.get('accuracy', 0):.4f}")
    return result


def summarize_model_results(results: List[Dict[str, object]], model_summary_dir: Path) -> pd.DataFrame:
    ensure_dir(model_summary_dir)
    metric_keys = [
        "accuracy",
        "auc",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "mcc",
        "loss",
    ]
    rows = []
    for item in results:
        metrics = item["test_metrics"]  # type: ignore[index]
        row = {
            "model_name": item["model_name"],
            "fold": item["fold"],
            "best_epoch": item["best_epoch"],
            "output_dir": item["output_dir"],
        }
        for key in metric_keys:
            row[key] = float(metrics.get(key, 0.0))  # type: ignore[union-attr]
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(model_summary_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for key in metric_keys:
        values = fold_df[key].astype(float).values
        summary_rows.append(
            {
                "model_name": str(results[0]["model_name"]) if results else "",
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(model_summary_dir / "mean_std.csv", index=False, encoding="utf-8-sig")
    save_json(model_summary_dir / "results.json", {"fold_results": results, "mean_std": summary_rows})
    return summary_df


def run_5fold_training(
    model_name: str,
    build_model: Callable[[TrainConfig], nn.Module],
    config: TrainConfig,
    n_splits: int = 5,
    inner_val_size: float = 0.20,
    early_stop_patience: int = 15,
    early_stop_min_delta: float = 0.0,
    selection_metric: str = "auc",
    run_name: Optional[str] = None,
    shared_splits_path: str = "",
) -> Dict[str, object]:
    if not config.img_root or not config.excel_path:
        raise ValueError("Please set --img-root and --excel-path.")
    if selection_metric not in {"auc", "accuracy", "f1_score", "mcc", "loss"}:
        raise ValueError("selection_metric must be one of auc, accuracy, f1_score, mcc, loss.")

    seed_everything(config.random_state)
    output_root = Path(config.output_root)
    run_name = run_name or f"private_5fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = ensure_dir(output_root / run_name)
    summary_dir = ensure_dir(run_root / "_summary" / model_name.replace(" ", "_"))
    save_json(summary_dir / "config.json", asdict(config))

    df = load_label_table(config)
    if len(df) < n_splits * 2:
        raise RuntimeError("Too few valid patients for 5-fold cross-validation.")

    if shared_splits_path and Path(shared_splits_path).exists():
        with Path(shared_splits_path).open("r", encoding="utf-8") as f:
            split_payload = json.load(f)
        folds = split_payload["folds"]
        print(f"[INFO] Loaded shared splits from: {shared_splits_path}")
    else:
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        patient_ids = df["patient_id"].astype(str).tolist()
        labels = df["label"].astype(int).values
        folds = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            if inner_val_size > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=inner_val_size,
                    stratify=train_val_df["label"],
                    random_state=config.random_state + fold_idx,
                )
            else:
                train_df = train_val_df
                val_df = test_df
            folds.append(
                {
                    "fold": fold_idx + 1,
                    "train_ids": train_df["patient_id"].astype(str).tolist(),
                    "val_ids": val_df["patient_id"].astype(str).tolist(),
                    "test_ids": test_df["patient_id"].astype(str).tolist(),
                }
            )
        split_payload = {
            "n_splits": n_splits,
            "inner_val_size": inner_val_size,
            "random_state": config.random_state,
            "folds": folds,
        }
        split_path = Path(shared_splits_path) if shared_splits_path else run_root / "cv_splits.json"
        save_json(split_path, split_payload)
        print(f"[INFO] Saved shared splits to: {split_path}")

    id_to_row = df.set_index("patient_id", drop=False)
    results = []
    for fold_idx, fold in enumerate(folds):
        fold_seed = config.random_state + fold_idx
        seed_everything(fold_seed)
        train_df = id_to_row.loc[fold["train_ids"]].reset_index(drop=True)
        val_df = id_to_row.loc[fold["val_ids"]].reset_index(drop=True)
        test_df = id_to_row.loc[fold["test_ids"]].reset_index(drop=True)
        result = train_fold(
            model_name=model_name,
            build_model=build_model,
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_root=output_root,
            run_name=run_name,
            fold_idx=fold_idx,
            n_splits=n_splits,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            selection_metric=selection_metric,
        )
        results.append(result)
        save_json(summary_dir / "partial_results.json", results)

    summary_df = summarize_model_results(results, summary_dir)
    print("\n" + "=" * 80)
    print(f"[SUMMARY] {model_name}")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f}")
    return {
        "model_name": model_name,
        "summary_dir": str(summary_dir),
        "fold_results": results,
        "summary": summary_df.to_dict(orient="records"),
    }


def parse_5fold_args(default_config: Optional[TrainConfig] = None) -> Tuple[TrainConfig, argparse.Namespace]:
    defaults = default_config or TrainConfig()
    parser = argparse.ArgumentParser(description="Run stratified 5-fold CV for one 3D classifier.")
    parser.add_argument("--img-root", default=defaults.img_root)
    parser.add_argument("--excel-path", default=defaults.excel_path)
    parser.add_argument("--id-col", default=defaults.id_col)
    parser.add_argument("--label-col", default=defaults.label_col)
    parser.add_argument("--output-root", default="Model_5Fold_Output")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--input-size", default=",".join(map(str, defaults.input_size)))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-val-size", type=float, default=0.20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="auc", choices=["auc", "accuracy", "f1_score", "mcc", "loss"])
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", default=defaults.pretrained_path)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=defaults.freeze_epochs)
    parser.add_argument("--head-lr-mult", type=float, default=defaults.head_lr_mult)
    parser.add_argument("--shared-splits", default="")
    args = parser.parse_args()

    config = TrainConfig(
        img_root=args.img_root,
        excel_path=args.excel_path,
        id_col=args.id_col,
        label_col=args.label_col,
        output_root=args.output_root,
        input_size=parse_input_size(args.input_size),
        random_state=args.random_state,
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
    return config, args
