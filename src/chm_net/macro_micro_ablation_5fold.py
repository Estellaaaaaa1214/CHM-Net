#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-fold ablation study for the Macro-to-Micro Density-Aware MIL Network.

This script runs all ablation variants on identical stratified 5-fold splits:

    full
    no_detector_roi
    no_gating
    no_mil
    no_macro
    single_view

Protocol:
    - same patient-level outer folds for every variant
    - inner validation split from each training fold for early stopping
    - detector is loaded once per fold and frozen
    - classifier/ablation heads are retrained from scratch in every fold
    - results are reported as fold metrics and mean +/- std

Put this file in the same server directory as:
    macro_micro_density_mil.py
    macro_micro_ablation.py
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import macro_micro_ablation as ab

base = ab.base


def import_base_module(base_file: str):
    candidates = []
    if base_file:
        candidates.append(base_file)
    candidates.extend(["macro_micro_density_mil.py", "final.py"])

    for path in candidates:
        if path and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("macro_micro_base", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"Loaded base model code from: {path}")
            return module
    raise FileNotFoundError(
        "Could not find macro_micro_density_mil.py or final.py. "
        "Put the model script beside this file, or pass --base-file."
    )


def patch_cached_dmm_loader(base_module, cache_dir: str) -> None:
    if not cache_dir:
        return
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"DMM cache dir not found: {cache_dir}")

    original_loader = base_module.load_patient_3d

    def cached_load_patient_3d(pid, config):
        patient_id = str(pid).strip()
        cache_path = os.path.join(cache_dir, f"{patient_id}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path, mmap_mode=None).astype(np.float32)
        print(f"[WARN] DMM cache miss for patient {patient_id}; falling back to NIfTI loading.")
        return original_loader(pid, config)

    base_module.load_patient_3d = cached_load_patient_3d
    print(f"Using preprocessed DMM tensor cache: {cache_dir}")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_base_args(args):
    return SimpleNamespace(
        img_root=args.img_root,
        excel_path=args.excel_path,
        id_col=args.id_col,
        label_col=args.label_col,
        output_dir=args.output_dir,
        detector_ckpt=args.detector_ckpt,
        resume_ckpt="",
        skip_detector_pretrain=False,
        detector_epochs=None,
        classifier_epochs=args.classifier_epochs,
        num_rois=args.num_rois,
        roi_size=args.roi_size,
        classifier_batch_size=args.classifier_batch_size,
        train_ratio=None,
        freeze_detector_epochs=10**9,
        no_detector_finetune=True,
    )


def build_fold_config(args, variant: str, fold_idx: int, run_name: str):
    config = base.EnhancedConfig(make_base_args(args))
    config.EXPERIMENT_NAME = f"{run_name}_{variant}_fold{fold_idx + 1}"
    config.exp_dir = ensure_dir(os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME))
    for d in ["checkpoints", "logs", "metrics", "predictions", "features", "visualizations"]:
        ensure_dir(os.path.join(config.exp_dir, d))
    config.CLASSIFIER_EPOCHS = args.classifier_epochs
    config.EARLY_STOP_PATIENCE = args.early_stop_patience
    config.FREEZE_DETECTOR_EPOCHS = 10**9
    config.FINETUNE_DETECTOR_LR = 0.0
    config.DETECTOR_CKPT = args.detector_ckpt
    if args.classifier_lr is not None:
        config.CLASSIFIER_LR = args.classifier_lr
    return config


def collect_labeled_data(config) -> Tuple[pd.DataFrame, List[str], Dict[str, int], List[int], Dict[str, object]]:
    df = pd.read_excel(config.EXCEL_PATH)
    _, valid_patients, labeled_patients, class_dist = base.collect_valid_patients(config, df)
    labels_by_pid: Dict[str, int] = {}
    for pid in labeled_patients:
        pid_str = str(pid).strip()
        values = df[df[config.ID_COL].astype(str).str.strip() == pid_str][config.CLS_COL].values
        if len(values) > 0 and not pd.isna(values[0]):
            label = int(values[0])
            if label in [0, 1]:
                labels_by_pid[pid_str] = label

    labeled_patients = [str(pid).strip() for pid in labeled_patients if str(pid).strip() in labels_by_pid]
    labels = [labels_by_pid[pid] for pid in labeled_patients]
    data_info = {
        "valid_patients": len(valid_patients),
        "labeled_patients": len(labeled_patients),
        "class_distribution": {
            "low_density": int(np.bincount(labels, minlength=2)[0]),
            "high_density": int(np.bincount(labels, minlength=2)[1]),
        },
    }
    return df, labeled_patients, labels_by_pid, labels, data_info


def make_loaders(config, train_ids, val_ids, test_ids, labels_by_pid):
    train_ds = base.Supervised3DVolumeDataset(train_ids, labels_by_pid, config, augment=True, mode="train")
    val_ds = base.Supervised3DVolumeDataset(val_ids, labels_by_pid, config, augment=False, mode="val")
    test_ds = base.Supervised3DVolumeDataset(test_ids, labels_by_pid, config, augment=False, mode="test")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader


def load_frozen_detector(config, ckpt_path: str):
    detector = base.EnhancedUnsupervisedTransformerDetector(config).to(config.device)
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    detector_state = {}
    has_detector_prefix = False
    for key, value in state_dict.items():
        if str(key).startswith("detector."):
            has_detector_prefix = True
            detector_state[str(key)[len("detector."):]] = value

    if has_detector_prefix:
        detector.load_state_dict(detector_state, strict=True)
        print(f"Loaded detector.* weights from full checkpoint: {ckpt_path}")
    else:
        detector.load_state_dict(state_dict, strict=True)
        print(f"Loaded detector checkpoint: {ckpt_path}")

    base.set_module_trainable(detector, False)
    detector.eval()
    return detector


def metric_score(metrics: Dict[str, object], metric_name: str) -> float:
    value = metrics.get(metric_name, 0.0)
    if value is None:
        return -1.0
    value = float(value)
    if np.isnan(value):
        return -1.0
    return value


def train_variant_fold(
    args,
    variant: str,
    fold_idx: int,
    run_name: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    labels_by_pid: Dict[str, int],
) -> Dict[str, object]:
    print("\n" + "=" * 90)
    print(f"Ablation 5-fold | variant={variant} | fold={fold_idx + 1}/{args.n_splits}")
    print(ab.ABLATION_DESCRIPTIONS[variant])
    print("=" * 90)

    set_seed(args.seed + fold_idx)
    config = build_fold_config(args, variant, fold_idx, run_name)
    train_labels = [labels_by_pid[pid] for pid in train_ids]
    val_labels = [labels_by_pid[pid] for pid in val_ids]
    test_labels = [labels_by_pid[pid] for pid in test_ids]
    print(f"Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    print("Train class distribution:", np.bincount(train_labels, minlength=2).tolist())
    print("Val class distribution:", np.bincount(val_labels, minlength=2).tolist())
    print("Test class distribution:", np.bincount(test_labels, minlength=2).tolist())

    save_json(
        os.path.join(config.exp_dir, "metrics", "fold_split.json"),
        {
            "variant": variant,
            "fold": fold_idx + 1,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "train_class_distribution": np.bincount(train_labels, minlength=2).tolist(),
            "val_class_distribution": np.bincount(val_labels, minlength=2).tolist(),
            "test_class_distribution": np.bincount(test_labels, minlength=2).tolist(),
        },
    )

    detector = load_frozen_detector(config, args.detector_ckpt)
    model = ab.AblationMacroToMicroNet(detector, config, variant).to(config.device)
    base.set_module_trainable(model.detector, False)
    model.detector.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_loader, val_loader, test_loader = make_loaders(config, train_ids, val_ids, test_ids, labels_by_pid)
    criterion = nn.CrossEntropyLoss(
        weight=base.compute_class_weights(train_labels, config.device),
        label_smoothing=config.LABEL_SMOOTHING,
    )
    optimizer = ab.optimizer_for_classifier(model, config)
    warmup = ab.WarmupScheduler(optimizer, config.WARMUP_EPOCHS, config.CLASSIFIER_LR)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        verbose=True,
        min_lr=1e-7,
    )

    best_score = -1.0
    best_metrics = None
    best_epoch = 0
    no_improve = 0
    history = []
    best_path = os.path.join(config.exp_dir, "checkpoints", "ablation_fold_best.pth")
    start_time = time.time()

    for epoch in range(config.CLASSIFIER_EPOCHS):
        print(f"\n[{variant} fold {fold_idx + 1}] Epoch {epoch + 1}/{config.CLASSIFIER_EPOCHS}")
        lr = warmup.step(epoch) if epoch < config.WARMUP_EPOCHS else optimizer.param_groups[0]["lr"]
        model.train()
        model.detector.eval()

        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"{variant} fold{fold_idx + 1} train")
        for images, labels, _ in pbar:
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = base.attention_smoothness_loss(outputs["heatmap"])
            loss = cls_loss + config.SPARSITY_WEIGHT * sparse_loss + config.SMOOTHNESS_WEIGHT * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            train_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / max(1, total):.2f}%")

        train_loss /= max(1, len(train_loader))
        train_acc = correct / max(1, total)
        val_metrics, val_loss = ab.evaluate(model, val_loader, criterion, config, save_predictions=False)
        score = metric_score(val_metrics, args.selection_metric)
        if epoch >= config.WARMUP_EPOCHS:
            plateau.step(score)

        row = {
            "epoch": epoch + 1,
            "variant": variant,
            "fold": fold_idx + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["accuracy"]),
            "val_auc": float(val_metrics["auc"]),
            "val_f1": float(val_metrics["f1_score"]),
            "val_recall": float(val_metrics["recall"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_balanced_acc": float(val_metrics["balanced_accuracy"]),
            "val_mcc": float(val_metrics["mcc"]),
            "selection_score": float(score),
            "lr": float(lr),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(config.exp_dir, "metrics", "history.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_metrics['accuracy']:.4f}, "
            f"AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1_score']:.4f}"
        )

        if score > best_score + args.early_stop_min_delta:
            best_score = float(score)
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "variant": variant,
                    "fold": fold_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "selection_score": best_score,
                    "config": config.__dict__,
                    "detector_ckpt": args.detector_ckpt,
                },
                best_path,
            )
            print(f"Saved best [{variant} fold {fold_idx + 1}] by {args.selection_metric}={best_score:.4f}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{config.EARLY_STOP_PATIENCE}")
            if no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopped [{variant} fold {fold_idx + 1}] at epoch {epoch + 1}")
                break

    checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_loss = ab.evaluate(model, test_loader, criterion, config, save_predictions=True)
    test_metrics["loss"] = float(test_loss)

    result = {
        "variant": variant,
        "description": ab.ABLATION_DESCRIPTIONS[variant],
        "fold": fold_idx + 1,
        "exp_dir": config.exp_dir,
        "detector_ckpt": args.detector_ckpt,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "time_seconds": time.time() - start_time,
    }
    save_json(os.path.join(config.exp_dir, "metrics", "ablation_fold_result.json"), result)
    print(
        f"Finished {variant} fold {fold_idx + 1}: "
        f"test AUC={test_metrics['auc']:.4f}, acc={test_metrics['accuracy']:.4f}"
    )
    return result


def build_or_load_splits(args, labeled_patients: List[str], labels_by_pid: Dict[str, int], summary_dir: str):
    if args.shared_splits and os.path.exists(args.shared_splits):
        with open(args.shared_splits, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"Loaded shared splits from: {args.shared_splits}")
        return payload["folds"]

    patients = np.asarray(labeled_patients)
    labels = np.asarray([labels_by_pid[pid] for pid in labeled_patients], dtype=np.int64)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patients, labels)):
        train_val_ids = patients[train_val_idx].tolist()
        test_ids = patients[test_idx].tolist()
        train_val_labels = [labels_by_pid[pid] for pid in train_val_ids]
        if args.inner_val_ratio > 0:
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=args.inner_val_ratio,
                stratify=train_val_labels,
                random_state=args.seed + fold_idx,
            )
        else:
            train_ids = train_val_ids
            val_ids = test_ids
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_ids": list(train_ids),
                "val_ids": list(val_ids),
                "test_ids": list(test_ids),
            }
        )

    payload = {
        "n_splits": args.n_splits,
        "inner_val_ratio": args.inner_val_ratio,
        "seed": args.seed,
        "folds": folds,
    }
    split_path = args.shared_splits or os.path.join(summary_dir, "cv_splits.json")
    save_json(split_path, payload)
    print(f"Saved shared splits to: {split_path}")
    return folds


def summarize_results(all_results: List[Dict[str, object]], summary_dir: str) -> None:
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "loss",
    ]

    fold_rows = []
    for item in all_results:
        metrics = item["test_metrics"]
        row = {
            "variant": item["variant"],
            "fold": item["fold"],
            "best_epoch": item["best_epoch"],
            "exp_dir": item["exp_dir"],
        }
        for key in metric_keys:
            row[key] = float(metrics.get(key, 0.0))
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(summary_dir, "ablation_5fold_fold_metrics.csv"), index=False, encoding="utf-8-sig")

    summary_rows = []
    for variant, group in fold_df.groupby("variant"):
        row = {"variant": variant, "description": ab.ABLATION_DESCRIPTIONS.get(variant, "")}
        for key in metric_keys:
            values = group[key].astype(float).values
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{key}_min"] = float(np.min(values))
            row[f"{key}_max"] = float(np.max(values))
            row[f"{key}_mean_std"] = f"{row[f'{key}_mean']:.4f} +/- {row[f'{key}_std']:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if "auc_mean" in summary_df.columns:
        summary_df = summary_df.sort_values("auc_mean", ascending=False)
    summary_df.to_csv(os.path.join(summary_dir, "ablation_5fold_mean_std.csv"), index=False, encoding="utf-8-sig")

    compact_cols = [
        "variant",
        "accuracy_mean_std",
        "auc_mean_std",
        "precision_mean_std",
        "recall_mean_std",
        "specificity_mean_std",
        "f1_score_mean_std",
        "mcc_mean_std",
    ]
    compact_cols = [c for c in compact_cols if c in summary_df.columns]
    summary_df[compact_cols].to_csv(
        os.path.join(summary_dir, "ablation_5fold_compact_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        os.path.join(summary_dir, "ablation_5fold_results.json"),
        {
            "fold_results": all_results,
            "summary": summary_rows,
        },
    )

    print("\n" + "=" * 90)
    print("Ablation 5-fold summary")
    print("=" * 90)
    print(summary_df[compact_cols].to_string(index=False))
    print(f"Summary dir: {summary_dir}")


def parse_args():
    parser = argparse.ArgumentParser("Five-fold ablation study for Macro-to-Micro MIL")
    parser.add_argument("--base-file", type=str, default="", help="Path to final.py or macro_micro_density_mil.py.")
    parser.add_argument("--img-root", type=str, default="")
    parser.add_argument("--excel-path", type=str, default="")
    parser.add_argument("--id-col", type=str, default="case_id")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--output-dir", type=str, default="MacroMicro_Ablation_5Fold_Output")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--detector-ckpt", type=str, required=True)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-batch-size", type=int, default=2)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--num-rois", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-val-ratio", type=float, default=0.20)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-metric", type=str, default="auc", choices=["auc", "accuracy", "f1_score", "balanced_accuracy", "mcc"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shared-splits", type=str, default="")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional directory containing preprocessed patient .npy tensors.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_detector_roi,no_gating,no_mil,no_macro,single_view",
        help="Comma-separated variants. Choices: " + ",".join(ab.ABLATION_DESCRIPTIONS.keys()),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global base
    base = import_base_module(args.base_file)
    ab.base = base
    patch_cached_dmm_loader(base, args.cache_dir)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ab.ABLATION_DESCRIPTIONS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    run_name = args.run_name or f"ablation5fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_dir = ensure_dir(os.path.join(args.output_dir, f"{run_name}_summary"))
    set_seed(args.seed)

    probe_config = build_fold_config(args, "full", 0, run_name + "_probe")
    _, labeled_patients, labels_by_pid, labels, data_info = collect_labeled_data(probe_config)
    save_json(os.path.join(summary_dir, "data_info.json"), data_info)
    if len(labeled_patients) < args.n_splits * 2:
        raise RuntimeError("Too few labeled patients for five-fold ablation.")

    folds = build_or_load_splits(args, labeled_patients, labels_by_pid, summary_dir)
    all_results: List[Dict[str, object]] = []
    start = time.time()

    for variant in variants:
        for fold_idx, fold in enumerate(folds):
            result = train_variant_fold(
                args=args,
                variant=variant,
                fold_idx=fold_idx,
                run_name=run_name,
                train_ids=[str(x) for x in fold["train_ids"]],
                val_ids=[str(x) for x in fold["val_ids"]],
                test_ids=[str(x) for x in fold["test_ids"]],
                labels_by_pid=labels_by_pid,
            )
            all_results.append(result)
            save_json(os.path.join(summary_dir, "partial_results.json"), all_results)
            summarize_results(all_results, summary_dir)

    print(f"Total time: {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
