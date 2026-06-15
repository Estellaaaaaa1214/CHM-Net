#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five-fold cross-validation runner for the Macro-to-Micro Density-Aware MIL Network.

Keep this file in the same directory as final.py, or pass --base-file to point to
the model script. The model architecture, datasets, losses, and metric functions
are reused from final.py; this file only changes the experimental protocol from
a single 7:3 split to stratified K-fold validation.

Important:
    A locked full classifier checkpoint must not be used as a real 5-fold result,
    because it has already been trained on a previous split. For a practical
    classifier-only 5-fold run, reuse only the detector weights and retrain the
    classifier in every fold.
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


def import_base_module(base_file):
    candidates = []
    if base_file:
        candidates.append(base_file)
    candidates.extend(["final.py", "macro_micro_density_mil.py"])

    for path in candidates:
        if path and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("macro_micro_base", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"Loaded base model code from: {path}")
            return module
    raise FileNotFoundError(
        "Could not find final.py or macro_micro_density_mil.py. "
        "Put final_5fold.py beside final.py, or pass --base-file."
    )


def patch_cached_dmm_loader(base, cache_dir):
    """Redirect DMM-Net volume loading to preprocessed .npy tensors."""
    if not cache_dir:
        return

    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"DMM cache dir not found: {cache_dir}")

    original_loader = base.load_patient_3d

    def cached_load_patient_3d(pid, config):
        patient_id = str(pid).strip()
        cache_path = os.path.join(cache_dir, f"{patient_id}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path, mmap_mode=None).astype(np.float32)
        print(f"[WARN] DMM cache miss for patient {patient_id}; falling back to NIfTI loading.")
        return original_loader(pid, config)

    base.load_patient_3d = cached_load_patient_3d
    print(f"Using preprocessed DMM tensor cache: {cache_dir}")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


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
        detector_epochs=args.detector_epochs,
        classifier_epochs=args.classifier_epochs,
        num_rois=args.num_rois,
        roi_size=args.roi_size,
        classifier_batch_size=args.classifier_batch_size,
        train_ratio=None,
        freeze_detector_epochs=10**9,
        no_detector_finetune=True,
    )


def build_fold_config(base, args, fold_idx):
    config = base.EnhancedConfig(make_base_args(args))
    run_stamp = args.run_name or f"MacroMicro_5fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.EXPERIMENT_NAME = f"{run_stamp}_fold{fold_idx + 1}"
    config.exp_dir = ensure_dir(os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME))

    for d in ["checkpoints", "logs", "metrics", "predictions", "features", "visualizations"]:
        ensure_dir(os.path.join(config.exp_dir, d))

    config.CLASSIFIER_EPOCHS = args.classifier_epochs
    config.EARLY_STOP_PATIENCE = args.early_stop_patience
    config.FREEZE_DETECTOR_EPOCHS = 10**9
    config.FINETUNE_DETECTOR_LR = 0.0
    config.DETECTOR_CKPT = args.detector_ckpt
    if args.detector_batch_size is not None:
        config.DETECTOR_BATCH_SIZE = args.detector_batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    return config


def collect_labeled_patients(base, config):
    df = pd.read_excel(config.EXCEL_PATH)
    all_patients, valid_patients, labeled_patients, class_dist = base.collect_valid_patients(
        config,
        df,
    )

    labels_by_pid = {}
    for pid in labeled_patients:
        pid_str = str(pid).strip()
        values = df[df[config.ID_COL].astype(str).str.strip() == pid_str][config.CLS_COL].values
        if len(values) > 0 and not pd.isna(values[0]):
            label = int(values[0])
            if label in [0, 1]:
                labels_by_pid[pid_str] = label

    labeled_patients = [pid for pid in labeled_patients if pid in labels_by_pid]
    labels = [labels_by_pid[pid] for pid in labeled_patients]
    data_info = {
        "total_patients": len(all_patients),
        "valid_patients": len(valid_patients),
        "labeled_patients": len(labeled_patients),
        "class_distribution": {
            "low_density": int(np.bincount(labels, minlength=2)[0]),
            "high_density": int(np.bincount(labels, minlength=2)[1]),
        },
    }
    return df, labeled_patients, labels_by_pid, labels, data_info


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def load_detector_from_any_checkpoint(base, config, ckpt_path):
    detector = base.EnhancedUnsupervisedTransformerDetector(config).to(config.device)
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    detector_state = {}
    has_detector_prefix = False
    for key, value in state_dict.items():
        if key.startswith("detector."):
            has_detector_prefix = True
            detector_state[key[len("detector."):]] = value

    if has_detector_prefix:
        detector.load_state_dict(detector_state, strict=True)
        print(f"Loaded detector weights extracted from full checkpoint: {ckpt_path}")
    else:
        detector.load_state_dict(state_dict, strict=True)
        print(f"Loaded detector checkpoint: {ckpt_path}")

    freeze_module(detector)
    return detector


def build_loaders(base, config, train_ids, val_ids, test_ids, labels_by_pid):
    train_dataset = base.Supervised3DVolumeDataset(
        train_ids,
        labels_by_pid,
        config,
        augment=True,
        mode="train",
    )
    val_dataset = base.Supervised3DVolumeDataset(
        val_ids,
        labels_by_pid,
        config,
        augment=False,
        mode="val",
    )
    test_dataset = base.Supervised3DVolumeDataset(
        test_ids,
        labels_by_pid,
        config,
        augment=False,
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader


def selection_score(metrics, metric_name):
    return float(metrics.get(metric_name, 0.0))


def train_one_fold(
    base,
    args,
    fold_idx,
    train_core_ids,
    val_ids,
    test_ids,
    labels_by_pid,
    detector_source_patients,
):
    print("\n" + "=" * 80)
    print(f"Fold {fold_idx + 1}/{args.n_splits}")
    print("=" * 80)

    set_seed(args.seed + fold_idx)
    config = build_fold_config(base, args, fold_idx)
    fold_start = time.time()

    train_labels = [labels_by_pid[pid] for pid in train_core_ids]
    val_labels = [labels_by_pid[pid] for pid in val_ids]
    test_labels = [labels_by_pid[pid] for pid in test_ids]
    print(f"Train: {len(train_core_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Train class distribution: {np.bincount(train_labels, minlength=2).tolist()}")
    print(f"Val class distribution: {np.bincount(val_labels, minlength=2).tolist()}")
    print(f"Test class distribution: {np.bincount(test_labels, minlength=2).tolist()}")

    split_info = {
        "fold": fold_idx + 1,
        "train_ids": list(train_core_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
        "train_class_distribution": np.bincount(train_labels, minlength=2).tolist(),
        "val_class_distribution": np.bincount(val_labels, minlength=2).tolist(),
        "test_class_distribution": np.bincount(test_labels, minlength=2).tolist(),
    }
    with open(os.path.join(config.exp_dir, "metrics", "fold_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    if args.train_detector_per_fold:
        print("Training detector inside this fold using only non-test patients.")
        detector, detector_history = base.train_enhanced_unsupervised_detector(
            config,
            detector_source_patients,
        )
        freeze_module(detector)
    else:
        if not args.detector_ckpt:
            raise ValueError(
                "Pass --detector-ckpt for practical 5-fold training, "
                "or use --train-detector-per-fold for strict full retraining."
            )
        detector = load_detector_from_any_checkpoint(base, config, args.detector_ckpt)
        detector_history = {
            "source": args.detector_ckpt,
            "mode": "loaded_and_frozen",
        }

    train_loader, val_loader, test_loader = build_loaders(
        base,
        config,
        train_core_ids,
        val_ids,
        test_ids,
        labels_by_pid,
    )

    model = base.MacroToMicroDensityNet(detector, config).to(config.device)
    freeze_module(model.detector)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    class_weights = base.compute_class_weights(train_labels, config.device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.CLASSIFIER_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    warmup_scheduler = base.EnhancedWarmupScheduler(
        optimizer,
        config.WARMUP_EPOCHS,
        config.CLASSIFIER_LR,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
    early_stop_counter = 0
    history = []
    best_path = os.path.join(config.exp_dir, "checkpoints", "fold_best.pth")

    for epoch in range(config.CLASSIFIER_EPOCHS):
        print(f"\nFold {fold_idx + 1} Classifier Epoch {epoch + 1}/{config.CLASSIFIER_EPOCHS}")
        if epoch < config.WARMUP_EPOCHS:
            current_lr = warmup_scheduler.step(epoch)
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        model.detector.eval()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"fold{fold_idx + 1} train")
        for images, labels_batch, _ in pbar:
            images = images.to(config.device, non_blocking=True)
            labels_batch = labels_batch.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels_batch)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = base.attention_smoothness_loss(outputs["heatmap"])
            loss = (
                cls_loss
                + config.SPARSITY_WEIGHT * sparse_loss
                + config.SMOOTHNESS_WEIGHT * smooth_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            train_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total += int(labels_batch.size(0))
            correct += int((preds == labels_batch).sum().item())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / max(1, total):.2f}%",
                }
            )

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        val_metrics, val_loss = base.evaluate_macro_micro_classifier(
            model,
            val_loader,
            criterion,
            config,
            save_predictions=False,
        )

        if epoch >= config.WARMUP_EPOCHS:
            plateau_scheduler.step(selection_score(val_metrics, args.selection_metric))

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "mcc",
        ]:
            row[f"val_{key}"] = float(val_metrics.get(key, 0.0))
        history.append(row)

        print(f"Train loss: {avg_train_loss:.4f}, acc: {train_acc:.4f}")
        print(
            "Val loss: "
            f"{val_loss:.4f}, acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc']:.4f}"
        )

        score = selection_score(val_metrics, args.selection_metric)
        if score > best_score + args.early_stop_min_delta:
            best_score = score
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(
                {
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "selection_score": score,
                    "config": config.__dict__,
                },
                best_path,
            )
            print(f"Saved best fold model by {args.selection_metric}={score:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement: {early_stop_counter}/{config.EARLY_STOP_PATIENCE}")
            if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopped fold {fold_idx + 1} at epoch {epoch + 1}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(config.exp_dir, "metrics", "training_history.csv"), index=False)

    checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best fold model from epoch {checkpoint['epoch']}")

    test_metrics, test_loss = base.evaluate_macro_micro_classifier(
        model,
        test_loader,
        criterion,
        config,
        save_predictions=True,
    )
    test_metrics["loss"] = float(test_loss)

    result = {
        "fold": fold_idx + 1,
        "exp_dir": config.exp_dir,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "detector_history": detector_history,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "time_seconds": time.time() - fold_start,
    }
    with open(os.path.join(config.exp_dir, "metrics", "fold_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nFold test metrics:")
    for key in ["accuracy", "precision", "recall", "f1_score", "auc", "specificity", "mcc"]:
        print(f"  {key}: {test_metrics.get(key, 0):.4f}")
    return result


def summarize_results(results, summary_dir):
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

    rows = []
    for item in results:
        metrics = item["test_metrics"]
        row = {
            "fold": item["fold"],
            "best_epoch": item["best_epoch"],
            "exp_dir": item["exp_dir"],
        }
        for key in metric_keys:
            row[key] = float(metrics.get(key, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(summary_dir, "fold_metrics.csv"), index=False, encoding="utf-8-sig")

    summary_rows = []
    for key in metric_keys:
        values = df[key].astype(float).values
        summary_rows.append(
            {
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(summary_dir, "fivefold_mean_std.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(summary_dir, "fivefold_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold_results": results,
                "mean_std": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("Five-fold cross-validation summary")
    print("=" * 80)
    for item in summary_rows:
        print(f"{item['metric']}: {item['mean']:.4f} +/- {item['std']:.4f}")
    print(f"Summary dir: {summary_dir}")


def parse_args():
    parser = argparse.ArgumentParser("Five-fold validation for Macro-to-Micro MIL")
    parser.add_argument("--base-file", type=str, default="", help="Path to final.py or macro_micro_density_mil.py.")
    parser.add_argument("--img-root", type=str, default="")
    parser.add_argument("--excel-path", type=str, default="")
    parser.add_argument("--id-col", type=str, default="case_id")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--output-dir", type=str, default="MacroMicro_5Fold_Output")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--detector-ckpt",
        type=str,
        default="",
        help=(
            "Detector checkpoint. A full locked model checkpoint is also accepted; "
            "only its detector.* weights will be extracted."
        ),
    )
    parser.add_argument("--train-detector-per-fold", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-val-ratio", type=float, default=0.20)
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="auc",
        choices=["auc", "accuracy", "f1_score", "balanced_accuracy", "mcc"],
    )
    parser.add_argument("--classifier-batch-size", type=int, default=None)
    parser.add_argument("--detector-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--num-rois", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--shared-splits",
        type=str,
        default="",
        help="Optional cv_splits.json shared by all models. If missing, splits are generated and saved there.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional directory containing DMM preprocessed tensors named <patient_id>.npy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    base = import_base_module(args.base_file)
    patch_cached_dmm_loader(base, args.cache_dir)

    probe_config = build_fold_config(base, args, fold_idx=0)
    df, labeled_patients, labels_by_pid, labels, data_info = collect_labeled_patients(
        base,
        probe_config,
    )

    if len(labeled_patients) < args.n_splits * 2:
        raise RuntimeError("Too few labeled patients for K-fold validation.")

    summary_stamp = args.run_name or f"MacroMicro_5fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_dir = ensure_dir(os.path.join(args.output_dir, f"{summary_stamp}_summary"))
    with open(os.path.join(summary_dir, "data_info.json"), "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)

    if args.shared_splits and os.path.exists(args.shared_splits):
        with open(args.shared_splits, "r", encoding="utf-8") as f:
            split_payload = json.load(f)
        folds = split_payload["folds"]
        print(f"Loaded shared CV splits from: {args.shared_splits}")
    else:
        skf = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed,
        )
        patient_array = np.asarray(labeled_patients)
        label_array = np.asarray(labels, dtype=np.int64)
        folds = []
        for fold_idx, (train_val_index, test_index) in enumerate(skf.split(patient_array, label_array)):
            train_val_ids = patient_array[train_val_index].tolist()
            test_ids = patient_array[test_index].tolist()
            train_val_labels = [labels_by_pid[pid] for pid in train_val_ids]

            if args.inner_val_ratio > 0:
                train_core_ids, val_ids = train_test_split(
                    train_val_ids,
                    test_size=args.inner_val_ratio,
                    stratify=train_val_labels,
                    random_state=args.seed + fold_idx,
                )
            else:
                train_core_ids = train_val_ids
                val_ids = test_ids
            folds.append(
                {
                    "fold": fold_idx + 1,
                    "train_ids": list(train_core_ids),
                    "val_ids": list(val_ids),
                    "test_ids": list(test_ids),
                }
            )
        if args.shared_splits:
            os.makedirs(os.path.dirname(args.shared_splits) or ".", exist_ok=True)
            with open(args.shared_splits, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n_splits": args.n_splits,
                        "inner_val_ratio": args.inner_val_ratio,
                        "seed": args.seed,
                        "folds": folds,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"Saved shared CV splits to: {args.shared_splits}")

    all_results = []
    start_time = time.time()
    for fold_idx, fold in enumerate(folds):
        train_core_ids = [str(x) for x in fold["train_ids"]]
        val_ids = [str(x) for x in fold["val_ids"]]
        test_ids = [str(x) for x in fold["test_ids"]]
        detector_source_patients = list(train_core_ids) + list(val_ids)
        result = train_one_fold(
            base,
            args,
            fold_idx,
            train_core_ids,
            val_ids,
            test_ids,
            labels_by_pid,
            detector_source_patients,
        )
        all_results.append(result)

        with open(os.path.join(summary_dir, "partial_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    summarize_results(all_results, summary_dir)
    print(f"Total five-fold time: {time.time() - start_time:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
