#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one baseline model with the shared private-dataset 5-fold split.

This script is designed for parallel multi-GPU launching: start one process per
model and set CUDA_VISIBLE_DEVICES before each process.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch.nn as nn

from common_3d_train_7_3 import Safe3DClassifier, TrainConfig
from common_3d_train_5fold import parse_input_size, run_5fold_training


SCRIPT_MAP = {
    "AMSNet": ("amsnet_7_3_private_dataset.py", "AMSNet"),
    "Med3D": ("train_Med3D_ResNet_7_3.py", "Med3D"),
    "Med3D_ResNet": ("train_Med3D_ResNet_7_3.py", "Med3D"),
    "ResNet": ("train_ResNet_7_3.py", "ResNet"),
    "MedVit-3D": ("train_Vit_7_3.py", "MedVit-3D"),
    "ViT": ("train_Vit_7_3.py", "MedVit-3D"),
    "M3T": ("train_M3T_7_3.py", "M3T"),
    "3DCT-ICH": ("train_Hybrid_CNN_Transformer_7_3.py", "3DCT-ICH"),
    "Hybrid_CNN_Transformer": ("train_Hybrid_CNN_Transformer_7_3.py", "3DCT-ICH"),
    "SwinTransformer-3D": ("train_Swin3D_Classifier_7_3.py", "SwinTransformer-3D"),
    "Swin3D_Classifier": ("train_Swin3D_Classifier_7_3.py", "SwinTransformer-3D"),
    "X3D": ("train_X3D_Efficient_7_3.py", "X3D"),
    "X3D_Efficient": ("train_X3D_Efficient_7_3.py", "X3D"),
    "XFMamba": ("train_XMamba_7_3.py", "XFMamba"),
    "XMamba": ("train_XMamba_7_3.py", "XFMamba"),
}

HEAVY_MODELS = {"MedVit-3D", "ViT", "M3T", "SwinTransformer-3D", "Swin3D_Classifier", "XFMamba", "XMamba"}


def import_model(model_key: str, script_dir: Path) -> Tuple[str, Callable[[TrainConfig], nn.Module]]:
    if model_key == "Safe3D_CNN":
        def build_safe(config: TrainConfig) -> nn.Module:
            return Safe3DClassifier(in_channels=len(config.modalities), num_classes=1)

        return "Safe3D_CNN", build_safe

    if model_key not in SCRIPT_MAP:
        raise ValueError(f"Unknown model: {model_key}. Known: {sorted(SCRIPT_MAP)}")

    script_name, display_name = SCRIPT_MAP[model_key]
    script_path = script_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Model script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(f"private_cv_{model_key}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if model_key == "AMSNet":
        def build_amsnet(config: TrainConfig) -> nn.Module:
            return module.AMSNet(in_channels=len(config.modalities), num_classes=1)

        return display_name, build_amsnet

    return display_name, getattr(module, "build_model")


def parse_args():
    parser = argparse.ArgumentParser("Train one private baseline model with shared 5-fold CV.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--excel-path", required=True)
    parser.add_argument("--id-col", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--output-root", default="Model_5Fold_Output")
    parser.add_argument("--run-name", default="private_cv5_parallel_baselines")
    parser.add_argument("--input-size", default="64,64,64")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heavy-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-val-size", type=float, default=0.20)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="auc", choices=["auc", "accuracy", "f1_score", "mcc", "loss"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--head-lr-mult", type=float, default=5.0)
    parser.add_argument("--shared-splits", required=True)
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional directory containing preprocessed patient tensors named <patient_id>.npy.",
    )
    return parser.parse_args()


def patch_cached_loader(cache_dir: str) -> None:
    """Redirect the shared dataset loader to preprocessed .npy tensors."""
    if not cache_dir:
        return

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_path}")

    import common_3d_train_7_3 as base_loader

    original_loader = base_loader.load_patient_tensor

    def cached_load_patient_tensor(img_root, patient_id, config):
        patient_id = str(patient_id).strip()
        tensor_path = cache_path / f"{patient_id}.npy"
        if tensor_path.exists():
            return np.load(tensor_path, mmap_mode=None).astype(np.float32)
        print(f"[WARN] Cache miss for patient {patient_id}; falling back to NIfTI loading.")
        return original_loader(img_root, patient_id, config)

    base_loader.load_patient_tensor = cached_load_patient_tensor
    print(f"[INFO] Using preprocessed tensor cache: {cache_path}")


def main() -> int:
    args = parse_args()
    patch_cached_loader(args.cache_dir)
    script_dir = Path(__file__).resolve().parent
    model_name, build_model = import_model(args.model, script_dir)
    batch_size = args.heavy_batch_size if args.model in HEAVY_MODELS else args.batch_size

    config = TrainConfig(
        img_root=args.img_root,
        excel_path=args.excel_path,
        id_col=args.id_col,
        label_col=args.label_col,
        output_root=args.output_root,
        input_size=parse_input_size(args.input_size),
        random_state=args.random_state,
        batch_size=batch_size,
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

    run_5fold_training(
        model_name=model_name,
        build_model=build_model,
        config=config,
        n_splits=args.n_splits,
        inner_val_size=args.inner_val_size,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        selection_metric=args.selection_metric,
        run_name=args.run_name,
        shared_splits_path=args.shared_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
