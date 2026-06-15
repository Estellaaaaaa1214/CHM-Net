#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study runner for the Macro-to-Micro Density-Aware MIL Network.

This script does not pretrain or fine-tune the detector. It loads an existing
detector checkpoint and trains only the classifier/fusion modules for each
ablation variant on the same 7:3 split.

Put this file in the same server directory as macro_micro_density_mil.py:
    <PROJECT_ROOT>/macro_micro_ablation.py
"""

import argparse
import copy
import glob
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import macro_micro_density_mil as base


ABLATION_DESCRIPTIONS = {
    "full": "Full M3D-MIL: detector Top-K ROI + heatmap gating + tri-planar CNN + AB-MIL + macro fusion",
    "no_detector_roi": "w/o detector-guided ROI mining: use fixed grid ROIs instead of heatmap Top-K ROIs",
    "no_gating": "w/o ROI heatmap gating: use selected ROIs but remove local heatmap modulation/projection",
    "no_mil": "w/o attention MIL: replace AB-MIL with simple mean pooling over ROI features",
    "no_macro": "w/o macro fusion: remove global macro feature, classify using patient-level micro feature only",
    "single_view": "w/o tri-planar views: use axial view only instead of axial/coronal/sagittal fusion",
}


def set_seed(seed=2024):
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


def latest_detector_ckpt(output_dir):
    candidates = glob.glob(os.path.join(output_dir, "*", "checkpoints", "detector_best.pth"))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


class FixedGridROISampler3D(nn.Module):
    """Deterministic non-detector ROI sampler for the ROI-mining ablation."""

    def __init__(self, roi_size=32, num_rois=6):
        super().__init__()
        self.roi_size = int(roi_size)
        self.num_rois = int(num_rois)

    @staticmethod
    def _crop_3d(volume, center, roi_size):
        channels, depth, height, width = volume.shape
        half = roi_size // 2
        z, y, x = [int(v) for v in center]
        z = min(max(z, half), depth - (roi_size - half))
        y = min(max(y, half), height - (roi_size - half))
        x = min(max(x, half), width - (roi_size - half))
        z1, y1, x1 = z - half, y - half, x - half
        z2, y2, x2 = z1 + roi_size, y1 + roi_size, x1 + roi_size
        return volume[:, z1:z2, y1:y2, x1:x2], (z, y, x)

    def _centers(self, depth, height, width, device):
        cz, cy, cx = depth // 2, height // 2, width // 2
        dz, dy, dx = max(depth // 5, 1), max(height // 5, 1), max(width // 5, 1)
        centers = [
            (cz, cy, cx),
            (cz - dz, cy, cx),
            (cz + dz, cy, cx),
            (cz, cy - dy, cx),
            (cz, cy + dy, cx),
            (cz, cy, cx + dx),
            (cz, cy, cx - dx),
        ]
        centers = centers[: self.num_rois]
        while len(centers) < self.num_rois:
            centers.append((cz, cy, cx))
        return centers

    def forward(self, images, heatmaps):
        batch_size, channels, depth, height, width = images.shape
        rois, roi_heatmaps, coords, roi_scores = [], [], [], []
        centers = self._centers(depth, height, width, images.device)

        for b in range(batch_size):
            patient_rois, patient_maps, patient_coords, patient_scores = [], [], [], []
            for center in centers:
                roi, safe_center = self._crop_3d(images[b], center, self.roi_size)
                roi_map, _ = self._crop_3d(heatmaps[b], safe_center, self.roi_size)
                patient_rois.append(roi)
                patient_maps.append(roi_map)
                patient_coords.append(torch.tensor(safe_center, device=images.device, dtype=torch.long))
                patient_scores.append(heatmaps[b, 0, safe_center[0], safe_center[1], safe_center[2]])
            rois.append(torch.stack(patient_rois, dim=0))
            roi_heatmaps.append(torch.stack(patient_maps, dim=0))
            coords.append(torch.stack(patient_coords, dim=0))
            roi_scores.append(torch.stack(patient_scores, dim=0))

        return (
            torch.stack(rois, dim=0),
            torch.stack(roi_heatmaps, dim=0),
            torch.stack(coords, dim=0),
            torch.stack(roi_scores, dim=0),
        )


class SingleAxialMicroEncoder(nn.Module):
    """Use only axial 2D view to test the contribution of tri-planar encoding."""

    def __init__(self, in_channels=3, embed_dim=256, input_size=128, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        ref = base.TriPlanarMicroEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            input_size=input_size,
            dropout=dropout,
        )
        self.encoder2d = ref.encoder2d
        self.proj = ref.proj

    @staticmethod
    def _normalize_2d(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    @staticmethod
    def _attention_projection(roi, roi_heatmap, axis):
        numerator = (roi * roi_heatmap).sum(dim=axis)
        denominator = roi_heatmap.sum(dim=axis).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, rois, roi_heatmaps, gate_mode="residual", gate_alpha=1.0):
        batch_size, num_rois, channels, size_d, size_h, size_w = rois.shape
        n = batch_size * num_rois
        rois = rois.reshape(n, channels, size_d, size_h, size_w)
        roi_heatmaps = roi_heatmaps.reshape(n, 1, size_d, size_h, size_w)
        if gate_mode == "multiply":
            gated = rois * roi_heatmaps
        else:
            gated = rois * (1.0 + gate_alpha * roi_heatmaps)
        axial = self._attention_projection(gated, roi_heatmaps, axis=2)
        axial = F.interpolate(
            axial,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        axial = self._normalize_2d(axial)
        roi_features = self.proj(self.encoder2d(axial)).reshape(batch_size, num_rois, -1)
        view_weights = torch.zeros(batch_size, num_rois, 3, device=rois.device)
        view_weights[:, :, 0] = 1.0
        return roi_features, view_weights


class AblationMILHead(nn.Module):
    def __init__(self, variant, macro_dim=384, micro_dim=256, num_classes=2, dropout=0.45):
        super().__init__()
        self.variant = variant
        self.macro_proj = nn.Sequential(
            nn.Linear(macro_dim, micro_dim),
            nn.LayerNorm(micro_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )
        self.mil_attention = nn.Sequential(
            nn.Linear(micro_dim, micro_dim // 2),
            nn.Tanh(),
            nn.Linear(micro_dim // 2, 1),
        )
        in_dim = micro_dim if variant == "no_macro" else micro_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, global_features, micro_features, roi_scores=None):
        batch_size, num_rois, _ = micro_features.shape
        if self.variant == "no_mil":
            mil_weights = torch.full(
                (batch_size, num_rois),
                1.0 / float(num_rois),
                device=micro_features.device,
            )
        else:
            mil_logits = self.mil_attention(micro_features).squeeze(-1)
            if roi_scores is not None and self.variant != "no_gating":
                roi_bias = (roi_scores - roi_scores.mean(dim=1, keepdim=True)) / (
                    roi_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
                )
                mil_logits = mil_logits + 0.25 * roi_bias
            mil_weights = F.softmax(mil_logits, dim=1)

        patient_micro = torch.sum(micro_features * mil_weights.unsqueeze(-1), dim=1)
        if self.variant == "no_macro":
            fused = patient_micro
        else:
            macro_feature = self.macro_proj(global_features)
            fused = torch.cat([macro_feature, patient_micro], dim=1)
        logits = self.classifier(fused)
        return logits, mil_weights


class AblationMacroToMicroNet(nn.Module):
    def __init__(self, detector, config, variant):
        super().__init__()
        self.config = config
        self.variant = variant
        self.detector = detector
        if variant == "no_detector_roi":
            self.roi_sampler = FixedGridROISampler3D(
                roi_size=config.ROI_SIZE,
                num_rois=config.NUM_ROIS,
            )
        else:
            self.roi_sampler = base.TopKROISampler3D(
                roi_size=config.ROI_SIZE,
                num_rois=config.NUM_ROIS,
                suppress_radius=config.ROI_SUPPRESS_RADIUS,
            )
        if variant == "single_view":
            self.micro_encoder = SingleAxialMicroEncoder(
                in_channels=config.IN_CHANNELS,
                embed_dim=config.MICRO_EMBED_DIM,
                input_size=config.MICRO_INPUT_SIZE,
                dropout=0.2,
            )
        else:
            self.micro_encoder = base.TriPlanarMicroEncoder(
                in_channels=config.IN_CHANNELS,
                embed_dim=config.MICRO_EMBED_DIM,
                input_size=config.MICRO_INPUT_SIZE,
                dropout=0.2,
            )
        self.mil_head = AblationMILHead(
            variant=variant,
            macro_dim=config.EMBED_DIM,
            micro_dim=config.MICRO_EMBED_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RATE,
        )

    def forward(self, images):
        detector_out = self.detector(images, return_dict=True)
        heatmap = detector_out["detection_map"]
        global_features = detector_out["global_features"]
        rois, roi_heatmaps, roi_coords, roi_scores = self.roi_sampler(images, heatmap)
        if self.variant == "no_gating":
            micro_heatmaps = torch.ones_like(roi_heatmaps)
            head_scores = None
        else:
            micro_heatmaps = roi_heatmaps
            head_scores = roi_scores
        micro_features, view_weights = self.micro_encoder(
            rois,
            micro_heatmaps,
            gate_mode=self.config.ROI_GATE_MODE,
            gate_alpha=self.config.ROI_GATE_ALPHA,
        )
        logits, mil_weights = self.mil_head(global_features, micro_features, head_scores)
        return {
            "logits": logits,
            "heatmap": heatmap,
            "roi_coords": roi_coords,
            "roi_scores": roi_scores,
            "mil_weights": mil_weights,
            "view_weights": view_weights,
        }


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (1 - np.cos(np.pi * (epoch + 1) / self.warmup_epochs)) / 2
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        return self.optimizer.param_groups[0]["lr"]


def build_config(args, variant):
    config = base.EnhancedConfig(args)
    config.EXPERIMENT_NAME = f"ablation_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.exp_dir = ensure_dir(os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME))
    for d in ["checkpoints", "logs", "metrics", "predictions", "features", "visualizations"]:
        ensure_dir(os.path.join(config.exp_dir, d))
    config.FREEZE_DETECTOR_EPOCHS = 10**9
    config.FINETUNE_DETECTOR_LR = 0.0
    config.EARLY_STOP_PATIENCE = args.early_stop_patience
    if args.classifier_lr is not None:
        config.CLASSIFIER_LR = args.classifier_lr
    return config


def prepare_data(config):
    df = pd.read_excel(config.EXCEL_PATH)
    _, valid_patients, labeled_patients, class_dist = base.collect_valid_patients(config, df)
    train_ids, val_ids, test_ids, labels_by_pid = base.split_labeled_patients(
        labeled_patients,
        df,
        config,
    )
    train_labels = [labels_by_pid[pid] for pid in train_ids]
    val_labels = [labels_by_pid[pid] for pid in val_ids]
    print(f"Split: train={len(train_ids)}, val/test merged={len(val_ids)}")
    print("Train class distribution:", np.bincount(train_labels, minlength=2).tolist())
    print("Val/Test class distribution:", np.bincount(val_labels, minlength=2).tolist())

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
    return train_loader, val_loader, test_loader, train_labels


def load_detector(config, ckpt_path):
    config.DETECTOR_CKPT = ckpt_path
    detector, _ = base.load_detector_from_checkpoint(config)
    base.set_module_trainable(detector, False)
    detector.eval()
    return detector


def optimizer_for_classifier(model, config):
    params = [p for n, p in model.named_parameters() if not n.startswith("detector.") and p.requires_grad]
    return torch.optim.AdamW(params, lr=config.CLASSIFIER_LR, weight_decay=config.WEIGHT_DECAY)


def evaluate(model, loader, criterion, config, save_predictions=False):
    model.eval()
    all_preds, all_probs, all_labels, all_pids = [], [], [], []
    all_roi_coords, all_roi_weights, all_view_weights = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, pids in tqdm(loader, desc="evaluate", leave=False):
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = base.attention_smoothness_loss(outputs["heatmap"])
            loss = cls_loss + config.SPARSITY_WEIGHT * sparse_loss + config.SMOOTHNESS_WEIGHT * smooth_loss
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_pids.extend(list(pids))
            all_roi_coords.extend(outputs["roi_coords"].cpu().numpy().tolist())
            all_roi_weights.extend(outputs["mil_weights"].cpu().numpy().tolist())
            all_view_weights.extend(outputs["view_weights"].cpu().numpy().tolist())

    metrics = base.compute_enhanced_metrics(all_labels, np.array(all_preds), all_probs)
    avg_loss = total_loss / max(1, len(loader))
    metrics["loss"] = float(avg_loss)

    if save_predictions:
        predictions = []
        for pid, true_label, pred_label, prob, coords, roi_w, view_w in zip(
            all_pids,
            all_labels,
            all_preds,
            all_probs,
            all_roi_coords,
            all_roi_weights,
            all_view_weights,
        ):
            predictions.append(
                {
                    "patient_id": str(pid),
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "probability": float(prob),
                    "correct": int(true_label == pred_label),
                    "roi_centers_zyx": coords,
                    "roi_mil_weights": roi_w,
                    "view_weights_axial_coronal_sagittal": view_w,
                }
            )
        with open(os.path.join(config.exp_dir, "predictions", "test_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    return metrics, avg_loss


def train_one_variant(args, variant, detector_ckpt):
    print("\n" + "=" * 80)
    print(f"Ablation variant: {variant}")
    print(ABLATION_DESCRIPTIONS[variant])
    print("=" * 80)
    set_seed(args.seed)
    config = build_config(args, variant)
    train_loader, val_loader, test_loader, train_labels = prepare_data(config)
    detector = load_detector(config, detector_ckpt)
    model = AblationMacroToMicroNet(detector, config, variant).to(config.device)
    base.set_module_trainable(model.detector, False)
    model.detector.eval()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=base.compute_class_weights(train_labels, config.device),
        label_smoothing=config.LABEL_SMOOTHING,
    )
    optimizer = optimizer_for_classifier(model, config)
    warmup = WarmupScheduler(optimizer, config.WARMUP_EPOCHS, config.CLASSIFIER_LR)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=12,
        verbose=True,
        min_lr=1e-7,
    )

    best_auc = -1.0
    best_metrics = None
    best_epoch = 0
    no_improve = 0
    history = []
    best_path = os.path.join(config.exp_dir, "checkpoints", "ablation_best.pth")

    for epoch in range(config.CLASSIFIER_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.CLASSIFIER_EPOCHS} [{variant}]")
        lr = warmup.step(epoch) if epoch < config.WARMUP_EPOCHS else optimizer.param_groups[0]["lr"]
        model.train()
        model.detector.eval()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="train")
        for images, labels, _ in pbar:
            images = images.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            optimizer.zero_grad()
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
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / max(1, total):.2f}%")

        train_loss /= max(1, len(train_loader))
        train_acc = correct / max(1, total)
        val_metrics, val_loss = evaluate(model, val_loader, criterion, config, save_predictions=False)
        if epoch >= config.WARMUP_EPOCHS:
            plateau.step(val_metrics["auc"])

        row = {
            "epoch": epoch + 1,
            "variant": variant,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["accuracy"]),
            "val_auc": float(val_metrics["auc"]),
            "val_f1": float(val_metrics["f1_score"]),
            "val_recall": float(val_metrics["recall"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_balanced_acc": float(val_metrics["balanced_accuracy"]),
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

        if val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "variant": variant,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config.__dict__,
                    "detector_ckpt": detector_ckpt,
                },
                best_path,
            )
            print(f"Saved best [{variant}] by AUC={best_auc:.4f}")
        else:
            no_improve += 1
            print(f"No AUC improvement: {no_improve}/{config.EARLY_STOP_PATIENCE}")
            if no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopped [{variant}] at epoch {epoch + 1}")
                break

    checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_loss = evaluate(model, test_loader, criterion, config, save_predictions=True)
    test_metrics["loss"] = float(test_loss)

    result = {
        "variant": variant,
        "description": ABLATION_DESCRIPTIONS[variant],
        "exp_dir": config.exp_dir,
        "detector_ckpt": detector_ckpt,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    with open(os.path.join(config.exp_dir, "metrics", "ablation_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nFinished {variant}: test AUC={test_metrics['auc']:.4f}, acc={test_metrics['accuracy']:.4f}")
    return result


def parse_args():
    parser = argparse.ArgumentParser("Macro-to-Micro MIL ablation study")
    parser.add_argument("--img-root", type=str, default="")
    parser.add_argument("--excel-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="MacroMicro_Ablation_Output")
    parser.add_argument("--detector-ckpt", type=str, default="")
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--classifier-batch-size", type=int, default=2)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--num-rois", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_detector_roi,no_gating,no_mil,no_macro,single_view",
        help="Comma-separated variants. Choices: " + ",".join(ABLATION_DESCRIPTIONS.keys()),
    )
    # Attributes expected by base.EnhancedConfig.
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--skip-detector-pretrain", action="store_true")
    parser.add_argument("--freeze-detector-epochs", type=int, default=None)
    parser.add_argument("--no-detector-finetune", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ABLATION_DESCRIPTIONS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    detector_ckpt = args.detector_ckpt or latest_detector_ckpt("MacroMicro_MIL_Output")
    if not detector_ckpt:
        raise FileNotFoundError("No detector checkpoint found. Please pass --detector-ckpt.")
    print("Using detector checkpoint:", detector_ckpt)

    summary_dir = ensure_dir(os.path.join(args.output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    all_results = []
    start = time.time()
    for variant in variants:
        result = train_one_variant(args, variant, detector_ckpt)
        all_results.append(result)
        rows = []
        for item in all_results:
            m = item["test_metrics"]
            vm = item["best_val_metrics"] or {}
            rows.append(
                {
                    "variant": item["variant"],
                    "best_epoch": item["best_epoch"],
                    "val_auc": vm.get("auc", 0),
                    "val_acc": vm.get("accuracy", 0),
                    "test_auc": m.get("auc", 0),
                    "test_acc": m.get("accuracy", 0),
                    "test_f1": m.get("f1_score", 0),
                    "test_recall": m.get("recall", 0),
                    "test_specificity": m.get("specificity", 0),
                    "test_mcc": m.get("mcc", 0),
                    "exp_dir": item["exp_dir"],
                }
            )
        pd.DataFrame(rows).to_csv(
            os.path.join(summary_dir, "ablation_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        with open(os.path.join(summary_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Ablation finished")
    print(f"Summary dir: {summary_dir}")
    print(f"Total time: {time.time() - start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
