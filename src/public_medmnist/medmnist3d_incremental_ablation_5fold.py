#!/usr/bin/env python3
"""
Five-fold incremental ablation for NoduleMNIST3D and AdrenalMNIST3D.

Incremental settings:
  M1: detector-guided ROI mining
  M2: ROI heatmap gating
  M3: tri-planar encoder
  M4: attention-based MIL
  M5: macro-micro fusion

Base model:
  fixed-grid ROI + no heatmap gating + axial single-view CNN
  + mean pooling + micro-only classifier
"""

import argparse
import importlib
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


DATASETS = {
    "nodule": {
        "title": "NoduleMNIST3D",
        "module": "nodule_macro_micro_mil",
        "dataset_class": "Nodule3DDataset",
        "root_default": "data/public/NoduleMNIST3D",
        "shape_intensity_reg": False,
        "noise_std": 0.01,
    },
    "adrenal": {
        "title": "AdrenalMNIST3D",
        "module": "adrenal_macro_micro_mil",
        "dataset_class": "Adrenal3DDataset",
        "root_default": "data/public/AdrenalMNIST3D",
        "shape_intensity_reg": True,
        "noise_std": 0.0,
    },
}


VARIANTS = [
    {
        "name": "base",
        "settings": [0, 0, 0, 0, 0],
        "use_detector_roi": False,
        "use_gating": False,
        "use_triplanar": False,
        "use_mil": False,
        "use_macro": False,
        "description": "fixed grid ROI + no gating + axial view + mean pooling + micro-only",
    },
    {
        "name": "roi_mining",
        "settings": [1, 0, 0, 0, 0],
        "use_detector_roi": True,
        "use_gating": False,
        "use_triplanar": False,
        "use_mil": False,
        "use_macro": False,
        "description": "Base + detector-guided ROI mining",
    },
    {
        "name": "roi_gating",
        "settings": [1, 1, 0, 0, 0],
        "use_detector_roi": True,
        "use_gating": True,
        "use_triplanar": False,
        "use_mil": False,
        "use_macro": False,
        "description": "Base + detector-guided ROI mining + ROI heatmap gating",
    },
    {
        "name": "tri_planar",
        "settings": [1, 1, 1, 0, 0],
        "use_detector_roi": True,
        "use_gating": True,
        "use_triplanar": True,
        "use_mil": False,
        "use_macro": False,
        "description": "Base + detector-guided ROI mining + gating + tri-planar encoder",
    },
    {
        "name": "ab_mil",
        "settings": [1, 1, 1, 1, 0],
        "use_detector_roi": True,
        "use_gating": True,
        "use_triplanar": True,
        "use_mil": True,
        "use_macro": False,
        "description": "Base + detector-guided ROI mining + gating + tri-planar encoder + attention MIL",
    },
    {
        "name": "full",
        "settings": [1, 1, 1, 1, 1],
        "use_detector_roi": True,
        "use_gating": True,
        "use_triplanar": True,
        "use_mil": True,
        "use_macro": True,
        "description": "Base + detector-guided ROI mining + gating + tri-planar encoder + attention MIL + macro-micro fusion",
    },
]


METRICS = [
    ("auc", "AUC"),
    ("accuracy", "ACC"),
    ("f1_score", "F1"),
    ("sensitivity", "Sens."),
    ("specificity", "Spec."),
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def log(message=""):
    print(message, flush=True)


def parse_name_list(value, available):
    if value.lower() == "all":
        return list(available)
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    bad = [x for x in names if x not in available]
    if bad:
        raise ValueError(f"Unknown names {bad}. Available: {list(available)}")
    return names


class CombinedMedMNIST3D(Dataset):
    """Pool official train/val/test splits, then expose chosen indices for CV."""

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        dataset_cls,
        root,
        size,
        augment,
        download,
        shape_intensity_reg,
        noise_std,
        indices=None,
    ):
        self.datasets = {}
        self.index = []
        self.labels = []
        ensure_dir(root)
        for split in self.SPLITS:
            log(f"Loading {dataset_cls.__name__} split={split} root={root} download={download}")
            ds = dataset_cls(
                split=split,
                root=root,
                size=size,
                augment=augment,
                download=download,
                shape_intensity_reg=shape_intensity_reg,
                noise_std=noise_std,
            )
            self.datasets[split] = ds
            for local_i in range(len(ds)):
                _, y = ds.dataset[local_i]
                label = int(np.asarray(y).reshape(-1)[0])
                self.index.append((split, local_i))
                self.labels.append(label)
            log(f"Loaded split={split}, n={len(ds)}")
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.indices = np.arange(len(self.index)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_i = int(self.indices[i])
        split, local_i = self.index[global_i]
        x, y, sample_id = self.datasets[split][local_i]
        return x, y, f"{split}_{sample_id}"


class SingleAxialMicroEncoder(nn.Module):
    def __init__(self, med_module, feat_dim=192):
        super().__init__()
        self.encoder2d = med_module.MicroCNN2D(in_chans=1, feat_dim=feat_dim)

    def forward(self, rois):
        bsz, topk, channels, d, h, w = rois.shape
        axial = rois[:, :, :, d // 2, :, :].reshape(bsz * topk, channels, h, w)
        roi_features = self.encoder2d(axial).reshape(bsz, topk, -1)
        view_weights = rois.new_ones((bsz, topk, 1))
        return roi_features, view_weights


def fixed_grid_centers(volume, topk):
    bsz, _, d, h, w = volume.shape
    fractions = [
        (0.50, 0.50, 0.50),
        (0.35, 0.35, 0.35),
        (0.35, 0.65, 0.65),
        (0.65, 0.35, 0.65),
        (0.65, 0.65, 0.35),
        (0.50, 0.50, 0.75),
        (0.50, 0.75, 0.50),
        (0.75, 0.50, 0.50),
    ]
    while len(fractions) < topk:
        fractions.append(fractions[len(fractions) % 8])
    coords = []
    for fz, fy, fx in fractions[:topk]:
        z = int(round((d - 1) * fz))
        y = int(round((h - 1) * fy))
        x = int(round((w - 1) * fx))
        coords.append((z, y, x))
    centers = torch.tensor(coords, dtype=torch.long, device=volume.device)
    return centers.unsqueeze(0).repeat(bsz, 1, 1)


class IncrementalAblationNet(nn.Module):
    def __init__(self, med_module, variant, args, freeze_detector=True):
        super().__init__()
        self.med_module = med_module
        self.variant = variant
        self.topk = args.topk
        self.roi_size = args.roi_size
        self.embed_dim = args.embed_dim
        self.feat_dim = args.feat_dim
        self.freeze_detector = freeze_detector
        self.uses_detector = (
            variant["use_detector_roi"] or variant["use_gating"] or variant["use_macro"]
        )

        self.detector = med_module.MacroDetector3D(
            img_size=args.size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            heads=args.heads,
        )
        if variant["use_triplanar"]:
            self.micro_encoder = med_module.TriPlanarMicroEncoder(
                roi_size=args.roi_size,
                feat_dim=args.feat_dim,
            )
        else:
            self.micro_encoder = SingleAxialMicroEncoder(
                med_module=med_module,
                feat_dim=args.feat_dim,
            )

        self.mil = (
            med_module.AttentionMIL(feat_dim=args.feat_dim, hidden_dim=args.feat_dim // 2)
            if variant["use_mil"]
            else None
        )
        self.macro_proj = (
            nn.Sequential(
                nn.LayerNorm(args.embed_dim),
                nn.Linear(args.embed_dim, args.feat_dim),
                nn.GELU(),
            )
            if variant["use_macro"]
            else None
        )
        in_dim = args.feat_dim * 2 if variant["use_macro"] else args.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, args.feat_dim),
            nn.LayerNorm(args.feat_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.feat_dim, args.num_classes),
        )

        if not self.uses_detector:
            for p in self.detector.parameters():
                p.requires_grad = False

    def load_detector_state(self, state_dict):
        self.detector.load_state_dict(state_dict, strict=True)

    def set_detector_frozen(self, frozen=True):
        self.freeze_detector = frozen
        for p in self.detector.parameters():
            p.requires_grad = (not frozen) and self.uses_detector

    def _run_detector(self, x):
        if not self.uses_detector:
            bsz, _, d, h, w = x.shape
            return {
                "heatmap": x.new_zeros((bsz, 1, d, h, w)),
                "global_feature": x.new_zeros((bsz, self.embed_dim)),
            }
        if self.freeze_detector:
            self.detector.eval()
            with torch.no_grad():
                return self.detector(x)
        return self.detector(x)

    def forward(self, x):
        det = self._run_detector(x)
        heatmap = det["heatmap"]
        if self.variant["use_detector_roi"]:
            centers = self.med_module.select_topk_centers(
                heatmap,
                topk=self.topk,
                min_distance=max(4, self.roi_size // 3),
            )
        else:
            centers = fixed_grid_centers(x, self.topk)

        rois, roi_maps = self.med_module.crop_rois(x, heatmap, centers, self.roi_size)
        if self.variant["use_gating"]:
            encoder_input = rois * (1.0 + roi_maps)
        else:
            encoder_input = rois

        roi_features, view_weights = self.micro_encoder(encoder_input)
        if self.mil is None:
            micro_feature = roi_features.mean(dim=1)
            mil_weights = roi_features.new_full(
                (roi_features.shape[0], roi_features.shape[1]),
                1.0 / max(1, roi_features.shape[1]),
            )
        else:
            micro_feature, mil_weights = self.mil(roi_features)

        if self.macro_proj is None:
            fused = micro_feature
        else:
            macro_feature = self.macro_proj(det["global_feature"])
            fused = torch.cat([micro_feature, macro_feature], dim=1)

        logits = self.classifier(fused)
        return {
            "logits": logits,
            "heatmap": heatmap,
            "roi_centers": centers,
            "mil_weights": mil_weights,
            "view_weights": view_weights,
        }


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = 0.5
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels_all, probs_all, ids_all = [], [], [], []
    for images, labels, sample_ids in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        losses.append(float(loss.item()))
        labels_all.extend(labels.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
        ids_all.extend(list(sample_ids))
    metrics = compute_metrics(labels_all, probs_all)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, labels_all, probs_all, ids_all


def class_weighted_criterion(train_labels, args, device):
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=args.num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)


def pretrain_detector(med_module, train_loader, val_loader, args, device, out_dir):
    detector = med_module.MacroDetector3D(
        img_size=args.size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
    ).to(device)
    best_path = os.path.join(out_dir, "detector_best.pth")
    if args.detector_epochs <= 0:
        return detector.state_dict()

    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=args.detector_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.detector_epochs),
        eta_min=args.detector_lr * 0.05,
    )
    best_val = float("inf")
    print("\nStage 1: detector pretraining")
    for epoch in range(args.detector_epochs):
        detector.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"pretrain detector {epoch + 1}/{args.detector_epochs}")
        for images, _, _ in pbar:
            images = images.to(device, non_blocking=True)
            v1 = med_module.make_ssl_view(images)
            v2 = med_module.make_ssl_view(images)
            z1 = detector(v1)["ssl_feature"]
            z2 = detector(v2)["ssl_feature"]
            loss = med_module.info_nce(z1, z2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        detector.eval()
        val_losses = []
        with torch.no_grad():
            for images, _, _ in tqdm(val_loader, desc="validate detector", leave=False):
                images = images.to(device, non_blocking=True)
                z1 = detector(med_module.make_ssl_view(images))["ssl_feature"]
                z2 = detector(med_module.make_ssl_view(images))["ssl_feature"]
                val_losses.append(
                    float(med_module.info_nce(z1, z2, temperature=args.temperature).item())
                )
        scheduler.step()
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        print(
            f"Detector epoch {epoch + 1}: "
            f"train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": detector.state_dict(), "epoch": epoch + 1}, best_path)
            print(f"Saved detector best: val_loss={best_val:.4f}")

    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        detector.load_state_dict(state["model_state_dict"])
        print(f"Loaded detector best from epoch {state.get('epoch')}")
    return detector.state_dict()


def train_variant(
    med_module,
    variant,
    detector_state,
    train_loader,
    val_loader,
    test_loader,
    train_labels,
    args,
    device,
    out_dir,
):
    model = IncrementalAblationNet(
        med_module=med_module,
        variant=variant,
        args=args,
        freeze_detector=args.freeze_detector,
    ).to(device)
    if detector_state is not None and model.uses_detector:
        model.load_detector_state(detector_state)
        model.set_detector_frozen(args.freeze_detector)

    criterion = class_weighted_criterion(train_labels, args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 2),
        min_lr=1e-7,
    )

    best_auc, best_f1, no_improve = -1.0, -1.0, 0
    history = []
    best_path = os.path.join(out_dir, "best.pth")
    print(f"\nStage 2: {variant['name']} | {variant['description']}")
    for epoch in range(args.epochs):
        model.train()
        if model.freeze_detector and model.uses_detector:
            model.detector.eval()
        losses, correct, total = [], 0, 0
        pbar = tqdm(train_loader, desc=f"{variant['name']} train {epoch + 1}/{args.epochs}")
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            if model.uses_detector and args.sparsity_weight > 0:
                loss = loss + args.sparsity_weight * outputs["heatmap"].mean()
            if model.uses_detector and args.smoothness_weight > 0:
                loss = loss + args.smoothness_weight * med_module.heatmap_smoothness_loss(
                    outputs["heatmap"]
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(1, total):.3f}")

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = correct / max(1, total)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["auc"])
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_score": val_metrics["f1_score"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1_score']:.4f}, "
            f"val_sens={val_metrics['sensitivity']:.4f}, val_spec={val_metrics['specificity']:.4f}"
        )

        improved = (
            val_metrics["auc"] > best_auc + args.min_delta
            or (
                abs(val_metrics["auc"] - best_auc) <= args.min_delta
                and val_metrics["f1_score"] > best_f1 + args.min_delta
            )
        )
        if improved:
            best_auc = val_metrics["auc"]
            best_f1 = val_metrics["f1_score"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "variant": variant,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best: val_auc={best_auc:.4f}, val_f1={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"Early stopped at epoch {epoch + 1}")
                break

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics, labels, probs, ids = evaluate(model, test_loader, criterion, device)
    pred_rows = []
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    for sample_id, y_true, y_pred, prob in zip(ids, labels, preds, probs):
        pred_rows.append(
            {
                "sample_id": sample_id,
                "true_label": int(y_true),
                "pred_label": int(y_pred),
                "prob_class1": float(prob[1]),
                "confidence": float(prob.max()),
            }
        )
    pd.DataFrame(pred_rows).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(
        f"Test {variant['name']}: AUC={test_metrics['auc']:.4f}, "
        f"ACC={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_score']:.4f}, "
        f"Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f}"
    )
    return test_metrics


def safe_inner_split(train_val_idx, labels, val_ratio, seed):
    y = labels[train_val_idx]
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def summarize_results(fold_rows, out_dir):
    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)
    summary_rows = []
    for (dataset, variant), sub in df.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant}
        for key, _ in METRICS:
            row[f"{key}_mean"] = float(sub[key].mean())
            row[f"{key}_std"] = float(sub[key].std(ddof=1)) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return df, summary


def fmt_cell(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def latex_cell(summary, dataset_key, variant_name, metric_key, best_value):
    row = summary[(summary["dataset"] == dataset_key) & (summary["variant"] == variant_name)]
    if row.empty:
        return "--"
    mean = float(row.iloc[0][f"{metric_key}_mean"])
    std = float(row.iloc[0][f"{metric_key}_std"])
    text = fmt_cell(mean, std)
    if abs(mean - best_value) <= 1e-12:
        return "\\cellcolor{blue!18}\\textbf{" + text + "}"
    return text


def write_latex_table(summary, dataset_keys, out_dir):
    best = {}
    for dataset_key in dataset_keys:
        best[dataset_key] = {}
        ds_summary = summary[summary["dataset"] == dataset_key]
        for metric_key, _ in METRICS:
            best[dataset_key][metric_key] = float(ds_summary[f"{metric_key}_mean"].max())

    cols = "ccccc|" + "|".join(["ccccc" for _ in dataset_keys])
    lines = []
    lines.append("% Required packages: \\usepackage{booktabs,amssymb,graphicx} and \\usepackage[table]{xcolor}")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Incremental ablation study under five-fold cross-validation. "
                 "M1: detector-guided ROI mining; M2: ROI heatmap gating; "
                 "M3: tri-planar encoder; M4: attention-based MIL; M5: macro-micro fusion.}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = ["\\multicolumn{5}{c|}{Settings}"]
    for i, dataset_key in enumerate(dataset_keys):
        title = DATASETS[dataset_key]["title"]
        end = "" if i == len(dataset_keys) - 1 else "|"
        header.append(f"\\multicolumn{{5}}{{c{end}}}{{{title}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\cmidrule(lr){1-5} " + " ".join(
        f"\\cmidrule(lr){{{6 + 5 * i}-{10 + 5 * i}}}" for i in range(len(dataset_keys))
    ))
    metric_header = ["M1", "M2", "M3", "M4", "M5"]
    for _ in dataset_keys:
        metric_header.extend([f"{label}$\\uparrow$" for _, label in METRICS])
    lines.append(" & ".join(metric_header) + " \\\\")
    lines.append("\\midrule")

    variant_names = [v["name"] for v in VARIANTS]
    settings_map = {v["name"]: v["settings"] for v in VARIANTS}
    for variant_name in variant_names:
        settings = ["$\\checkmark$" if x else "$\\times$" for x in settings_map[variant_name]]
        row = list(settings)
        for dataset_key in dataset_keys:
            for metric_key, _ in METRICS:
                row.append(
                    latex_cell(
                        summary,
                        dataset_key=dataset_key,
                        variant_name=variant_name,
                        metric_key=metric_key,
                        best_value=best[dataset_key][metric_key],
                    )
                )
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")

    path = os.path.join(out_dir, "latex_incremental_ablation_table.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run_dataset(dataset_key, args, device, run_dir):
    cfg = DATASETS[dataset_key]
    med_module = importlib.import_module(cfg["module"])
    dataset_cls = getattr(med_module, cfg["dataset_class"])
    root = args.nodule_root if dataset_key == "nodule" else args.adrenal_root
    shape_intensity_reg = cfg["shape_intensity_reg"]
    if dataset_key == "nodule" and args.nodule_shape_intensity_reg:
        shape_intensity_reg = True
    if dataset_key == "adrenal" and args.no_adrenal_shape_intensity_reg:
        shape_intensity_reg = False

    log("\n" + "=" * 90)
    log(f"Dataset: {cfg['title']}")
    log("=" * 90)
    log(f"Root: {root}")
    log(f"Shape/intensity regularization: {shape_intensity_reg}")

    full_data = CombinedMedMNIST3D(
        dataset_cls=dataset_cls,
        root=root,
        size=args.size,
        augment=False,
        download=not args.no_download,
        shape_intensity_reg=shape_intensity_reg,
        noise_std=cfg["noise_std"],
    )
    labels = full_data.labels
    log(f"Total pooled samples: {len(labels)}")
    log(f"Class distribution: {np.bincount(labels, minlength=args.num_classes).tolist()}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    selected_variants = parse_name_list(args.variants, [v["name"] for v in VARIANTS])
    variant_defs = [v for v in VARIANTS if v["name"] in selected_variants]

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        fold_dir = ensure_dir(os.path.join(run_dir, dataset_key, f"fold{fold}"))
        train_idx, val_idx = safe_inner_split(
            np.asarray(train_val_idx, dtype=np.int64),
            labels,
            val_ratio=args.val_ratio,
            seed=args.seed + fold,
        )
        print("\n" + "-" * 90)
        print(f"{cfg['title']} fold {fold}/{args.folds}")
        print(
            f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} | "
            f"train classes={np.bincount(labels[train_idx], minlength=args.num_classes).tolist()}, "
            f"val classes={np.bincount(labels[val_idx], minlength=args.num_classes).tolist()}, "
            f"test classes={np.bincount(labels[test_idx], minlength=args.num_classes).tolist()}"
        )

        train_set = CombinedMedMNIST3D(
            dataset_cls=dataset_cls,
            root=root,
            size=args.size,
            augment=True,
            download=not args.no_download,
            shape_intensity_reg=shape_intensity_reg,
            noise_std=cfg["noise_std"],
            indices=train_idx,
        )
        val_set = CombinedMedMNIST3D(
            dataset_cls=dataset_cls,
            root=root,
            size=args.size,
            augment=False,
            download=not args.no_download,
            shape_intensity_reg=shape_intensity_reg,
            noise_std=cfg["noise_std"],
            indices=val_idx,
        )
        test_set = CombinedMedMNIST3D(
            dataset_cls=dataset_cls,
            root=root,
            size=args.size,
            augment=False,
            download=not args.no_download,
            shape_intensity_reg=shape_intensity_reg,
            noise_std=cfg["noise_std"],
            indices=test_idx,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        detector_state = None
        if any(v["use_detector_roi"] or v["use_gating"] or v["use_macro"] for v in variant_defs):
            detector_dir = ensure_dir(os.path.join(fold_dir, "detector_pretrain"))
            detector_state = pretrain_detector(
                med_module=med_module,
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                device=device,
                out_dir=detector_dir,
            )

        for variant in variant_defs:
            variant_dir = ensure_dir(os.path.join(fold_dir, variant["name"]))
            metrics = train_variant(
                med_module=med_module,
                variant=variant,
                detector_state=detector_state,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                train_labels=labels[train_idx],
                args=args,
                device=device,
                out_dir=variant_dir,
            )
            row = {
                "dataset": dataset_key,
                "dataset_title": cfg["title"],
                "fold": fold,
                "variant": variant["name"],
                "m1_detector_roi": variant["settings"][0],
                "m2_heatmap_gating": variant["settings"][1],
                "m3_triplanar": variant["settings"][2],
                "m4_attention_mil": variant["settings"][3],
                "m5_macro_fusion": variant["settings"][4],
            }
            for key, _ in METRICS:
                row[key] = metrics[key]
            row["precision"] = metrics["precision"]
            row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
            fold_rows.append(row)

    return fold_rows


def run_dataset_official(dataset_key, args, device, run_dir):
    cfg = DATASETS[dataset_key]
    med_module = importlib.import_module(cfg["module"])
    dataset_cls = getattr(med_module, cfg["dataset_class"])
    root = args.nodule_root if dataset_key == "nodule" else args.adrenal_root
    ensure_dir(root)
    shape_intensity_reg = cfg["shape_intensity_reg"]
    if dataset_key == "nodule" and args.nodule_shape_intensity_reg:
        shape_intensity_reg = True
    if dataset_key == "adrenal" and args.no_adrenal_shape_intensity_reg:
        shape_intensity_reg = False

    log("\n" + "=" * 90)
    log(f"Dataset: {cfg['title']} | official train/val/test split")
    log("=" * 90)
    log(f"Root: {root}")
    log(f"Shape/intensity regularization: {shape_intensity_reg}")

    log(f"Loading {cfg['title']} official train split")
    train_set = dataset_cls(
        split="train",
        root=root,
        size=args.size,
        augment=True,
        download=not args.no_download,
        shape_intensity_reg=shape_intensity_reg,
        noise_std=cfg["noise_std"],
    )
    log(f"Loading {cfg['title']} official val split")
    val_set = dataset_cls(
        split="val",
        root=root,
        size=args.size,
        augment=False,
        download=not args.no_download,
        shape_intensity_reg=shape_intensity_reg,
        noise_std=cfg["noise_std"],
    )
    log(f"Loading {cfg['title']} official test split")
    test_set = dataset_cls(
        split="test",
        root=root,
        size=args.size,
        augment=False,
        download=not args.no_download,
        shape_intensity_reg=shape_intensity_reg,
        noise_std=cfg["noise_std"],
    )
    train_labels = np.asarray(
        [int(np.asarray(train_set.dataset[i][1]).reshape(-1)[0]) for i in range(len(train_set))],
        dtype=np.int64,
    )
    val_labels = np.asarray(
        [int(np.asarray(val_set.dataset[i][1]).reshape(-1)[0]) for i in range(len(val_set))],
        dtype=np.int64,
    )
    test_labels = np.asarray(
        [int(np.asarray(test_set.dataset[i][1]).reshape(-1)[0]) for i in range(len(test_set))],
        dtype=np.int64,
    )
    log(
        f"Official counts: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )
    log(
        "Class distribution: "
        f"train={np.bincount(train_labels, minlength=args.num_classes).tolist()}, "
        f"val={np.bincount(val_labels, minlength=args.num_classes).tolist()}, "
        f"test={np.bincount(test_labels, minlength=args.num_classes).tolist()}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    selected_variants = parse_name_list(args.variants, [v["name"] for v in VARIANTS])
    variant_defs = [v for v in VARIANTS if v["name"] in selected_variants]
    dataset_dir = ensure_dir(os.path.join(run_dir, dataset_key, "official_split"))

    detector_state = None
    if any(v["use_detector_roi"] or v["use_gating"] or v["use_macro"] for v in variant_defs):
        detector_dir = ensure_dir(os.path.join(dataset_dir, "detector_pretrain"))
        detector_state = pretrain_detector(
            med_module=med_module,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            device=device,
            out_dir=detector_dir,
        )

    rows = []
    for variant in variant_defs:
        variant_dir = ensure_dir(os.path.join(dataset_dir, variant["name"]))
        metrics = train_variant(
            med_module=med_module,
            variant=variant,
            detector_state=detector_state,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_labels=train_labels,
            args=args,
            device=device,
            out_dir=variant_dir,
        )
        row = {
            "dataset": dataset_key,
            "dataset_title": cfg["title"],
            "protocol": "official",
            "variant": variant["name"],
            "m1_detector_roi": variant["settings"][0],
            "m2_heatmap_gating": variant["settings"][1],
            "m3_triplanar": variant["settings"][2],
            "m4_attention_mil": variant["settings"][3],
            "m5_macro_fusion": variant["settings"][4],
        }
        for key, _ in METRICS:
            row[key] = metrics[key]
        row["precision"] = metrics["precision"]
        row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
        rows.append(row)
    return rows


def summarize_official_results(rows, out_dir):
    df = pd.DataFrame(rows)
    official_path = os.path.join(out_dir, "official_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(official_path, index=False)
    df.to_csv(summary_path, index=False)
    return df


def latex_cell_official(summary, dataset_key, variant_name, metric_key, best_value):
    row = summary[(summary["dataset"] == dataset_key) & (summary["variant"] == variant_name)]
    if row.empty:
        return "--"
    value = float(row.iloc[0][metric_key])
    text = f"{value:.4f}"
    if abs(value - best_value) <= 1e-12:
        return "\\cellcolor{blue!18}\\textbf{" + text + "}"
    return text


def write_latex_table_official(summary, dataset_keys, out_dir):
    best = {}
    for dataset_key in dataset_keys:
        best[dataset_key] = {}
        ds_summary = summary[summary["dataset"] == dataset_key]
        for metric_key, _ in METRICS:
            best[dataset_key][metric_key] = float(ds_summary[metric_key].max())

    cols = "ccccc|" + "|".join(["ccccc" for _ in dataset_keys])
    lines = []
    lines.append("% Required packages: \\usepackage{booktabs,amssymb,graphicx} and \\usepackage[table]{xcolor}")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Incremental ablation study on official MedMNIST3D splits. "
                 "M1: detector-guided ROI mining; M2: ROI heatmap gating; "
                 "M3: tri-planar encoder; M4: attention-based MIL; M5: macro-micro fusion.}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = ["\\multicolumn{5}{c|}{Settings}"]
    for i, dataset_key in enumerate(dataset_keys):
        title = DATASETS[dataset_key]["title"]
        end = "" if i == len(dataset_keys) - 1 else "|"
        header.append(f"\\multicolumn{{5}}{{c{end}}}{{{title}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\cmidrule(lr){1-5} " + " ".join(
        f"\\cmidrule(lr){{{6 + 5 * i}-{10 + 5 * i}}}" for i in range(len(dataset_keys))
    ))
    metric_header = ["M1", "M2", "M3", "M4", "M5"]
    for _ in dataset_keys:
        metric_header.extend([f"{label}$\\uparrow$" for _, label in METRICS])
    lines.append(" & ".join(metric_header) + " \\\\")
    lines.append("\\midrule")

    variant_names = [v["name"] for v in VARIANTS]
    settings_map = {v["name"]: v["settings"] for v in VARIANTS}
    for variant_name in variant_names:
        settings = ["$\\checkmark$" if x else "$\\times$" for x in settings_map[variant_name]]
        row = list(settings)
        for dataset_key in dataset_keys:
            for metric_key, _ in METRICS:
                row.append(
                    latex_cell_official(
                        summary,
                        dataset_key=dataset_key,
                        variant_name=variant_name,
                        metric_key=metric_key,
                        best_value=best[dataset_key][metric_key],
                    )
                )
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")

    path = os.path.join(out_dir, "latex_incremental_ablation_table.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental five-fold ablation for NoduleMNIST3D and AdrenalMNIST3D."
    )
    parser.add_argument(
        "--protocol",
        choices=["official", "cv5"],
        default="official",
        help="official uses MedMNIST train/val/test; cv5 pools all official splits for 5-fold CV.",
    )
    parser.add_argument("--datasets", default="nodule,adrenal", help="nodule, adrenal, or all")
    parser.add_argument("--variants", default="all", help="Comma list or all")
    parser.add_argument("--nodule-root", default=DATASETS["nodule"]["root_default"])
    parser.add_argument("--adrenal-root", default=DATASETS["adrenal"]["root_default"])
    parser.add_argument("--output-root", default="./MedMNIST3D_Incremental_Ablation_5Fold_Output")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--detector-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--detector-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--roi-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feat-dim", type=int, default=192)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--sparsity-weight", type=float, default=0.004)
    parser.add_argument("--smoothness-weight", type=float, default=0.004)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--freeze-detector", action="store_true", default=True)
    parser.add_argument("--finetune-detector", dest="freeze_detector", action="store_false")
    parser.add_argument("--nodule-shape-intensity-reg", action="store_true")
    parser.add_argument("--no-adrenal-shape-intensity-reg", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    dataset_keys = parse_name_list(args.datasets, DATASETS.keys())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(args.output_root, f"incremental_ablation_{stamp}"))
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("MedMNIST3D incremental ablation")
    print("=" * 90)
    print("Run dir:", run_dir)
    print("Device:", device)
    print("Protocol:", args.protocol)
    print("Datasets:", dataset_keys)
    print("Variants:", parse_name_list(args.variants, [v["name"] for v in VARIANTS]))

    if args.protocol == "official":
        all_rows = []
        for dataset_key in dataset_keys:
            all_rows.extend(run_dataset_official(dataset_key, args, device, run_dir))
            summary = summarize_official_results(all_rows, run_dir)
            latex_path = write_latex_table_official(summary, dataset_keys, run_dir)
            print("Updated official results:", os.path.join(run_dir, "official_results.csv"))
            print("Updated summary:", os.path.join(run_dir, "summary.csv"))
            print("Updated LaTeX table:", latex_path)

        summary = summarize_official_results(all_rows, run_dir)
        latex_path = write_latex_table_official(summary, dataset_keys, run_dir)
        print("\n" + "=" * 90)
        print("Final official-split summary")
        print("=" * 90)
        for dataset_key in dataset_keys:
            print(f"\n{DATASETS[dataset_key]['title']}")
            ds = summary[summary["dataset"] == dataset_key]
            for variant in [v["name"] for v in VARIANTS]:
                row = ds[ds["variant"] == variant]
                if row.empty:
                    continue
                parts = [variant]
                for metric_key, label in METRICS:
                    parts.append(f"{label}={row.iloc[0][metric_key]:.4f}")
                print(" | ".join(parts))
        print("\nSaved official results:", os.path.join(run_dir, "official_results.csv"))
        print("Saved summary:", os.path.join(run_dir, "summary.csv"))
        print("Saved LaTeX:", latex_path)
        return

    all_rows = []
    for dataset_key in dataset_keys:
        all_rows.extend(run_dataset(dataset_key, args, device, run_dir))
        _, summary = summarize_results(all_rows, run_dir)
        latex_path = write_latex_table(summary, dataset_keys, run_dir)
        print("Updated summary:", os.path.join(run_dir, "summary.csv"))
        print("Updated LaTeX table:", latex_path)

    _, summary = summarize_results(all_rows, run_dir)
    latex_path = write_latex_table(summary, dataset_keys, run_dir)
    print("\n" + "=" * 90)
    print("Final five-fold summary")
    print("=" * 90)
    for dataset_key in dataset_keys:
        print(f"\n{DATASETS[dataset_key]['title']}")
        ds = summary[summary["dataset"] == dataset_key]
        for variant in [v["name"] for v in VARIANTS]:
            row = ds[ds["variant"] == variant]
            if row.empty:
                continue
            parts = [variant]
            for metric_key, label in METRICS:
                parts.append(
                    f"{label}={row.iloc[0][metric_key + '_mean']:.4f}"
                    f"+/-{row.iloc[0][metric_key + '_std']:.4f}"
                )
            print(" | ".join(parts))
    print("\nSaved fold results:", os.path.join(run_dir, "fold_results.csv"))
    print("Saved summary:", os.path.join(run_dir, "summary.csv"))
    print("Saved LaTeX:", latex_path)


if __name__ == "__main__":
    main()
