#!/usr/bin/env python3
import argparse
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from medmnist import NoduleMNIST3D, INFO
except Exception:
    NoduleMNIST3D = None
    INFO = {}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


class Nodule3DDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        size=64,
        augment=False,
        download=True,
        shape_intensity_reg=False,
        noise_std=0.01,
    ):
        if NoduleMNIST3D is None:
            raise ImportError(
                "medmnist is not installed. Run: pip install medmnist"
            )
        self.split = split
        self.augment = augment
        self.shape_intensity_reg = shape_intensity_reg
        self.noise_std = noise_std
        self.dataset = NoduleMNIST3D(
            split=split,
            size=size,
            download=download,
            root=root,
        )

    def __len__(self):
        return len(self.dataset)

    def _to_volume(self, x):
        x = np.asarray(x).astype(np.float32)
        if x.ndim == 4:
            if x.shape[0] in (1, 3):
                x = x[0]
            elif x.shape[-1] in (1, 3):
                x = x[..., 0]
            else:
                x = np.squeeze(x)
        if x.max() > 1.5:
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0)
        return x

    def _augment(self, x):
        if random.random() < 0.5:
            x = np.flip(x, axis=0).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()
        scale = random.uniform(0.0, 1.0) if self.shape_intensity_reg else random.uniform(0.85, 1.15)
        noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        x = np.clip(x * scale + noise, 0.0, 1.0)
        return x

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self._to_volume(x)
        if self.augment:
            x = self._augment(x)
        elif self.shape_intensity_reg:
            x = x * 0.5
        y = int(np.asarray(y).reshape(-1)[0])
        x = torch.from_numpy(x).float().unsqueeze(0)
        return x, torch.tensor(y, dtype=torch.long), str(index)


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=128):
        super().__init__()
        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1]]
        return x, grid


class MacroDetector3D(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=128, depth=3, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.det_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.ssl_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        tokens, grid = self.patch_embed(x)
        tokens = self.norm(self.encoder(tokens))
        global_feature = tokens.mean(dim=1)
        heatmap_logits = self.det_head(tokens).transpose(1, 2)
        heatmap_logits = heatmap_logits.reshape(x.shape[0], 1, *grid)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return {
            "heatmap": torch.sigmoid(heatmap_logits),
            "global_feature": global_feature,
            "ssl_feature": self.ssl_head(global_feature),
        }


class MicroCNN2D(nn.Module):
    def __init__(self, in_chans=1, feat_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(self.encoder(x))


class TriPlanarMicroEncoder(nn.Module):
    def __init__(self, roi_size=32, feat_dim=192):
        super().__init__()
        self.roi_size = roi_size
        self.encoder2d = MicroCNN2D(in_chans=1, feat_dim=feat_dim)
        self.view_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, rois):
        bsz, topk, channels, d, h, w = rois.shape
        cz, cy, cx = d // 2, h // 2, w // 2
        axial = rois[:, :, :, cz, :, :]
        coronal = rois[:, :, :, :, cy, :]
        sagittal = rois[:, :, :, :, :, cx]
        views = torch.stack([axial, coronal, sagittal], dim=2)
        views = views.reshape(bsz * topk * 3, channels, h, w)
        view_features = self.encoder2d(views).reshape(bsz, topk, 3, -1)
        view_scores = self.view_attention(view_features).squeeze(-1)
        view_weights = F.softmax(view_scores, dim=2)
        roi_features = (view_features * view_weights.unsqueeze(-1)).sum(dim=2)
        return roi_features, view_weights


class AttentionMIL(nn.Module):
    def __init__(self, feat_dim=192, hidden_dim=96):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, roi_features):
        scores = self.attn(roi_features).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        bag_feature = (roi_features * weights.unsqueeze(-1)).sum(dim=1)
        return bag_feature, weights


def set_module_trainable(module, trainable):
    for p in module.parameters():
        p.requires_grad = trainable


def heatmap_smoothness_loss(heatmap):
    dz = torch.abs(heatmap[:, :, 1:] - heatmap[:, :, :-1]).mean()
    dy = torch.abs(heatmap[:, :, :, 1:] - heatmap[:, :, :, :-1]).mean()
    dx = torch.abs(heatmap[:, :, :, :, 1:] - heatmap[:, :, :, :, :-1]).mean()
    return (dz + dy + dx) / 3.0


def select_topk_centers(heatmap, topk, min_distance, candidates=512):
    bsz, _, d, h, w = heatmap.shape
    smooth = F.avg_pool3d(heatmap, kernel_size=5, stride=1, padding=2)
    centers = []
    for b in range(bsz):
        score = smooth[b, 0].detach().flatten()
        k = min(score.numel(), max(candidates, topk * 64))
        _, indices = torch.topk(score, k=k)
        chosen = []
        for flat in indices.tolist():
            z = flat // (h * w)
            y = (flat % (h * w)) // w
            x = flat % w
            ok = True
            for zz, yy, xx in chosen:
                dist = ((z - zz) ** 2 + (y - yy) ** 2 + (x - xx) ** 2) ** 0.5
                if dist < min_distance:
                    ok = False
                    break
            if ok:
                chosen.append((z, y, x))
            if len(chosen) >= topk:
                break
        while len(chosen) < topk:
            chosen.append(chosen[-1] if chosen else (d // 2, h // 2, w // 2))
        centers.append(chosen[:topk])
    return torch.tensor(centers, device=heatmap.device, dtype=torch.long)


def crop_rois(volume, heatmap, centers, roi_size):
    bsz, channels, d, h, w = volume.shape
    topk = centers.shape[1]
    radius = roi_size // 2
    pad = (radius, radius, radius, radius, radius, radius)
    volume_pad = F.pad(volume, pad, mode="constant", value=0.0)
    heatmap_pad = F.pad(heatmap, pad, mode="constant", value=0.0)
    rois = []
    roi_maps = []
    for b in range(bsz):
        one_rois = []
        one_maps = []
        for k in range(topk):
            z, y, x = centers[b, k].tolist()
            z0, y0, x0 = z, y, x
            one_rois.append(
                volume_pad[
                    b,
                    :,
                    z0 : z0 + roi_size,
                    y0 : y0 + roi_size,
                    x0 : x0 + roi_size,
                ]
            )
            one_maps.append(
                heatmap_pad[
                    b,
                    :,
                    z0 : z0 + roi_size,
                    y0 : y0 + roi_size,
                    x0 : x0 + roi_size,
                ]
            )
        rois.append(torch.stack(one_rois, dim=0))
        roi_maps.append(torch.stack(one_maps, dim=0))
    return torch.stack(rois, dim=0), torch.stack(roi_maps, dim=0)


class MacroMicroMILNet(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=3,
        heads=4,
        topk=6,
        roi_size=32,
        feat_dim=192,
        num_classes=2,
    ):
        super().__init__()
        self.topk = topk
        self.roi_size = roi_size
        self.detector = MacroDetector3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
        )
        self.micro_encoder = TriPlanarMicroEncoder(roi_size=roi_size, feat_dim=feat_dim)
        self.mil = AttentionMIL(feat_dim=feat_dim, hidden_dim=feat_dim // 2)
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, feat_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        det = self.detector(x)
        heatmap = det["heatmap"]
        centers = select_topk_centers(
            heatmap,
            topk=self.topk,
            min_distance=max(4, self.roi_size // 3),
        )
        rois, roi_maps = crop_rois(x, heatmap, centers, self.roi_size)
        gated_rois = rois * (1.0 + roi_maps)
        roi_features, view_weights = self.micro_encoder(gated_rois)
        micro_feature, mil_weights = self.mil(roi_features)
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


def make_ssl_view(x):
    out = x.clone()
    if random.random() < 0.5:
        out = torch.flip(out, dims=[2])
    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])
    if random.random() < 0.5:
        out = torch.flip(out, dims=[4])
    scale = 0.75 + 0.50 * torch.rand(out.shape[0], 1, 1, 1, 1, device=out.device)
    out = out * scale + torch.randn_like(out) * 0.02
    return out.clamp(0.0, 1.0)


def info_nce(z1, z2, temperature=0.15):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def pretrain_detector(model, train_loader, val_loader, args, device):
    if args.detector_epochs <= 0:
        return
    print("\n" + "=" * 70)
    print("Stage 1: unsupervised 3D Transformer macro detector pretraining")
    print("=" * 70)
    optimizer = torch.optim.AdamW(
        model.detector.parameters(),
        lr=args.detector_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.detector_epochs),
        eta_min=args.detector_lr * 0.05,
    )
    best_val = float("inf")
    best_path = os.path.join(args.exp_dir, "checkpoints", "detector_best.pth")
    for epoch in range(args.detector_epochs):
        model.detector.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"pretrain detector {epoch + 1}/{args.detector_epochs}")
        for images, _, _ in pbar:
            images = images.to(device, non_blocking=True)
            v1 = make_ssl_view(images)
            v2 = make_ssl_view(images)
            z1 = model.detector(v1)["ssl_feature"]
            z2 = model.detector(v2)["ssl_feature"]
            loss = info_nce(z1, z2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / max(1, len(train_loader))

        model.detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, _ in tqdm(val_loader, desc="validate detector", leave=False):
                images = images.to(device, non_blocking=True)
                z1 = model.detector(make_ssl_view(images))["ssl_feature"]
                z2 = model.detector(make_ssl_view(images))["ssl_feature"]
                val_loss += info_nce(z1, z2, temperature=args.temperature).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step()
        print(f"Detector train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.detector.state_dict(), "epoch": epoch + 1}, best_path)
            print(f"Saved detector best: val_loss={best_val:.4f}")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.detector.load_state_dict(state["model_state_dict"])
        print(f"Loaded best detector from epoch {state.get('epoch')}")


def compute_metrics(labels, probs, num_classes=2):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "f1_score": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    if len(np.unique(labels)) > 1:
        if num_classes == 2:
            metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            metrics["auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
    else:
        metrics["auc"] = 0.5
    metrics["classification_report"] = classification_report(
        labels,
        preds,
        output_dict=True,
        zero_division=0,
    )
    return metrics, preds


@torch.no_grad()
def evaluate(model, loader, criterion, args, device, split, save_predictions=False):
    model.eval()
    labels_all, probs_all, ids_all = [], [], []
    centers_all, mil_all, view_all = [], [], []
    loss_sum = 0.0
    for images, labels, sample_ids in tqdm(loader, desc=f"evaluate {split}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        loss_sum += loss.item()
        labels_all.extend(labels.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
        ids_all.extend(list(sample_ids))
        if save_predictions:
            centers_all.extend(outputs["roi_centers"].cpu().numpy().tolist())
            mil_all.extend(outputs["mil_weights"].cpu().numpy().tolist())
            view_all.extend(outputs["view_weights"].cpu().numpy().tolist())

    metrics, preds = compute_metrics(labels_all, probs_all, num_classes=args.num_classes)
    avg_loss = loss_sum / max(1, len(loader))
    metrics["loss"] = float(avg_loss)
    if save_predictions:
        rows = []
        probs_all = np.asarray(probs_all)
        for i, sample_id in enumerate(ids_all):
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_label": int(labels_all[i]),
                    "pred_label": int(preds[i]),
                    "dl_score_class1": float(probs_all[i, 1]),
                    "confidence": float(probs_all[i].max()),
                    "correct": int(preds[i] == labels_all[i]),
                    "roi_centers_zyx": json.dumps(centers_all[i]),
                    "roi_mil_weights": json.dumps(mil_all[i]),
                    "view_weights_axial_coronal_sagittal": json.dumps(view_all[i]),
                }
            )
        pred_path = os.path.join(args.exp_dir, "predictions", f"{split}_predictions.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"Saved predictions: {pred_path}")
    return metrics


def train_classifier(model, train_loader, val_loader, test_loader, train_labels, args, device):
    print("\n" + "=" * 70)
    print("Stage 2: macro-to-micro tri-planar MIL classifier")
    print("=" * 70)
    counts = np.bincount(train_labels, minlength=args.num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    detector_params = list(model.detector.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("detector.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": detector_params, "lr": args.detector_finetune_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=True,
    )
    best_auc = -1.0
    best_balanced = -1.0
    no_improve = 0
    history = []
    best_path = os.path.join(args.exp_dir, "checkpoints", "macro_micro_best.pth")

    for epoch in range(args.epochs):
        detector_trainable = (
            (epoch + 1) > args.freeze_detector_epochs and not args.no_detector_finetune
        )
        set_module_trainable(model.detector, detector_trainable)
        optimizer.param_groups[1]["lr"] = args.detector_finetune_lr if detector_trainable else 0.0
        model.train()
        if not detector_trainable:
            model.detector.eval()
        print(f"\nClassifier Epoch {epoch + 1}/{args.epochs}")
        print(f"Detector fine-tuning: {'ON' if detector_trainable else 'OFF'}")

        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="train classifier")
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = heatmap_smoothness_loss(outputs["heatmap"])
            loss = (
                cls_loss
                + args.sparsity_weight * sparse_loss
                + args.smoothness_weight * smooth_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                acc=f"{correct / max(1, total):.3f}",
            )

        train_loss = loss_sum / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        val_metrics = evaluate(model, val_loader, criterion, args, device, "val")
        scheduler.step(val_metrics["auc"])

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1_score"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "detector_finetune": int(detector_trainable),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(args.exp_dir, "metrics", "history.csv"),
            index=False,
        )
        print(
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}, "
            f"balanced_acc: {val_metrics['balanced_accuracy']:.4f}"
        )

        improved = (
            val_metrics["auc"] > best_auc + args.min_delta
            or (
                abs(val_metrics["auc"] - best_auc) <= args.min_delta
                and val_metrics["balanced_accuracy"] > best_balanced + args.min_delta
            )
        )
        if improved:
            best_auc = val_metrics["auc"]
            best_balanced = val_metrics["balanced_accuracy"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )
            print(
                f"Saved best model: val_auc={best_auc:.4f}, "
                f"balanced_acc={best_balanced:.4f}"
            )
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"Early stopped at epoch {epoch + 1}")
                break

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"\nLoaded best model from epoch {state.get('epoch')}")
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        args,
        device,
        "test",
        save_predictions=True,
    )
    with open(os.path.join(args.exp_dir, "metrics", "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("Final official test metrics")
    print("=" * 70)
    for key in [
        "accuracy",
        "auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "mcc",
    ]:
        print(f"{key}: {test_metrics[key]:.4f}")
    print("confusion_matrix:", test_metrics["confusion_matrix"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Macro-to-micro MIL network on NoduleMNIST3D."
    )
    parser.add_argument("--root", default="data/public/NoduleMNIST3D")
    parser.add_argument("--output-dir", default="./Nodule_MacroMicro_Output")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shape-intensity-reg", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--detector-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--detector-lr", type=float, default=1e-4)
    parser.add_argument("--detector-finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--freeze-detector-epochs", type=int, default=10)
    parser.add_argument("--no-detector-finetune", action="store_true")
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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.exp_dir = ensure_dir(os.path.join(args.output_dir, f"nodule_macro_micro_{stamp}"))
    ensure_dir(os.path.join(args.exp_dir, "checkpoints"))
    ensure_dir(os.path.join(args.exp_dir, "metrics"))
    ensure_dir(os.path.join(args.exp_dir, "predictions"))

    info = INFO.get("nodulemnist3d", {})
    print("=" * 70)
    print("NoduleMNIST3D Macro-to-Micro MIL Benchmark")
    print("=" * 70)
    print("Root:", args.root)
    print("Output:", args.exp_dir)
    print("Device:", device)
    print("Task:", info.get("task", "binary-class"))
    print("Labels:", info.get("label", {"0": "benign", "1": "malignant"}))

    train_set = Nodule3DDataset(
        "train",
        args.root,
        args.size,
        augment=True,
        shape_intensity_reg=args.shape_intensity_reg,
        noise_std=args.noise_std,
    )
    val_set = Nodule3DDataset(
        "val",
        args.root,
        args.size,
        augment=False,
        shape_intensity_reg=args.shape_intensity_reg,
    )
    test_set = Nodule3DDataset(
        "test",
        args.root,
        args.size,
        augment=False,
        shape_intensity_reg=args.shape_intensity_reg,
    )
    print(f"Official split counts: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
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
    train_labels = [int(np.asarray(train_set.dataset[i][1]).reshape(-1)[0]) for i in range(len(train_set))]
    print("Train class distribution:", np.bincount(train_labels, minlength=args.num_classes).tolist())

    model = MacroMicroMILNet(
        img_size=args.size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        topk=args.topk,
        roi_size=args.roi_size,
        feat_dim=args.feat_dim,
        num_classes=args.num_classes,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
    with open(os.path.join(args.exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    pretrain_detector(model, train_loader, val_loader, args, device)
    train_classifier(model, train_loader, val_loader, test_loader, train_labels, args, device)


if __name__ == "__main__":
    main()
