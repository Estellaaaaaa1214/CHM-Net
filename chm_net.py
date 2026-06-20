#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro-to-Micro Density-Aware MIL Network

This script keeps the original data format and detector pretraining idea:
3 MRI modalities per patient:
    IMG_ROOT / patient_id / T1WI   / patient_id.nii.gz
    IMG_ROOT / patient_id / T1WI+C / patient_id.nii.gz
    IMG_ROOT / patient_id / T2WI   / patient_id.nii.gz

Classification is changed from:
    detector heatmap -> top 2D slices -> 2D CNN

to:
    detector heatmap -> top-K 3D ROIs -> ROI heatmap gating
    -> tri-planar 2D micro encoder -> attention MIL -> patient label
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import time
import traceback
from datetime import datetime

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2026)


class EnhancedConfig:
    """Configuration for the anonymous CHM-Net release."""

    # Data paths
    IMG_ROOT = ""
    EXCEL_PATH = ""
    ID_COL = "case_id"
    CLS_COL = "label"

    # Macro detector
    DETECTOR_IMG_SIZE = (128, 128, 128)
    IN_CHANNELS = 3
    EMBED_DIM = 384
    NUM_HEADS = 12
    DEPTH = 6
    DROP_RATE = 0.1
    PATCH_SIZE = 16

    # Macro-to-micro bridge
    NUM_ROIS = 6
    ROI_SIZE = 32
    ROI_SUPPRESS_RADIUS = 16
    ROI_GATE_MODE = "residual"  # "residual" is safer than hard multiplication.
    ROI_GATE_ALPHA = 1.0

    # Tri-planar micro encoder
    MICRO_INPUT_SIZE = 128
    MICRO_EMBED_DIM = 256
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.45

    # Training
    DETECTOR_BATCH_SIZE = 4
    CLASSIFIER_BATCH_SIZE = 2
    DETECTOR_EPOCHS = 100
    CLASSIFIER_EPOCHS = 150
    DETECTOR_LR = 8e-5
    CLASSIFIER_LR = 1.5e-4
    FINETUNE_DETECTOR_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.1
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.10
    TEST_RATIO = 0.20
    SPLIT_SEED = 42
    CV_FOLDS = 5
    INNER_VAL_RATIO = 0.20
    SINGLE_SPLIT = False
    EARLY_STOP_PATIENCE = 30
    WARMUP_EPOCHS = 10
    FREEZE_DETECTOR_EPOCHS = 10**9
    LABEL_SMOOTHING = 0.10
    SPARSITY_WEIGHT = 0.01
    SMOOTHNESS_WEIGHT = 0.05

    # Runtime
    NUM_WORKERS = 0
    PIN_MEMORY = True
    OUTPUT_DIR = "CHM-Net_GBNPC2026_Output"
    EXPERIMENT_NAME = f"CHM-Net_GBNPC2026_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    DETECTOR_CKPT = ""
    RESUME_CKPT = ""
    SKIP_DETECTOR_PRETRAIN = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args=None):
        if args is not None:
            if args.img_root:
                self.IMG_ROOT = args.img_root
            if args.excel_path:
                self.EXCEL_PATH = args.excel_path
            if args.label_file:
                self.EXCEL_PATH = args.label_file
            if args.id_col:
                self.ID_COL = args.id_col
            if args.label_col:
                self.CLS_COL = args.label_col
            if args.output_dir:
                self.OUTPUT_DIR = args.output_dir
            if args.detector_ckpt:
                self.DETECTOR_CKPT = args.detector_ckpt
            if args.resume_ckpt:
                self.RESUME_CKPT = args.resume_ckpt
            if args.skip_detector_pretrain:
                self.SKIP_DETECTOR_PRETRAIN = True
            if args.detector_epochs is not None:
                self.DETECTOR_EPOCHS = args.detector_epochs
            if args.classifier_epochs is not None:
                self.CLASSIFIER_EPOCHS = args.classifier_epochs
            if args.num_rois is not None:
                self.NUM_ROIS = args.num_rois
            if args.roi_size is not None:
                self.ROI_SIZE = args.roi_size
                self.ROI_SUPPRESS_RADIUS = max(1, args.roi_size // 2)
            if args.classifier_batch_size is not None:
                self.CLASSIFIER_BATCH_SIZE = args.classifier_batch_size
            if args.train_ratio is not None:
                self.TRAIN_RATIO = args.train_ratio
            if args.val_ratio is not None:
                self.VAL_RATIO = args.val_ratio
            if args.test_ratio is not None:
                self.TEST_RATIO = args.test_ratio
            if args.split_seed is not None:
                self.SPLIT_SEED = args.split_seed
            if args.cv_folds is not None:
                self.CV_FOLDS = args.cv_folds
            if args.inner_val_ratio is not None:
                self.INNER_VAL_RATIO = args.inner_val_ratio
            if args.single_split:
                self.SINGLE_SPLIT = True
            if args.freeze_detector_epochs is not None:
                self.FREEZE_DETECTOR_EPOCHS = args.freeze_detector_epochs
            if args.no_detector_finetune:
                self.FREEZE_DETECTOR_EPOCHS = 10**9

        self.exp_dir = os.path.join(self.OUTPUT_DIR, self.EXPERIMENT_NAME)
        for d in [
            "checkpoints",
            "logs",
            "metrics",
            "predictions",
            "features",
            "visualizations",
        ]:
            os.makedirs(os.path.join(self.exp_dir, d), exist_ok=True)

        print(f"Experiment dir: {self.exp_dir}")
        print(f"Device: {self.device}")
        print(f"Image root: {self.IMG_ROOT}")
        print(f"Excel path: {self.EXCEL_PATH}")
        print(f"Detector epochs: {self.DETECTOR_EPOCHS}")
        print(f"Classifier epochs: {self.CLASSIFIER_EPOCHS}")
        print(f"Top-K ROIs: {self.NUM_ROIS}, ROI size: {self.ROI_SIZE}")


def load_nifti_volume(path):
    try:
        if not os.path.exists(path):
            return None
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        if len(data.shape) == 4:
            data = data[..., 0]
        return data
    except Exception:
        return None


def normalize_volume(volume):
    if volume is None or volume.size == 0:
        return np.zeros((128, 128, 128), dtype=np.float32)
    if volume.std() > 0:
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    return volume.astype(np.float32)


def resize_volume(volume, target_shape):
    if volume is None or volume.size == 0:
        return np.zeros(target_shape, dtype=np.float32)
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32)
    try:
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return ndimage.zoom(volume, factors, order=1).astype(np.float32)
    except Exception:
        return np.zeros(target_shape, dtype=np.float32)


def resolve_modality_path(img_root, pid, modality):
    path = os.path.join(img_root, pid, modality, f"{pid}.nii.gz")
    if os.path.exists(path):
        return path
    if modality == "T1WI+C":
        alt_path = os.path.join(img_root, pid, "T1WI_C", f"{pid}.nii.gz")
        if os.path.exists(alt_path):
            return alt_path
    return path


def load_patient_3d(pid, config):
    modalities = []
    for mod in ["T1WI", "T1WI+C", "T2WI"]:
        path = resolve_modality_path(config.IMG_ROOT, pid, mod)
        volume = load_nifti_volume(path)
        if volume is None:
            volume = np.zeros(config.DETECTOR_IMG_SIZE, dtype=np.float32)
        volume = resize_volume(volume, config.DETECTOR_IMG_SIZE)
        volume = normalize_volume(volume)
        modalities.append(volume)
    return np.stack(modalities, axis=0).astype(np.float32)


def has_all_modalities(img_root, pid):
    for mod in ["T1WI", "T1WI+C", "T2WI"]:
        path = resolve_modality_path(img_root, pid, mod)
        if not os.path.exists(path):
            return False
    return True


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        batch_size, channels, depth, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (depth, height, width)


class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim, max_patches=512):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        return x + self.position_embedding[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class EnhancedUnsupervisedTransformerDetector(nn.Module):
    """Macro detector. It now returns heatmap + global macro feature."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding3D(
            in_channels=config.IN_CHANNELS,
            embed_dim=config.EMBED_DIM,
            patch_size=config.PATCH_SIZE,
        )
        self.pos_embed = PositionalEncoding3D(config.EMBED_DIM)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.EMBED_DIM,
                    num_heads=config.NUM_HEADS,
                    drop_rate=config.DROP_RATE,
                )
                for _ in range(config.DEPTH)
            ]
        )
        self.norm = nn.LayerNorm(config.EMBED_DIM)
        self.projection_head = nn.Sequential(
            nn.Linear(config.EMBED_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.detection_head = nn.Sequential(
            nn.Conv3d(config.EMBED_DIM, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_dict=False):
        batch_size, channels, depth, height, width = x.shape
        patch_embeddings, patch_shape = self.patch_embed(x)
        patch_embeddings = self.pos_embed(patch_embeddings)
        for block in self.blocks:
            patch_embeddings = block(patch_embeddings)
        patch_embeddings = self.norm(patch_embeddings)

        patch_embeddings_3d = patch_embeddings.transpose(1, 2).reshape(
            batch_size,
            self.config.EMBED_DIM,
            *patch_shape,
        )
        detection_map = self.detection_head(patch_embeddings_3d)
        detection_map = F.interpolate(
            detection_map,
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        )

        global_features = patch_embeddings.mean(dim=1)
        if return_dict:
            return {
                "detection_map": detection_map,
                "projected_features": None,
                "global_features": global_features,
                "patch_features_3d": patch_embeddings_3d,
            }
        projected_features = self.projection_head(global_features)
        return detection_map, projected_features


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features1, features2):
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        batch_size = features1.shape[0]
        labels = torch.arange(batch_size, device=features1.device)
        loss_12 = self.criterion(similarity_matrix, labels)
        loss_21 = self.criterion(similarity_matrix.T, labels)
        return (loss_12 + loss_21) / 2


class UnsupervisedDataset(Dataset):
    def __init__(self, patient_ids, config):
        self.patient_ids = patient_ids
        self.config = config

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = str(self.patient_ids[idx]).strip()
        image = load_patient_3d(pid, self.config)
        return torch.FloatTensor(image), pid


class Supervised3DVolumeDataset(Dataset):
    """Patient-level dataset for the new macro-to-micro classifier."""

    def __init__(self, patient_ids, labels_by_pid, config, augment=False, mode="train"):
        self.patient_ids = [str(pid).strip() for pid in patient_ids]
        self.labels_by_pid = labels_by_pid
        self.config = config
        self.augment = augment
        self.mode = mode

    def __len__(self):
        return len(self.patient_ids)

    def _augment_volume(self, image):
        if not self.augment or self.mode != "train":
            return image

        # image: [C, D, H, W]
        if random.random() < 0.5:
            image = np.flip(image, axis=3).copy()
        if random.random() < 0.3:
            image = np.flip(image, axis=2).copy()
        if random.random() < 0.2:
            image = np.flip(image, axis=1).copy()
        if random.random() < 0.5:
            image = image * random.uniform(0.9, 1.1)
        if random.random() < 0.3:
            image = image + np.random.normal(0, 0.03, size=image.shape).astype(np.float32)
        return image.astype(np.float32)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        image = load_patient_3d(pid, self.config)
        image = self._augment_volume(image)
        label = int(self.labels_by_pid[pid])
        return torch.FloatTensor(image), torch.tensor(label, dtype=torch.long), pid


def create_enhanced_augmented_views(images, config):
    batch_size, channels, depth, height, width = images.shape
    crop_size = min(96, depth, height, width)

    view1_list, view2_list = [], []
    for i in range(batch_size):
        d_start = random.randint(0, depth - crop_size)
        h_start = random.randint(0, height - crop_size)
        w_start = random.randint(0, width - crop_size)
        view1 = images[
            i,
            :,
            d_start : d_start + crop_size,
            h_start : h_start + crop_size,
            w_start : w_start + crop_size,
        ]

        d_start2 = random.randint(0, depth - crop_size)
        h_start2 = random.randint(0, height - crop_size)
        w_start2 = random.randint(0, width - crop_size)
        view2 = images[
            i,
            :,
            d_start2 : d_start2 + crop_size,
            h_start2 : h_start2 + crop_size,
            w_start2 : w_start2 + crop_size,
        ]

        if random.random() < 0.3:
            view2 = view2 + torch.randn_like(view2) * 0.05
        if random.random() < 0.3:
            view2 = view2 * random.uniform(0.9, 1.1)

        view1_list.append(view1)
        view2_list.append(view2)

    return torch.stack(view1_list, dim=0), torch.stack(view2_list, dim=0)


class TopKROISampler3D(nn.Module):
    """Discrete top-K ROI mining from the 3D detector heatmap."""

    def __init__(self, roi_size=32, num_rois=6, suppress_radius=16):
        super().__init__()
        self.roi_size = int(roi_size)
        self.num_rois = int(num_rois)
        self.suppress_radius = int(suppress_radius)

    @staticmethod
    def _crop_3d(volume, center, roi_size):
        # volume: [C, D, H, W]
        channels, depth, height, width = volume.shape
        half = roi_size // 2
        z, y, x = [int(v) for v in center]

        z = min(max(z, half), depth - (roi_size - half))
        y = min(max(y, half), height - (roi_size - half))
        x = min(max(x, half), width - (roi_size - half))

        z1, y1, x1 = z - half, y - half, x - half
        z2, y2, x2 = z1 + roi_size, y1 + roi_size, x1 + roi_size
        return volume[:, z1:z2, y1:y2, x1:x2], (z, y, x)

    def forward(self, images, heatmaps):
        # images: [B, C, D, H, W], heatmaps: [B, 1, D, H, W]
        batch_size, channels, depth, height, width = images.shape
        rois, roi_heatmaps, coords, roi_scores = [], [], [], []
        score_maps = F.avg_pool3d(
            heatmaps,
            kernel_size=3,
            stride=1,
            padding=1,
        ).detach()

        for b in range(batch_size):
            score_map = score_maps[b, 0].clone()
            patient_rois, patient_maps, patient_coords, patient_scores = [], [], [], []

            for _ in range(self.num_rois):
                flat_idx = torch.argmax(score_map).item()
                z = flat_idx // (height * width)
                rem = flat_idx % (height * width)
                y = rem // width
                x = rem % width

                roi, center = self._crop_3d(images[b], (z, y, x), self.roi_size)
                roi_map, _ = self._crop_3d(heatmaps[b], center, self.roi_size)

                patient_rois.append(roi)
                patient_maps.append(roi_map)
                patient_coords.append(torch.tensor(center, device=images.device, dtype=torch.long))
                patient_scores.append(heatmaps[b, 0, center[0], center[1], center[2]])

                rz = self.suppress_radius
                z1, z2 = max(0, center[0] - rz), min(depth, center[0] + rz + 1)
                y1, y2 = max(0, center[1] - rz), min(height, center[1] + rz + 1)
                x1, x2 = max(0, center[2] - rz), min(width, center[2] + rz + 1)
                score_map[z1:z2, y1:y2, x1:x2] = -1.0

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


class TriPlanarMicroEncoder(nn.Module):
    """
    Encodes each 3D ROI with three attention-weighted 2D projections.
    This is a 2.5D alternative to heavy 3D CNNs.
    """

    def __init__(self, in_channels=3, embed_dim=256, input_size=128, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.encoder2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.view_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    @staticmethod
    def _normalize_2d(x):
        # x: [N, C, H, W]
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _attention_projection(self, roi, roi_heatmap, axis):
        # roi: [N, C, S, S, S], heatmap: [N, 1, S, S, S]
        numerator = (roi * roi_heatmap).sum(dim=axis)
        denominator = roi_heatmap.sum(dim=axis).clamp_min(1e-6)
        projected = numerator / denominator
        return projected

    def forward(self, rois, roi_heatmaps, gate_mode="residual", gate_alpha=1.0):
        # rois: [B, K, C, S, S, S]
        batch_size, num_rois, channels, size_d, size_h, size_w = rois.shape
        n = batch_size * num_rois
        rois = rois.reshape(n, channels, size_d, size_h, size_w)
        roi_heatmaps = roi_heatmaps.reshape(n, 1, size_d, size_h, size_w)

        if gate_mode == "multiply":
            gated = rois * roi_heatmaps
        else:
            gated = rois * (1.0 + gate_alpha * roi_heatmaps)

        # Attention-weighted projections:
        # axial: sum along D -> [N, C, H, W]
        # coronal: sum along H -> [N, C, D, W]
        # sagittal: sum along W -> [N, C, D, H]
        axial = self._attention_projection(gated, roi_heatmaps, axis=2)
        coronal = self._attention_projection(gated, roi_heatmaps, axis=3)
        sagittal = self._attention_projection(gated, roi_heatmaps, axis=4)

        views = torch.stack([axial, coronal, sagittal], dim=1)  # [N, 3 views, C, S, S]
        views = views.reshape(n * 3, channels, views.size(-2), views.size(-1))
        views = F.interpolate(
            views,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        views = self._normalize_2d(views)

        view_features = self.proj(self.encoder2d(views))
        view_features = view_features.reshape(n, 3, -1)
        view_logits = self.view_attention(view_features).squeeze(-1)
        view_weights = F.softmax(view_logits, dim=1)
        roi_features = torch.sum(view_features * view_weights.unsqueeze(-1), dim=1)
        roi_features = roi_features.reshape(batch_size, num_rois, -1)
        view_weights = view_weights.reshape(batch_size, num_rois, 3)
        return roi_features, view_weights


class MacroMicroMILHead(nn.Module):
    def __init__(self, macro_dim=384, micro_dim=256, num_classes=2, dropout=0.45):
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Linear(micro_dim * 2, 256),
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
        macro_feature = self.macro_proj(global_features)
        mil_logits = self.mil_attention(micro_features).squeeze(-1)
        if roi_scores is not None:
            roi_bias = (roi_scores - roi_scores.mean(dim=1, keepdim=True)) / (
                roi_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
            )
            mil_logits = mil_logits + 0.25 * roi_bias
        mil_weights = F.softmax(mil_logits, dim=1)
        patient_micro = torch.sum(micro_features * mil_weights.unsqueeze(-1), dim=1)
        fused = torch.cat([macro_feature, patient_micro], dim=1)
        logits = self.classifier(fused)
        return logits, mil_weights


class MacroToMicroDensityNet(nn.Module):
    def __init__(self, detector, config):
        super().__init__()
        self.config = config
        self.detector = detector
        self.roi_sampler = TopKROISampler3D(
            roi_size=config.ROI_SIZE,
            num_rois=config.NUM_ROIS,
            suppress_radius=config.ROI_SUPPRESS_RADIUS,
        )
        self.micro_encoder = TriPlanarMicroEncoder(
            in_channels=config.IN_CHANNELS,
            embed_dim=config.MICRO_EMBED_DIM,
            input_size=config.MICRO_INPUT_SIZE,
            dropout=0.2,
        )
        self.mil_head = MacroMicroMILHead(
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
        micro_features, view_weights = self.micro_encoder(
            rois,
            roi_heatmaps,
            gate_mode=self.config.ROI_GATE_MODE,
            gate_alpha=self.config.ROI_GATE_ALPHA,
        )
        logits, mil_weights = self.mil_head(global_features, micro_features, roi_scores)
        return {
            "logits": logits,
            "heatmap": heatmap,
            "roi_coords": roi_coords,
            "roi_scores": roi_scores,
            "mil_weights": mil_weights,
            "view_weights": view_weights,
        }


class EnhancedWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (
                1 - math.cos(math.pi * (self.current_epoch + 1) / self.warmup_epochs)
            ) / 2
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        self.current_epoch += 1
        return self.optimizer.param_groups[0]["lr"]


def attention_smoothness_loss(heatmap):
    dz = torch.abs(heatmap[:, :, 1:, :, :] - heatmap[:, :, :-1, :, :]).mean()
    dy = torch.abs(heatmap[:, :, :, 1:, :] - heatmap[:, :, :, :-1, :]).mean()
    dx = torch.abs(heatmap[:, :, :, :, 1:] - heatmap[:, :, :, :, :-1]).mean()
    return (dx + dy + dz) / 3.0


def set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def compute_class_weights(labels, device):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.FloatTensor(weights).to(device)


def compute_enhanced_metrics(true, pred, probs):
    if len(true) == 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "auc": 0.5,
            "sensitivity": 0,
            "specificity": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "sample_stats": {"total": 0, "class_distribution": [0, 0]},
            "balanced_accuracy": 0,
            "mcc": 0,
        }

    accuracy = float(accuracy_score(true, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        pred,
        average="binary",
        zero_division=0,
    )

    from sklearn.metrics import matthews_corrcoef

    try:
        mcc = float(matthews_corrcoef(true, pred))
    except Exception:
        mcc = 0.0

    try:
        if len(set(true)) > 1:
            auc_score = float(roc_auc_score(true, probs))
            fpr, tpr, _ = roc_curve(true, probs)
        else:
            auc_score = 0.5
            fpr, tpr = [], []
    except Exception:
        auc_score = 0.5
        fpr, tpr = [], []

    cm = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    cls_report = classification_report(
        true,
        pred,
        labels=[0, 1],
        target_names=["low_density", "high_density"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc_score),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist() if hasattr(fpr, "tolist") else fpr,
            "tpr": tpr.tolist() if hasattr(tpr, "tolist") else tpr,
        },
        "classification_report": cls_report,
        "sample_stats": {
            "total": len(true),
            "class_distribution": np.bincount(true, minlength=2).tolist(),
        },
    }


def train_enhanced_unsupervised_detector(config, patient_ids):
    print("\n" + "=" * 60)
    print("Stage 1: unsupervised macro detector pretraining")
    print("=" * 60)

    train_pids, val_pids = train_test_split(patient_ids, test_size=0.2, random_state=2024)
    print(f"Detector train: {len(train_pids)}, val: {len(val_pids)}")

    train_dataset = UnsupervisedDataset(train_pids, config)
    val_dataset = UnsupervisedDataset(val_pids, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = EnhancedUnsupervisedTransformerDetector(config).to(config.device)
    detector_params = sum(p.numel() for p in model.parameters())
    print(f"Detector parameters: {detector_params:,}")

    criterion = NTXentLoss(temperature=config.TEMPERATURE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.DETECTOR_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-7,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "best_loss": float("inf"),
        "best_epoch": 0,
        "parameters": detector_params,
    }
    early_stop_counter = 0

    for epoch in range(config.DETECTOR_EPOCHS):
        print(f"\nDetector Epoch {epoch + 1}/{config.DETECTOR_EPOCHS}")
        model.train()
        train_loss, train_steps = 0.0, 0

        pbar = tqdm(train_loader, desc="train detector")
        for images, _ in pbar:
            images = images.to(config.device, non_blocking=True)
            view1, view2 = create_enhanced_augmented_views(images, config)

            optimizer.zero_grad()
            _, features1 = model(view1)
            _, features2 = model(view2)
            loss = criterion(features1, features2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(config.device, non_blocking=True)
                view1, view2 = create_enhanced_augmented_views(images, config)
                _, features1 = model(view1)
                _, features2 = model(view2)
                loss = criterion(features1, features2)
                val_loss += loss.item()
                val_steps += 1

        avg_train_loss = train_loss / max(1, train_steps)
        avg_val_loss = val_loss / max(1, val_steps)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["learning_rate"].append(float(lr))

        print(f"Train loss: {avg_train_loss:.6f}")
        print(f"Val loss: {avg_val_loss:.6f}")
        print(f"LR: {lr:.8f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val_loss),
                    "config": config.__dict__,
                },
                os.path.join(config.exp_dir, "checkpoints", f"detector_epoch_{epoch + 1}.pth"),
            )

        if avg_val_loss < history["best_loss"]:
            history["best_loss"] = float(avg_val_loss)
            history["best_epoch"] = epoch + 1
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val_loss),
                    "config": config.__dict__,
                },
                os.path.join(config.exp_dir, "checkpoints", "detector_best.pth"),
            )
            print(f"Saved best detector: val_loss={avg_val_loss:.6f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Detector early stopped at epoch {epoch + 1}")
            break

    best_path = os.path.join(config.exp_dir, "checkpoints", "detector_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best detector from epoch {checkpoint['epoch']}")

    with open(
        os.path.join(config.exp_dir, "metrics", "detector_training_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return model, history


def load_detector_from_checkpoint(config):
    model = EnhancedUnsupervisedTransformerDetector(config).to(config.device)
    checkpoint = torch.load(config.DETECTOR_CKPT, map_location=config.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded detector checkpoint: {config.DETECTOR_CKPT}")
    history = {
        "best_loss": None,
        "best_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, history



def ensure_experiment_dirs(config):
    for d in [
        "checkpoints",
        "logs",
        "metrics",
        "predictions",
        "features",
        "visualizations",
    ]:
        os.makedirs(os.path.join(config.exp_dir, d), exist_ok=True)


def build_labels_by_pid(valid_patients, df, config):
    labels_by_pid = {}
    labeled_patients = []
    for pid in valid_patients:
        pid_str = str(pid).strip()
        values = df[df[config.ID_COL] == pid_str][config.CLS_COL].values
        if len(values) > 0 and not pd.isna(values[0]):
            label = int(values[0])
            if label in [0, 1]:
                labels_by_pid[pid_str] = label
                labeled_patients.append(pid_str)
    return labeled_patients, labels_by_pid


def make_single_split(labeled_patients, labels_by_pid, config):
    split_sum = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    if not np.isclose(split_sum, 1.0):
        raise ValueError(
            "For --single-split, --train-ratio + --val-ratio + --test-ratio must equal 1.0."
        )
    labels = [labels_by_pid[pid] for pid in labeled_patients]
    train_val_ids, test_ids = train_test_split(
        labeled_patients,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=config.SPLIT_SEED,
    )
    train_val_labels = [labels_by_pid[pid] for pid in train_val_ids]
    val_ratio_within_train_val = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio_within_train_val,
        stratify=train_val_labels,
        random_state=config.SPLIT_SEED + 1,
    )
    return train_ids, val_ids, test_ids


def make_five_fold_splits(labeled_patients, labels_by_pid, config):
    labels = np.asarray([labels_by_pid[pid] for pid in labeled_patients], dtype=np.int64)
    patient_ids = np.asarray(labeled_patients, dtype=object)
    splitter = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.SPLIT_SEED,
    )

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(patient_ids, labels), start=1):
        train_val_ids = patient_ids[train_val_idx].tolist()
        test_ids = patient_ids[test_idx].tolist()
        train_val_labels = [labels_by_pid[pid] for pid in train_val_ids]
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=config.INNER_VAL_RATIO,
            stratify=train_val_labels,
            random_state=config.SPLIT_SEED + fold_idx,
        )
        folds.append(
            {
                "fold": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            }
        )
    return folds


def make_fold_config(config, fold_idx):
    fold_config = copy.copy(config)
    fold_config.EXPERIMENT_NAME = f"{config.EXPERIMENT_NAME}_fold{fold_idx}"
    fold_config.exp_dir = os.path.join(config.exp_dir, f"fold_{fold_idx}")
    ensure_experiment_dirs(fold_config)
    return fold_config

def build_classifier_optimizer(model, config):
    detector_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith("detector."):
            detector_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": config.CLASSIFIER_LR},
            {"params": detector_params, "lr": config.FINETUNE_DETECTOR_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def train_macro_micro_classifier(
    config,
    detector,
    labels_by_pid,
    train_ids,
    val_ids,
    test_ids,
    split_name="single_split",
):
    print("\n" + "=" * 60)
    print(f"Stage 2: macro-to-micro tri-planar MIL classifier ({split_name})")
    print("=" * 60)

    print(
        f"Classifier split: train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    train_labels = [labels_by_pid[pid] for pid in train_ids]
    val_labels = [labels_by_pid[pid] for pid in val_ids]
    test_labels = [labels_by_pid[pid] for pid in test_ids]
    print(f"Train class distribution: {np.bincount(train_labels, minlength=2).tolist()}")
    print(f"Val class distribution: {np.bincount(val_labels, minlength=2).tolist()}")
    print(f"Test class distribution: {np.bincount(test_labels, minlength=2).tolist()}")

    train_dataset = Supervised3DVolumeDataset(
        train_ids,
        labels_by_pid,
        config,
        augment=True,
        mode="train",
    )
    val_dataset = Supervised3DVolumeDataset(
        val_ids,
        labels_by_pid,
        config,
        augment=False,
        mode="val",
    )
    test_dataset = Supervised3DVolumeDataset(
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

    model = MacroToMicroDensityNet(detector, config).to(config.device)
    classifier_params = sum(p.numel() for p in model.parameters())
    print(f"Macro-to-micro classifier parameters: {classifier_params:,}")

    class_weights = compute_class_weights(train_labels, config.device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    )
    optimizer = build_classifier_optimizer(model, config)
    warmup_scheduler = EnhancedWarmupScheduler(
        optimizer,
        config.WARMUP_EPOCHS,
        config.CLASSIFIER_LR,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=12,
        verbose=True,
        min_lr=1e-7,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "val_recall": [],
        "val_specificity": [],
        "val_balanced_acc": [],
        "learning_rate": [],
        "best_val_balanced_acc": 0.0,
        "best_val_acc": 0.0,
        "best_val_auc": 0.0,
        "best_val_f1": 0.0,
        "best_epoch": 0,
        "parameters": classifier_params,
    }
    early_stop_counter = 0
    start_epoch = 0

    if config.RESUME_CKPT:
        print(f"Resuming macro-to-micro classifier from: {config.RESUME_CKPT}")
        checkpoint = torch.load(config.RESUME_CKPT, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                print(f"Optimizer state was not restored: {exc}")
        start_epoch = int(checkpoint.get("epoch", 0))
        val_metrics = checkpoint.get("val_metrics", {})
        if val_metrics:
            history["best_val_balanced_acc"] = float(
                val_metrics.get("balanced_accuracy", history["best_val_balanced_acc"])
            )
            history["best_val_acc"] = float(val_metrics.get("accuracy", history["best_val_acc"]))
            history["best_val_auc"] = float(val_metrics.get("auc", history["best_val_auc"]))
            history["best_val_f1"] = float(val_metrics.get("f1_score", history["best_val_f1"]))
            history["best_epoch"] = start_epoch
        print(f"Resume will continue from classifier epoch {start_epoch + 1}.")

    for epoch in range(start_epoch, config.CLASSIFIER_EPOCHS):
        print(f"\nClassifier Epoch {epoch + 1}/{config.CLASSIFIER_EPOCHS}")

        detector_trainable = epoch >= config.FREEZE_DETECTOR_EPOCHS
        set_module_trainable(model.detector, detector_trainable)
        if detector_trainable:
            print("Detector fine-tuning: ON")
        else:
            print("Detector fine-tuning: OFF")

        if epoch < config.WARMUP_EPOCHS:
            current_lr = warmup_scheduler.step(epoch)
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = (
                config.FINETUNE_DETECTOR_LR if detector_trainable else 0.0
            )

        model.train()
        if not detector_trainable:
            model.detector.eval()

        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="train classifier")
        for images, labels_batch, _ in pbar:
            images = images.to(config.device, non_blocking=True)
            labels_batch = labels_batch.to(config.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels_batch)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = attention_smoothness_loss(outputs["heatmap"])
            loss = (
                cls_loss
                + config.SPARSITY_WEIGHT * sparse_loss
                + config.SMOOTHNESS_WEIGHT * smooth_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total += labels_batch.size(0)
            correct += (preds == labels_batch).sum().item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "acc": f"{100 * correct / max(1, total):.2f}%",
                }
            )

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        val_metrics, avg_val_loss = evaluate_macro_micro_classifier(
            model,
            val_loader,
            criterion,
            config,
            save_predictions=False,
        )
        val_acc = val_metrics["accuracy"]
        f1 = val_metrics["f1_score"]
        auc_score = val_metrics["auc"]
        recall = val_metrics["recall"]
        specificity = val_metrics["specificity"]
        balanced_acc = val_metrics["balanced_accuracy"]

        if epoch >= config.WARMUP_EPOCHS:
            plateau_scheduler.step(balanced_acc)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(avg_train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(avg_val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(f1))
        history["val_auc"].append(float(auc_score))
        history["val_recall"].append(float(recall))
        history["val_specificity"].append(float(specificity))
        history["val_balanced_acc"].append(float(balanced_acc))
        history["learning_rate"].append(float(current_lr))

        print(f"Train loss: {avg_train_loss:.4f}, acc: {train_acc:.2f}%")
        print(
            "Val loss: "
            f"{avg_val_loss:.4f}, acc: {val_acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}"
        )
        print(
            "Val recall: "
            f"{recall:.4f}, specificity: {specificity:.4f}, balanced_acc: {balanced_acc:.4f}"
        )
        print(f"LR: {current_lr:.8f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config.__dict__,
                },
                os.path.join(
                    config.exp_dir,
                    "checkpoints",
                    f"macro_micro_epoch_{epoch + 1}.pth",
                ),
            )

        if balanced_acc > history["best_val_balanced_acc"]:
            history["best_val_balanced_acc"] = balanced_acc
            history["best_val_acc"] = val_acc
            history["best_val_auc"] = auc_score
            history["best_val_f1"] = f1
            history["best_epoch"] = epoch + 1
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config.__dict__,
                },
                os.path.join(config.exp_dir, "checkpoints", "macro_micro_best.pth"),
            )
            print(f"Saved best macro-to-micro model: balanced_acc={balanced_acc:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Classifier early stopped at epoch {epoch + 1}")
            break

    best_path = os.path.join(config.exp_dir, "checkpoints", "macro_micro_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best macro-to-micro model from epoch {checkpoint['epoch']}")

    print("\n" + "=" * 40)
    print("Test evaluation")
    print("=" * 40)
    test_metrics, _ = evaluate_macro_micro_classifier(
        model,
        test_loader,
        criterion,
        config,
        save_predictions=True,
    )

    with open(
        os.path.join(config.exp_dir, "metrics", "classifier_training_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    classifier_info = {
        "model_name": "MacroToMicroDensityNet",
        "parameters": classifier_params,
        "num_rois": config.NUM_ROIS,
        "roi_size": config.ROI_SIZE,
        "micro_input_size": config.MICRO_INPUT_SIZE,
        "micro_embed_dim": config.MICRO_EMBED_DIM,
        "roi_gate_mode": config.ROI_GATE_MODE,
        "training_epochs": len(history["epoch"]),
        "best_val_acc": float(history["best_val_acc"]),
        "best_val_auc": float(history["best_val_auc"]),
        "best_val_f1": float(history["best_val_f1"]),
        "best_val_balanced_acc": float(history["best_val_balanced_acc"]),
        "best_epoch": history["best_epoch"],
    }
    with open(
        os.path.join(config.exp_dir, "metrics", "classifier_info.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(classifier_info, f, indent=2, ensure_ascii=False)

    return model, history, test_metrics


def evaluate_macro_micro_classifier(
    model,
    data_loader,
    criterion,
    config,
    save_predictions=False,
):
    model.eval()
    all_preds, all_probs, all_labels, all_pids = [], [], [], []
    all_roi_coords, all_roi_weights, all_view_weights = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels_batch, pids in tqdm(data_loader, desc="evaluate"):
            images = images.to(config.device, non_blocking=True)
            labels_batch = labels_batch.to(config.device, non_blocking=True)

            outputs = model(images)
            logits = outputs["logits"]
            cls_loss = criterion(logits, labels_batch)
            sparse_loss = outputs["heatmap"].mean()
            smooth_loss = attention_smoothness_loss(outputs["heatmap"])
            loss = (
                cls_loss
                + config.SPARSITY_WEIGHT * sparse_loss
                + config.SMOOTHNESS_WEIGHT * smooth_loss
            )
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(labels_batch.cpu().numpy().tolist())
            all_pids.extend(list(pids))
            all_roi_coords.extend(outputs["roi_coords"].cpu().numpy().tolist())
            all_roi_weights.extend(outputs["mil_weights"].cpu().numpy().tolist())
            all_view_weights.extend(outputs["view_weights"].cpu().numpy().tolist())

    metrics = compute_enhanced_metrics(all_labels, np.array(all_preds), all_probs)
    avg_loss = total_loss / max(1, len(data_loader))

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

        with open(
            os.path.join(config.exp_dir, "predictions", "test_predictions.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(config.exp_dir, "metrics", "test_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics, avg_loss


def collect_valid_patients(config, df):
    df[config.ID_COL] = df[config.ID_COL].astype(str).str.strip()
    all_patients = [
        d
        for d in os.listdir(config.IMG_ROOT)
        if os.path.isdir(os.path.join(config.IMG_ROOT, d))
    ]

    valid_patients = []
    for pid in all_patients:
        pid_str = str(pid).strip()
        if pid_str not in df[config.ID_COL].values:
            continue
        if has_all_modalities(config.IMG_ROOT, pid_str):
            valid_patients.append(pid_str)

    labeled_patients = []
    labels = []
    for pid in valid_patients:
        values = df[df[config.ID_COL] == pid][config.CLS_COL].values
        if len(values) > 0 and not pd.isna(values[0]):
            label = int(values[0])
            if label in [0, 1]:
                labeled_patients.append(pid)
                labels.append(label)

    class_dist = np.bincount(labels, minlength=2).tolist() if labels else [0, 0]
    print(f"Total patient folders: {len(all_patients)}")
    print(f"Valid patients with all modalities and label row: {len(valid_patients)}")
    print(f"Labeled patients: {len(labeled_patients)}")
    print(f"Class distribution: low={class_dist[0]}, high={class_dist[1]}")
    return all_patients, valid_patients, labeled_patients, class_dist



def prepare_detector(config, detector_patient_ids):
    detector_start = time.time()
    if config.RESUME_CKPT:
        print("Resume checkpoint provided: skip detector pretraining and load detector through full model checkpoint.")
        detector = EnhancedUnsupervisedTransformerDetector(config).to(config.device)
        detector_history = {
            "best_loss": None,
            "best_epoch": None,
            "parameters": sum(p.numel() for p in detector.parameters()),
        }
    elif config.DETECTOR_CKPT:
        detector, detector_history = load_detector_from_checkpoint(config)
    elif config.SKIP_DETECTOR_PRETRAIN:
        print("Skip detector pretraining: using randomly initialized detector.")
        detector = EnhancedUnsupervisedTransformerDetector(config).to(config.device)
        detector_history = {
            "best_loss": None,
            "best_epoch": None,
            "parameters": sum(p.numel() for p in detector.parameters()),
        }
    else:
        detector, detector_history = train_enhanced_unsupervised_detector(
            config,
            detector_patient_ids,
        )
    detector_time = time.time() - detector_start
    return detector, detector_history, detector_time


def summarize_cv_metrics(fold_results):
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
    ]
    summary = {}
    for key in metric_keys:
        values = [float(item["test_metrics"].get(key, 0.0)) for item in fold_results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return summary


def print_cv_summary(summary):
    print("\n" + "=" * 80)
    print("Five-fold cross-validation summary")
    print("=" * 80)
    for key in ["accuracy", "precision", "recall", "f1_score", "auc", "sensitivity", "specificity"]:
        item = summary[key]
        print(f"{key}: {item['mean']:.4f} +/- {item['std']:.4f}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = EnhancedConfig(args)

    print("=" * 80)
    print("CHM-Net for GBNPC2026")
    print("Stage 1: unsupervised 3D Transformer macro detector")
    print("Stage 2: attention-guided 3D ROI + tri-planar micro MIL classifier")
    print("=" * 80)

    start_time = time.time()
    if not config.IMG_ROOT:
        raise ValueError("Please provide --img-root for GBNPC2026 images.")
    if not config.EXCEL_PATH:
        raise ValueError("Please provide --label-file or --excel-path for GBNPC2026 labels.")
    label_path = str(config.EXCEL_PATH)
    if label_path.lower().endswith(".csv"):
        df = pd.read_csv(label_path)
    else:
        df = pd.read_excel(label_path)
    all_patients, valid_patients, labeled_patients, class_dist = collect_valid_patients(
        config,
        df,
    )

    if len(valid_patients) < 20:
        print("Not enough valid patients. Stop.")
        return 1
    if len(labeled_patients) < 20:
        print("Not enough labeled patients. Stop.")
        return 1

    labeled_patients, labels_by_pid = build_labels_by_pid(labeled_patients, df, config)
    if len(labeled_patients) < config.CV_FOLDS * 2:
        print("Not enough labeled patients for the requested number of CV folds. Stop.")
        return 1

    data_info = {
        "total_patients": len(all_patients),
        "valid_patients": len(valid_patients),
        "labeled_patients": len(labeled_patients),
        "class_distribution": {
            "low_density": int(class_dist[0]),
            "high_density": int(class_dist[1]),
        },
        "split_seed": int(config.SPLIT_SEED),
        "cv_folds": int(config.CV_FOLDS),
        "inner_val_ratio": float(config.INNER_VAL_RATIO),
        "single_split_train_ratio": float(config.TRAIN_RATIO),
        "single_split_val_ratio": float(config.VAL_RATIO),
        "single_split_test_ratio": float(config.TEST_RATIO),
    }
    with open(
        os.path.join(config.exp_dir, "metrics", "data_info.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)

    fold_results = []
    if config.SINGLE_SPLIT:
        train_ids, val_ids, test_ids = make_single_split(labeled_patients, labels_by_pid, config)
        detector, detector_history, detector_time = prepare_detector(config, train_ids + val_ids)
        classifier_start = time.time()
        _, classifier_history, test_metrics = train_macro_micro_classifier(
            config,
            detector,
            labels_by_pid,
            train_ids,
            val_ids,
            test_ids,
            split_name="single_split",
        )
        classifier_time = time.time() - classifier_start
        fold_results.append(
            {
                "fold": 1,
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "detector_history": detector_history,
                "detector_time_seconds": detector_time,
                "classifier_history": classifier_history,
                "classifier_time_seconds": classifier_time,
                "test_metrics": test_metrics,
            }
        )
    else:
        folds = make_five_fold_splits(labeled_patients, labels_by_pid, config)
        for fold in folds:
            fold_idx = fold["fold"]
            fold_config = make_fold_config(config, fold_idx)
            train_ids = fold["train_ids"]
            val_ids = fold["val_ids"]
            test_ids = fold["test_ids"]

            print("\n" + "=" * 80)
            print(f"Fold {fold_idx}/{config.CV_FOLDS}")
            print("=" * 80)
            detector, detector_history, detector_time = prepare_detector(
                fold_config,
                train_ids + val_ids,
            )

            classifier_start = time.time()
            _, classifier_history, test_metrics = train_macro_micro_classifier(
                fold_config,
                detector,
                labels_by_pid,
                train_ids,
                val_ids,
                test_ids,
                split_name=f"fold_{fold_idx}",
            )
            classifier_time = time.time() - classifier_start
            fold_results.append(
                {
                    "fold": fold_idx,
                    "train": len(train_ids),
                    "val": len(val_ids),
                    "test": len(test_ids),
                    "detector_history": detector_history,
                    "detector_time_seconds": detector_time,
                    "classifier_history": classifier_history,
                    "classifier_time_seconds": classifier_time,
                    "test_metrics": test_metrics,
                }
            )

    cv_summary = summarize_cv_metrics(fold_results)
    print_cv_summary(cv_summary)

    total_time = time.time() - start_time
    result = {
        "experiment_info": {
            "experiment_name": config.EXPERIMENT_NAME,
            "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": total_time,
            "device": str(config.device),
            "mode": "single_split" if config.SINGLE_SPLIT else "five_fold_cv",
        },
        "data_info": data_info,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
    }

    with open(
        os.path.join(config.exp_dir, "metrics", "final_results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Result dir: {config.exp_dir}")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 80)
    return 0

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Macro-to-Micro Density-Aware MIL Network"
    )
    parser.add_argument("--img-root", type=str, default="")
    parser.add_argument("--excel-path", type=str, default="", help="Excel label file path; kept for backward compatibility.")
    parser.add_argument("--label-file", type=str, default="", help="CSV or Excel label file for GBNPC2026.")
    parser.add_argument("--id-col", type=str, default="case_id", help="Case-id column in the label file.")
    parser.add_argument("--label-col", type=str, default="label", help="Binary label column: 0=low, 1=high.")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--detector-ckpt", type=str, default="")
    parser.add_argument("--skip-detector-pretrain", action="store_true")
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--classifier-epochs", type=int, default=None)
    parser.add_argument("--num-rois", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument("--classifier-batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None, help="Training ratio used only with --single-split. Default: 0.70.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation ratio used only with --single-split. Default: 0.10.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio used only with --single-split. Default: 0.20.")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None, help="Number of outer CV folds. Default: 5.")
    parser.add_argument("--inner-val-ratio", type=float, default=None, help="Validation ratio inside each outer training fold. Default: 0.20.")
    parser.add_argument("--single-split", action="store_true", help="Run one stratified train/val/test split instead of five-fold CV.")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--freeze-detector-epochs", type=int, default=None)
    parser.add_argument("--no-detector-finetune", action="store_true")
    return parser


if __name__ == "__main__":
    start = time.time()
    try:
        exit_code = main()
        if exit_code == 0:
            print(f"Program completed successfully, total time: {time.time() - start:.2f}s")
        else:
            print(f"Program exited with code: {exit_code}")
    except Exception as exc:
        print(f"\nProgram failed: {exc}")
        traceback.print_exc()
