import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv3d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.projection(x)
        shape = x.shape[2:]
        return x.flatten(2).transpose(1, 2), shape


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        normalized = self.norm1(x)
        x = x + self.dropout(self.attention(normalized, normalized, normalized, need_weights=False)[0])
        return x + self.dropout(self.mlp(self.norm2(x)))


class MacroEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, heads, depth, mlp_ratio, dropout, max_tokens, projection_hidden_1, projection_hidden_2, projection_dim, projection_dropout_1, projection_dropout_2):
        super().__init__()
        self.embedding = PatchEmbedding3D(in_channels, embed_dim, patch_size)
        self.position = nn.Parameter(torch.empty(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.heatmap = nn.Sequential(nn.Conv3d(embed_dim, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True), nn.Conv3d(256, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True), nn.Conv3d(128, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.Conv3d(64, 1, 1), nn.Sigmoid())
        self.projection = nn.Sequential(nn.Linear(embed_dim, projection_hidden_1), nn.BatchNorm1d(projection_hidden_1), nn.ReLU(inplace=True), nn.Dropout(projection_dropout_1), nn.Linear(projection_hidden_1, projection_hidden_2), nn.BatchNorm1d(projection_hidden_2), nn.ReLU(inplace=True), nn.Dropout(projection_dropout_2), nn.Linear(projection_hidden_2, projection_dim))

    def forward(self, x):
        batch, _, depth, height, width = x.shape
        tokens, shape = self.embedding(x)
        tokens = tokens + self.position[:, :tokens.size(1)]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        features = tokens.transpose(1, 2).reshape(batch, -1, *shape)
        heatmap = F.interpolate(self.heatmap(features), (depth, height, width), mode="trilinear", align_corners=False)
        macro_features = tokens.mean(1)
        return heatmap, macro_features, self.projection(macro_features)


class TopKROIs(nn.Module):
    def __init__(self, number, size, suppression_radius):
        super().__init__()
        self.number = number
        self.size = size
        self.radius = suppression_radius

    def forward(self, volumes, heatmaps):
        outputs, maps, centers, scores = [], [], [], []
        _, _, depth, height, width = volumes.shape
        pooled = F.avg_pool3d(heatmaps, 3, 1, 1).detach()
        for volume, heatmap, score_map in zip(volumes, heatmaps, pooled):
            score_map = score_map[0].clone()
            sample_rois, sample_maps, sample_centers, sample_scores = [], [], [], []
            for _ in range(self.number):
                index = score_map.argmax()
                z, y, x = torch.unravel_index(index, score_map.shape)
                center = (int(z), int(y), int(x))
                sample_rois.append(self._crop(volume, center))
                sample_maps.append(self._crop(heatmap, center))
                sample_centers.append(torch.tensor(center, device=volumes.device))
                sample_scores.append(heatmap[0, center[0], center[1], center[2]])
                z1, z2 = max(0, center[0]-self.radius), min(depth, center[0]+self.radius+1)
                y1, y2 = max(0, center[1]-self.radius), min(height, center[1]+self.radius+1)
                x1, x2 = max(0, center[2]-self.radius), min(width, center[2]+self.radius+1)
                score_map[z1:z2, y1:y2, x1:x2] = -1
            outputs.append(torch.stack(sample_rois))
            maps.append(torch.stack(sample_maps))
            centers.append(torch.stack(sample_centers))
            scores.append(torch.stack(sample_scores))
        return torch.stack(outputs), torch.stack(maps), torch.stack(centers), torch.stack(scores)

    def _crop(self, volume, center):
        half = self.size // 2
        shape = volume.shape[1:]
        center = [min(max(v, half), shape[i] - (self.size - half)) for i, v in enumerate(center)]
        slices = tuple(slice(v-half, v-half+self.size) for v in center)
        return volume[(slice(None),) + slices]


class TriPlanarEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, input_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(128, 192, 3, padding=1, bias=False), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(192, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
        self.view_attention = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.Tanh(), nn.Linear(embed_dim // 2, 1))

    def forward(self, rois, roi_heatmaps, gate_mode, gate_alpha):
        batch, number, channels, depth, height, width = rois.shape
        count = batch * number
        rois = rois.reshape(count, channels, depth, height, width)
        roi_heatmaps = roi_heatmaps.reshape(count, 1, depth, height, width)
        gated = rois * roi_heatmaps if gate_mode == "multiply" else rois * (1 + gate_alpha * roi_heatmaps)
        axial = self._project(gated, roi_heatmaps, 2)
        coronal = self._project(gated, roi_heatmaps, 3)
        sagittal = self._project(gated, roi_heatmaps, 4)
        views = torch.stack((axial, coronal, sagittal), 1).reshape(count * 3, channels, height, width)
        views = F.interpolate(views, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
        views = self._normalize(views)
        features = self.projection(self.encoder(views)).reshape(count, 3, -1)
        weights = self.view_attention(features).squeeze(-1).softmax(1)
        features = (features * weights.unsqueeze(-1)).sum(1).reshape(batch, number, -1)
        return features, weights.reshape(batch, number, 3)

    @staticmethod
    def _project(roi, heatmap, axis):
        return (roi * heatmap).sum(axis) / heatmap.sum(axis).clamp_min(1e-6)

    @staticmethod
    def _normalize(x):
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std


class MacroMicroMILHead(nn.Module):
    def __init__(self, macro_dim, micro_dim, classes, dropout, score_bias):
        super().__init__()
        self.macro_projection = nn.Sequential(nn.Linear(macro_dim, micro_dim), nn.LayerNorm(micro_dim), nn.ReLU(inplace=True), nn.Dropout(dropout / 2))
        self.attention = nn.Sequential(nn.Linear(micro_dim, micro_dim // 2), nn.Tanh(), nn.Linear(micro_dim // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(micro_dim * 2, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout / 2), nn.Linear(128, classes))
        self.score_bias = score_bias

    def forward(self, macro_features, roi_features, roi_scores):
        weights = self.attention(roi_features).squeeze(-1)
        bias = (roi_scores - roi_scores.mean(1, keepdim=True)) / roi_scores.std(1, keepdim=True).clamp_min(1e-6)
        weights = (weights + self.score_bias * bias).softmax(1)
        micro_features = (roi_features * weights.unsqueeze(-1)).sum(1)
        fused = torch.cat((self.macro_projection(macro_features), micro_features), 1)
        return self.classifier(fused), weights


class MacroToMicroDensityNet(nn.Module):
    def __init__(self, in_channels, macro_dim, patch_size, heads, depth, mlp_ratio, macro_dropout, max_tokens, projection_hidden_1, projection_hidden_2, projection_dim, projection_dropout_1, projection_dropout_2, roi_number, roi_size, suppression_radius, micro_dim, input_size, classes, micro_encoder_dropout, mil_dropout, gate_mode, gate_alpha, score_bias):
        super().__init__()
        self.macro = MacroEncoder(in_channels, macro_dim, patch_size, heads, depth, mlp_ratio, macro_dropout, max_tokens, projection_hidden_1, projection_hidden_2, projection_dim, projection_dropout_1, projection_dropout_2)
        self.rois = TopKROIs(roi_number, roi_size, suppression_radius)
        self.micro = TriPlanarEncoder(in_channels, micro_dim, input_size, micro_encoder_dropout)
        self.mil = MacroMicroMILHead(macro_dim, micro_dim, classes, mil_dropout, score_bias)
        self.gate_mode = gate_mode
        self.gate_alpha = gate_alpha

    def forward(self, x):
        heatmap, macro_features, projected_features = self.macro(x)
        rois, roi_maps, centers, scores = self.rois(x, heatmap)
        roi_features, view_weights = self.micro(rois, roi_maps, self.gate_mode, self.gate_alpha)
        logits, mil_weights = self.mil(macro_features, roi_features, scores)
        probabilities = logits.softmax(1)
        return {"logits": logits, "probabilities": probabilities, "high_density_confidence": probabilities[:, 1], "predictions": probabilities.argmax(1), "heatmap": heatmap, "macro_features": macro_features, "projected_features": projected_features, "roi_features": roi_features, "roi_centers": centers, "roi_scores": scores, "mil_weights": mil_weights, "view_weights": view_weights}
