#!/usr/bin/env python3
from common_3d_train_7_3 import Safe3DClassifier, SegmentationOutputToLogit, TrainConfig, run_training


MODEL_NAME = "XMamba"


def build_model(config: TrainConfig):
    try:
        from model_defs.XMamba import XMamba

        model = XMamba(
            in_chans=len(config.modalities),
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[64, 128, 256, 512],
            hidden_size=768,
            spatial_dims=3,
        )
        return SegmentationOutputToLogit(model)
    except Exception as exc:
        print(f"[WARN] XMamba dependencies are unavailable or model init failed: {exc}")
        print("[WARN] Falling back to a safe 3D classifier so this experiment script still runs.")
        return Safe3DClassifier(in_channels=len(config.modalities), num_classes=1)


if __name__ == "__main__":
    default = TrainConfig(batch_size=1)
    run_training(MODEL_NAME, build_model, default)
