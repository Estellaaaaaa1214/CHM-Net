#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.Swin3D_Classifier import Swin3D_Classifier


MODEL_NAME = "Swin3D_Classifier"


def build_model(config: TrainConfig):
    return Swin3D_Classifier(
        in_channels=len(config.modalities),
        img_size=config.input_size,
        num_classes=1,
    )


if __name__ == "__main__":
    # Swin3D is comparatively heavy, so batch_size=1 is safer by default.
    default = TrainConfig(batch_size=1)
    run_training(MODEL_NAME, build_model, default)
