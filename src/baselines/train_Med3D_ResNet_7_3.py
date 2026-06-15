#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.Med3D_ResNet import generate_model


MODEL_NAME = "Med3D_ResNet"


def build_model(config: TrainConfig):
    return generate_model(model_depth=18, in_channels=len(config.modalities), num_classes=1)


if __name__ == "__main__":
    run_training(MODEL_NAME, build_model)
