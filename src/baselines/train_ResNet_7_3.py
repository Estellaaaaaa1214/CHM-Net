#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.ResNet import resnet18


MODEL_NAME = "ResNet"


def build_model(config: TrainConfig):
    return resnet18(
        in_classes=len(config.modalities),
        num_classes=1,
        shortcut_type="B",
        spatial_size=config.input_size[1],
        sample_count=config.input_size[0],
    )


if __name__ == "__main__":
    run_training(MODEL_NAME, build_model)
