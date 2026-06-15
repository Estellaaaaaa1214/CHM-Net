#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.X3D_Efficient import X3D_Classifier


MODEL_NAME = "X3D_Efficient"


def build_model(config: TrainConfig):
    return X3D_Classifier(in_channels=len(config.modalities), num_classes=1)


if __name__ == "__main__":
    run_training(MODEL_NAME, build_model)
