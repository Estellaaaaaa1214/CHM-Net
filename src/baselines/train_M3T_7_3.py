#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.M3T import M3T


MODEL_NAME = "M3T"


def build_model(config: TrainConfig):
    img_size = (config.input_size[2], config.input_size[1], config.input_size[0])
    return M3T(
        in_channels=len(config.modalities),
        img_size=img_size,
        num_classes=1,
        embed_dim=256,
        depth=4,
        num_heads=8,
    )


if __name__ == "__main__":
    default = TrainConfig(batch_size=1)
    run_training(MODEL_NAME, build_model, default)
