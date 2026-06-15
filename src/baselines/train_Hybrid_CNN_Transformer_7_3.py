#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.Hybrid_CNN_Transformer import Hybrid_CNN_Transformer


MODEL_NAME = "Hybrid_CNN_Transformer"


def build_model(config: TrainConfig):
    # The source model records img_size as (W, H, Z); our config is (D, H, W).
    img_size = (config.input_size[2], config.input_size[1], config.input_size[0])
    return Hybrid_CNN_Transformer(
        in_channels=len(config.modalities),
        img_size=img_size,
        num_classes=1,
        embed_dim=256,
        trans_depth=4,
        num_heads=8,
    )


if __name__ == "__main__":
    run_training(MODEL_NAME, build_model)
