#!/usr/bin/env python3
from common_3d_train_7_3 import TrainConfig, run_training
from model_defs.Vit import Vit


MODEL_NAME = "ViT"


def build_model(config: TrainConfig):
    d, h, w = config.input_size
    embedding_dim = max(1, (d // 32) * (h // 32) * (w // 32))
    return Vit(
        in_channels=len(config.modalities),
        out_channels=1,
        embed_dim=96,
        embedding_dim=embedding_dim,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    )


if __name__ == "__main__":
    default = TrainConfig(batch_size=1)
    run_training(MODEL_NAME, build_model, default)
