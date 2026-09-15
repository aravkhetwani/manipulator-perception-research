#!/usr/bin/env python3
"""
Trains the learned MLP coordinate regressor on the train split and fits the
(parameter-free) classical baseline for reference, saving both models plus
training-time metadata as JSON.

Usage:
    python -m perception_research.train --data-dir data/perception --out-dir models
"""

import argparse
import json
import os

import numpy as np

from perception_research import dataset as ds
from perception_research.models import ClassicalCentroidBaseline, MLPCoordinateRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/perception")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = ds.load_split(args.data_dir, "train")
    val = ds.load_split(args.data_dir, "val")

    baseline = ClassicalCentroidBaseline().fit(train["images"], train["pixel_uv"])
    baseline.save(os.path.join(args.out_dir, "classical_baseline.pkl"))

    mlp = MLPCoordinateRegressor(seed=args.seed)
    mlp.fit(train["images"], train["pixel_uv"])
    mlp.save(os.path.join(args.out_dir, "mlp_regressor.pkl"))

    val_pred = mlp.predict(val["images"])
    val_err = np.linalg.norm(val_pred - val["pixel_uv"], axis=1)

    meta = dict(
        n_train=len(train["images"]),
        n_val=len(val["images"]),
        mlp_hidden_layers=list(mlp.model.hidden_layer_sizes),
        mlp_n_iter=mlp.n_iter_,
        mlp_train_time_s=mlp.train_time_s,
        mlp_val_pixel_mae=float(val_err.mean()),
        mlp_val_pixel_rmse=float(np.sqrt((val_err ** 2).mean())),
        seed=args.seed,
    )
    with open(os.path.join(args.out_dir, "train_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"Models saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
