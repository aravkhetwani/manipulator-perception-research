#!/usr/bin/env python3
"""
Reproducible dataset generation for the vision-based object-localization
task (see sim_camera.py for the renderer and its honesty note about it
being a NumPy pinhole-camera mock, not Gazebo).

Splits produced (fixed seeds, non-overlapping by construction since each
split draws from its own seeded RNG stream and the underlying sampling
space is continuous, so accidental duplication across splits has
probability 0):

  train              6000 samples, backgrounds in {flat, gradient}
  val                1200 samples, backgrounds in {flat, gradient}
  test               1200 samples, backgrounds in {flat, gradient}
  test_generalization 800 samples, background ALWAYS "checker"
                       (never seen during training/val/model selection) --
                       this is the distribution-shift / generalization split.

Usage:
    python -m perception_research.dataset --out-dir data/perception
"""

import argparse
import os
import time

import numpy as np

from perception_research import sim_camera as sc

SPLIT_SEEDS = {
    "train": 100,
    "val": 200,
    "test": 300,
    "test_generalization": 400,
}
SPLIT_SIZES = {
    "train": 6000,
    "val": 1200,
    "test": 1200,
    "test_generalization": 800,
}
# In-distribution splits only ever see flat/gradient backgrounds; the
# generalization split is checker-ONLY, an unseen condition.
IN_DIST_BACKGROUNDS = ("flat", "gradient")
OOD_BACKGROUNDS = ("checker",)


def generate_split(name, n, seed):
    rng = np.random.default_rng(seed)
    bg_types = OOD_BACKGROUNDS if name == "test_generalization" else IN_DIST_BACKGROUNDS

    images = np.empty((n, sc.IMG_SIZE, sc.IMG_SIZE), dtype=np.float32)
    pixel_uv = np.empty((n, 2), dtype=np.float64)
    world_xy = np.empty((n, 2), dtype=np.float64)
    orientation = np.empty(n, dtype=np.float64)
    size_scale = np.empty(n, dtype=np.float64)
    brightness = np.empty(n, dtype=np.float64)
    noise_level = np.empty(n, dtype=np.float64)
    bg_type = np.empty(n, dtype=object)
    cam_jitter_deg = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = sc.render_scene(rng, bg_types=bg_types)
        images[i] = s.image
        pixel_uv[i] = s.pixel_uv
        world_xy[i] = s.world_xy
        orientation[i] = s.orientation_rad
        size_scale[i] = s.size_scale
        brightness[i] = s.brightness
        noise_level[i] = s.noise_level
        bg_type[i] = s.bg_type
        cam_jitter_deg[i] = s.cam_jitter_deg

    return dict(images=images, pixel_uv=pixel_uv, world_xy=world_xy,
                orientation=orientation, size_scale=size_scale,
                brightness=brightness, noise_level=noise_level,
                bg_type=bg_type.astype(str), cam_jitter_deg=cam_jitter_deg)


def save_split(out_dir, name, data):
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, **data)
    return path


def load_split(out_dir, name):
    path = os.path.join(out_dir, f"{name}.npz")
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/perception")
    parser.add_argument("--splits", nargs="*", default=list(SPLIT_SEEDS.keys()))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for name in args.splits:
        t0 = time.time()
        data = generate_split(name, SPLIT_SIZES[name], SPLIT_SEEDS[name])
        path = save_split(args.out_dir, name, data)
        dt = time.time() - t0
        manifest[name] = dict(n=SPLIT_SIZES[name], seed=SPLIT_SEEDS[name], path=path,
                               backgrounds=list(OOD_BACKGROUNDS if name == "test_generalization"
                                                 else IN_DIST_BACKGROUNDS))
        print(f"[{name}] {SPLIT_SIZES[name]} samples -> {path} ({dt:.1f}s)")

    import json
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
