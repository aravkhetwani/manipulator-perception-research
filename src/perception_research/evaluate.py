#!/usr/bin/env python3
"""
Evaluation script producing every metric the research write-up reports:

  1. Analytical/ground-truth localization "baseline" (zero perception
     error -- i.e. what manipulation success looks like if perception were
     perfect; see ``ground_truth_manipulation_baseline``).
  2. Classical-CV and MLP localization accuracy on the held-out test split
     (pixel MAE/RMSE, world MAE/RMSE, localization success rate, inference
     time).
  3. Robustness experiment: accuracy of both models as a function of
     synthetic image-noise level (independent, freshly-rendered scenes at
     each noise level -- not the stored test split, so this is a genuine
     sweep, not a re-labelling of existing data).
  4. Generalization experiment: accuracy on ``test_generalization`` (the
     checker-background split never seen in train/val).
  5. End-to-end manipulation experiment: predicted-position pipeline
     (perception -> backproject -> IK -> trajectory -> grasp proxy) on the
     test split, for both models.
  6. Ground-truth-vs-predicted manipulation comparison: the same pipeline
     fed true positions instead of predictions, to isolate perception's
     contribution to manipulation failure.
  7. Sensitivity analysis: grasp-proxy success rate vs. injected position
     error magnitude (0 .. 5 cm), independent of any model, to show how
     localization error alone propagates into pick success.

All numbers are computed from the actual generated dataset/models -- none
are hardcoded. Results are written to ``results/metrics.json``.

Usage:
    python -m perception_research.evaluate --data-dir data/perception --model-dir models --out results/metrics.json
"""

import argparse
import json
import os
import time

import numpy as np

from perception_research import dataset as ds
from perception_research import sim_camera as sc
from perception_research import metrics as met
from perception_research import integration as ig
from perception_research.models import ClassicalCentroidBaseline, MLPCoordinateRegressor, timed_predict

LOCALIZATION_SUCCESS_PX = 5.0  # pixel-error threshold for "localized correctly"


def eval_model_on_split(model, split, model_name):
    images, gt_uv, gt_xy = split["images"], split["pixel_uv"], split["world_xy"]
    pred_uv, per_sample_ms = timed_predict(model, images)
    pix_err = met.pixel_errors(pred_uv, gt_uv)
    world_err, world_pred = met.world_errors(pred_uv, gt_xy)

    return dict(
        model=model_name,
        pixel_error=met.summarize(pix_err),
        world_error_m=met.summarize(world_err),
        localization_success_rate=met.localization_success_rate(pix_err, LOCALIZATION_SUCCESS_PX),
        inference_time_ms_per_sample=float(per_sample_ms),
        n=len(images),
    ), pred_uv, pix_err, world_pred


def ground_truth_manipulation_baseline(split, n=None):
    """Manipulation pipeline fed EXACT ground-truth object positions (zero
    perception error) -- the analytical upper bound the ML pipeline is
    compared against."""
    xy = split["world_xy"]
    if n is not None:
        xy = xy[:n]
    results = ig.run_batch(xy, xy, pixel_err_arr=np.zeros(len(xy)))
    return ig.summarize_batch(results)


def end_to_end_manipulation(pred_world_xy, gt_world_xy, pix_err):
    results = ig.run_batch(pred_world_xy, gt_world_xy, pixel_err_arr=pix_err)
    return ig.summarize_batch(results)


def robustness_sweep(model, model_name, noise_levels, n_per_level=200, seed=999):
    out = []
    for noise in noise_levels:
        rng = np.random.default_rng(seed + int(noise * 10000))
        images = np.empty((n_per_level, sc.IMG_SIZE, sc.IMG_SIZE), dtype=np.float32)
        gt_uv = np.empty((n_per_level, 2))
        gt_xy = np.empty((n_per_level, 2))
        for i in range(n_per_level):
            s = sc.render_scene(rng, noise_range=(noise, noise))
            images[i] = s.image
            gt_uv[i] = s.pixel_uv
            gt_xy[i] = s.world_xy
        pred_uv, ms = timed_predict(model, images)
        pix_err = met.pixel_errors(pred_uv, gt_uv)
        world_err, _ = met.world_errors(pred_uv, gt_xy)
        out.append(dict(
            noise_level=float(noise),
            model=model_name,
            pixel_mae=float(pix_err.mean()),
            pixel_rmse=float(np.sqrt((pix_err ** 2).mean())),
            world_mae_m=float(world_err.mean()),
            localization_success_rate=met.localization_success_rate(pix_err, LOCALIZATION_SUCCESS_PX),
            n=n_per_level,
        ))
    return out


def sensitivity_position_error(error_levels_m, n_per_level=300, seed=1234):
    """Model-independent: injects a KNOWN, controlled position error
    (Gaussian, given std in meters) directly onto ground-truth positions and
    measures grasp-proxy success -- isolates how localization error alone
    propagates into manipulation outcome, with no perception model in the
    loop at all."""
    out = []
    for err_std in error_levels_m:
        rng = np.random.default_rng(seed + int(err_std * 100000))
        gt_xy = np.array([sc.sample_reachable_object_xy(rng) for _ in range(n_per_level)])
        noise = rng.normal(0.0, err_std, size=gt_xy.shape)
        pred_xy = gt_xy + noise
        results = ig.run_batch(pred_xy, gt_xy)
        summary = ig.summarize_batch(results)
        summary["injected_error_std_m"] = float(err_std)
        out.append(summary)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/perception")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    test = ds.load_split(args.data_dir, "test")
    test_gen = ds.load_split(args.data_dir, "test_generalization")

    baseline = ClassicalCentroidBaseline.load(os.path.join(args.model_dir, "classical_baseline.pkl"))
    mlp = MLPCoordinateRegressor.load(os.path.join(args.model_dir, "mlp_regressor.pkl"))

    results = {}

    # 1. Analytical ground-truth manipulation baseline (perfect perception)
    results["ground_truth_manipulation_baseline"] = ground_truth_manipulation_baseline(test)

    # 2. Localization accuracy on held-out test split, both models
    test_eval = {}
    pred_cache = {}
    for name, model in (("classical_baseline", baseline), ("mlp_regressor", mlp)):
        summary, pred_uv, pix_err, world_pred = eval_model_on_split(model, test, name)
        test_eval[name] = summary
        pred_cache[name] = (pred_uv, pix_err, world_pred)
    results["test_localization"] = test_eval

    # 4. Generalization split (checker background, unseen)
    gen_eval = {}
    for name, model in (("classical_baseline", baseline), ("mlp_regressor", mlp)):
        summary, _, _, _ = eval_model_on_split(model, test_gen, name)
        gen_eval[name] = summary
    results["generalization_localization"] = gen_eval

    # 3. Robustness sweep vs. injected image noise
    noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    robustness = {}
    for name, model in (("classical_baseline", baseline), ("mlp_regressor", mlp)):
        robustness[name] = robustness_sweep(model, name, noise_levels)
    results["robustness_vs_noise"] = robustness

    # 5. End-to-end manipulation experiment using PREDICTED positions
    e2e = {}
    for name in ("classical_baseline", "mlp_regressor"):
        pred_uv, pix_err, world_pred = pred_cache[name]
        e2e[name] = end_to_end_manipulation(world_pred, test["world_xy"], pix_err)
    results["end_to_end_manipulation_predicted"] = e2e

    # 6. Ground-truth-vs-predicted comparison is baseline (1) vs (5) side by side
    results["manipulation_comparison"] = dict(
        ground_truth=results["ground_truth_manipulation_baseline"],
        predicted=e2e,
    )

    # 7. Sensitivity analysis: grasp success vs. injected position error, model-free
    error_levels_m = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    results["sensitivity_position_error"] = sensitivity_position_error(error_levels_m)

    results["meta"] = dict(
        localization_success_threshold_px=LOCALIZATION_SUCCESS_PX,
        grasp_tolerance_m=ig.GRASP_TOLERANCE_M,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")
    print(json.dumps({k: v for k, v in results.items() if k not in
                       ("robustness_vs_noise",)}, indent=2)[:4000])


if __name__ == "__main__":
    main()
