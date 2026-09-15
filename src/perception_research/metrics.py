#!/usr/bin/env python3
"""Shared metric helpers for perception evaluation."""

import numpy as np

from perception_research import sim_camera as sc


def pixel_errors(pred_uv, gt_uv):
    diff = np.asarray(pred_uv) - np.asarray(gt_uv)
    return np.linalg.norm(diff, axis=1)  # euclidean pixel error per sample


def world_positions_from_pixels(pred_uv):
    """Back-project predicted pixels to world (x, y) using the NOMINAL camera
    calibration only (see sim_camera.backproject_nominal's docstring)."""
    pred_uv = np.asarray(pred_uv)
    out = np.empty_like(pred_uv, dtype=np.float64)
    for i, (u, v) in enumerate(pred_uv):
        out[i] = sc.backproject_nominal(u, v)
    return out


def world_errors(pred_uv, gt_world_xy):
    world_pred = world_positions_from_pixels(pred_uv)
    diff = world_pred - np.asarray(gt_world_xy)
    return np.linalg.norm(diff, axis=1), world_pred


def summarize(errors):
    errors = np.asarray(errors)
    return dict(
        mae=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        max=float(np.max(errors)),
        n=int(len(errors)),
    )


def localization_success_rate(pixel_err, threshold_px):
    return float(np.mean(np.asarray(pixel_err) <= threshold_px))
