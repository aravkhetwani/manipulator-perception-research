#!/usr/bin/env python3
"""
Builds a static HTML research dashboard from the REAL, previously-computed
results in results/metrics.json (produced by evaluate.py) and the stored
dataset (for sample images). Every number and every plot is generated from
those files -- nothing in this script is a hardcoded metric.

Usage:
    python -m perception_research.report --data-dir data/perception --results results/metrics.json --out results/dashboard.html
"""

import argparse
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perception_research import dataset as ds
from perception_research import sim_camera as sc


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_sample_images(data_dir, n=8):
    test = ds.load_split(data_dir, "test")
    idx = np.random.default_rng(0).choice(len(test["images"]), size=n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.2))
    for ax, i in zip(axes, idx):
        ax.imshow(test["images"][i], cmap="gray", vmin=0, vmax=1)
        u, v = test["pixel_uv"][i]
        ax.scatter([u], [v], c="red", s=25, marker="+")
        ax.set_title(f"{test['bg_type'][i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample test scenes (red + = ground-truth pixel centroid)")
    return _fig_to_base64(fig)


def plot_predicted_vs_gt(data_dir, model_dir):
    from perception_research.models import ClassicalCentroidBaseline, MLPCoordinateRegressor
    test = ds.load_split(data_dir, "test")
    baseline = ClassicalCentroidBaseline.load(os.path.join(model_dir, "classical_baseline.pkl"))
    mlp = MLPCoordinateRegressor.load(os.path.join(model_dir, "mlp_regressor.pkl"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, model) in zip(axes, (("Classical baseline", baseline), ("MLP regressor", mlp))):
        pred = model.predict(test["images"])
        ax.scatter(test["pixel_uv"][:, 0], test["pixel_uv"][:, 1], s=4, alpha=0.4, label="ground truth")
        ax.scatter(pred[:, 0], pred[:, 1], s=4, alpha=0.4, label="predicted")
        ax.set_title(name)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs. ground-truth pixel centroid (test split)")
    return _fig_to_base64(fig)


def plot_error_distribution(metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Only summary stats are stored, not raw per-sample errors, so plot the
    # summary distribution shape (mean/median/p90/max) as a bar comparison.
    names, means, medians, p90s = [], [], [], []
    for name, m in metrics["test_localization"].items():
        names.append(name)
        means.append(m["pixel_error"]["mae"])
        medians.append(m["pixel_error"]["median"])
        p90s.append(m["pixel_error"]["p90"])
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, means, width, label="MAE")
    ax.bar(x, medians, width, label="median")
    ax.bar(x + width, p90s, width, label="p90")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("pixel error (px)")
    ax.set_title("Test-split localization error distribution")
    ax.legend()
    return _fig_to_base64(fig)


def plot_robustness(metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in metrics["robustness_vs_noise"].items():
        noise = [r["noise_level"] for r in rows]
        mae = [r["pixel_mae"] for r in rows]
        ax.plot(noise, mae, marker="o", label=name)
    ax.set_xlabel("injected image noise std")
    ax.set_ylabel("pixel MAE (px)")
    ax.set_title("Robustness: localization error vs. image noise")
    ax.legend()
    return _fig_to_base64(fig)


def plot_sensitivity(metrics):
    rows = metrics["sensitivity_position_error"]
    fig, ax = plt.subplots(figsize=(6, 4))
    err = [r["injected_error_std_m"] * 100 for r in rows]
    grasp = [r["grasp_proxy_ok_rate"] for r in rows]
    reach = [r["reachable_rate"] for r in rows]
    ax.plot(err, grasp, marker="o", label="grasp-proxy success rate")
    ax.plot(err, reach, marker="s", label="reachable (IK success) rate")
    ax.axvline(ig_tol_cm(metrics), color="gray", linestyle="--", linewidth=1,
               label="grasp tolerance")
    ax.set_xlabel("injected position error std (cm)")
    ax.set_ylabel("rate")
    ax.set_title("Sensitivity: manipulation success vs. localization error")
    ax.legend(fontsize=8)
    return _fig_to_base64(fig)


def ig_tol_cm(metrics):
    return metrics["meta"]["grasp_tolerance_m"] * 100


def plot_workspace_reachability():
    from perception_research import sim_camera as sc
    rng = np.random.default_rng(0)
    pts = np.array([sc.sample_reachable_object_xy(rng) for _ in range(1500)])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5)
    ax.scatter([0], [0], c="black", marker="^", s=80, label="base axis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Sampled reachable object workspace (rejection-sampled vs. real IK)")
    ax.legend()
    return _fig_to_base64(fig)


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Perception + Manipulation Research Dashboard</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; background: #fafafa; color: #1a1a1a; }}
h1 {{ margin-bottom: 0.2rem; }}
.subtitle {{ color: #666; margin-top: 0; }}
.section {{ background: white; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
th, td {{ text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #eee; font-size: 0.92rem; }}
th {{ background: #f2f2f2; }}
.metric-flag {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:600; }}
.ok {{ background:#e3f7e9; color:#1a7f37; }}
.warn {{ background:#fff3e0; color:#9a5b00; }}
img {{ max-width: 100%; border-radius: 6px; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.note {{ font-size: 0.85rem; color: #555; background:#fff8e1; border-left:3px solid #f0ad4e; padding:0.6rem 0.9rem; margin-top:0.6rem;}}
code {{ background:#f2f2f2; padding:0.1rem 0.35rem; border-radius:3px; }}
</style></head>
<body>
<h1>3-DOF RRR Manipulator -- Perception/ML Research Dashboard</h1>
<p class="subtitle">Generated entirely from results/metrics.json and the stored dataset -- no hardcoded numbers. Run <code>python -m perception_research.evaluate</code> then this script to regenerate.</p>

<div class="section">
<h2>1. Sample scenes</h2>
<img src="data:image/png;base64,{sample_images}">
</div>

<div class="section">
<h2>2. Ground-truth manipulation baseline (perfect perception)</h2>
<p>Upper bound on manipulation success if perception were exact -- isolates the analytical robotics stack's own reachability/IK/trajectory success rate on the test-split object positions.</p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{gt_baseline_rows}
</table>
</div>

<div class="section">
<h2>3. Localization accuracy (test split)</h2>
<table>
<tr><th>Model</th><th>Pixel MAE</th><th>Pixel RMSE</th><th>World MAE (m)</th><th>World RMSE (m)</th>
<th>Localization success rate (&le;{loc_thresh}px)</th><th>Inference (ms/sample)</th></tr>
{loc_rows}
</table>
<div class="grid">
<img src="data:image/png;base64,{pred_vs_gt}">
<img src="data:image/png;base64,{error_dist}">
</div>
</div>

<div class="section">
<h2>4. Generalization (checker background, never seen in training)</h2>
<table>
<tr><th>Model</th><th>Pixel MAE</th><th>World MAE (m)</th><th>Localization success rate</th></tr>
{gen_rows}
</table>
<p class="note">test_generalization only contains the "checker" background type, which train/val never see (train/val are flat/gradient only). A large jump here vs. Section 3 indicates overfitting to background appearance rather than learning object shape/position.</p>
</div>

<div class="section">
<h2>5. Robustness to image noise</h2>
<img src="data:image/png;base64,{robustness}">
</div>

<div class="section">
<h2>6. End-to-end manipulation: predicted vs. ground truth</h2>
<table>
<tr><th>Pipeline</th><th>Reachable rate</th><th>Trajectory-OK rate</th><th>Grasp-proxy success rate</th><th>Mean position error (m)</th></tr>
{e2e_rows}
</table>
<p class="note">"Grasp-proxy success" is a GEOMETRIC criterion (predicted position within {grasp_tol_cm:.1f} cm of the true object position, with a valid IK solution and feasible trajectory) -- not a physics simulation. This project's own Gazebo-based grasp mechanism has a documented limitation (DetachableJoint does not physically couple arm motion to the held object in the installed gz-sim build; see the repository's existing docs/CHANGELOG.md and pick_and_place.py docstring), so no experiment here claims physically-simulated grasp dynamics.</p>
</div>

<div class="section">
<h2>7. Sensitivity: manipulation success vs. localization error magnitude</h2>
<img src="data:image/png;base64,{sensitivity}">
</div>

<div class="section">
<h2>8. Robot workspace / reachability</h2>
<img src="data:image/png;base64,{workspace}">
</div>

<div class="section">
<p class="subtitle">Generated at {generated_at}</p>
</div>
</body></html>
"""


def _fmt_rate(x):
    flag = "ok" if x >= 0.8 else "warn"
    return f'<span class="metric-flag {flag}">{x*100:.1f}%</span>'


def build_dashboard(data_dir, model_dir, metrics_path, out_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

    sample_images = plot_sample_images(data_dir)
    pred_vs_gt = plot_predicted_vs_gt(data_dir, model_dir)
    error_dist = plot_error_distribution(metrics)
    robustness = plot_robustness(metrics)
    sensitivity = plot_sensitivity(metrics)
    workspace = plot_workspace_reachability()

    gtb = metrics["ground_truth_manipulation_baseline"]
    gt_baseline_rows = "\n".join(
        f"<tr><td>{k}</td><td>{_fmt_rate(v) if 'rate' in k else round(v, 5)}</td></tr>"
        for k, v in gtb.items() if k != "n")
    gt_baseline_rows += f"<tr><td>n samples</td><td>{gtb['n']}</td></tr>"

    loc_rows = ""
    for name, m in metrics["test_localization"].items():
        loc_rows += (f"<tr><td>{name}</td><td>{m['pixel_error']['mae']:.3f}</td>"
                     f"<td>{m['pixel_error']['rmse']:.3f}</td>"
                     f"<td>{m['world_error_m']['mae']:.4f}</td>"
                     f"<td>{m['world_error_m']['rmse']:.4f}</td>"
                     f"<td>{_fmt_rate(m['localization_success_rate'])}</td>"
                     f"<td>{m['inference_time_ms_per_sample']:.4f}</td></tr>\n")

    gen_rows = ""
    for name, m in metrics["generalization_localization"].items():
        gen_rows += (f"<tr><td>{name}</td><td>{m['pixel_error']['mae']:.3f}</td>"
                     f"<td>{m['world_error_m']['mae']:.4f}</td>"
                     f"<td>{_fmt_rate(m['localization_success_rate'])}</td></tr>\n")

    e2e_rows = f"<tr><td>Ground truth (perfect perception)</td>" + "".join(
        f"<td>{_fmt_rate(gtb[k]) if 'rate' in k else round(gtb[k], 5)}</td>"
        for k in ("reachable_rate", "trajectory_ok_rate", "grasp_proxy_ok_rate", "position_error_mean_m")
    ) + "</tr>\n"
    for name, m in metrics["end_to_end_manipulation_predicted"].items():
        e2e_rows += f"<tr><td>Predicted -- {name}</td>" + "".join(
            f"<td>{_fmt_rate(m[k]) if 'rate' in k else round(m[k], 5)}</td>"
            for k in ("reachable_rate", "trajectory_ok_rate", "grasp_proxy_ok_rate", "position_error_mean_m")
        ) + "</tr>\n"

    html = HTML_TEMPLATE.format(
        sample_images=sample_images,
        gt_baseline_rows=gt_baseline_rows,
        loc_thresh=metrics["meta"]["localization_success_threshold_px"],
        loc_rows=loc_rows,
        pred_vs_gt=pred_vs_gt,
        error_dist=error_dist,
        gen_rows=gen_rows,
        robustness=robustness,
        e2e_rows=e2e_rows,
        grasp_tol_cm=metrics["meta"]["grasp_tolerance_m"] * 100,
        sensitivity=sensitivity,
        workspace=workspace,
        generated_at=metrics["meta"]["generated_at"],
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/perception")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--results", default="results/metrics.json")
    parser.add_argument("--out", default="results/dashboard.html")
    args = parser.parse_args()
    path = build_dashboard(args.data_dir, args.model_dir, args.results, args.out)
    print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()
