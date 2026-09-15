#!/usr/bin/env bash
# Reproduces the full perception/ML research pipeline end to end.
# Requires only: numpy, scikit-learn, matplotlib, pytest (see
# src/perception_research/requirements.txt). Does NOT require ROS 2 or
# Gazebo -- see docs/RESEARCH.md "Environment availability".
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1/5 Regression tests (existing robotics stack + new perception stack) =="
python -m pytest src/perception_research/tests -q

echo "== 2/5 Dataset generation (fixed seeds; train/val/test/test_generalization) =="
python -m perception_research.dataset --out-dir data/perception

echo "== 3/5 Training (classical baseline + MLP regressor) =="
python -m perception_research.train --data-dir data/perception --out-dir models

echo "== 4/5 Evaluation (localization, robustness, generalization, end-to-end manipulation, sensitivity) =="
python -m perception_research.evaluate --data-dir data/perception --model-dir models --out results/metrics.json

echo "== 5/5 Dashboard =="
python -m perception_research.report --data-dir data/perception --model-dir models --results results/metrics.json --out results/dashboard.html

echo
echo "Done. Open results/dashboard.html in a browser, and see results/metrics.json for raw numbers."
