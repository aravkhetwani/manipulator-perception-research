# Reproduces the full perception/ML research pipeline end to end (PowerShell).
# Requires only numpy/scikit-learn/matplotlib/pytest -- no ROS 2 / Gazebo.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"

Write-Host "== 1/5 Regression tests =="
python -m pytest src/perception_research/tests -q

Write-Host "== 2/5 Dataset generation =="
python -m perception_research.dataset --out-dir data/perception

Write-Host "== 3/5 Training =="
python -m perception_research.train --data-dir data/perception --out-dir models

Write-Host "== 4/5 Evaluation =="
python -m perception_research.evaluate --data-dir data/perception --model-dir models --out results/metrics.json

Write-Host "== 5/5 Dashboard =="
python -m perception_research.report --data-dir data/perception --model-dir models --results results/metrics.json --out results/dashboard.html

Write-Host ""
Write-Host "Done. Open results/dashboard.html, and see results/metrics.json for raw numbers."
