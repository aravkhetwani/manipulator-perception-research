import math

import numpy as np

from perception_research import sim_camera as sc
from perception_research import integration as ig


def test_zero_error_pick_always_succeeds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = sc.sample_reachable_object_xy(rng)
        result = ig.run_pick_trial((x, y), (x, y))
        assert result["reachable"]
        assert result["trajectory_ok"]
        assert result["grasp_proxy_ok"]
        assert result["position_error_m"] == 0.0


def test_large_error_pick_fails_grasp_proxy():
    rng = np.random.default_rng(1)
    x, y = sc.sample_reachable_object_xy(rng)
    # 10 cm off is far more than GRASP_TOLERANCE_M
    result = ig.run_pick_trial((x + 0.10, y + 0.10), (x, y))
    assert not result["grasp_proxy_ok"]


def test_unreachable_target_reports_not_reachable():
    # Far outside the workspace annulus entirely.
    result = ig.run_pick_trial((5.0, 5.0), (5.0, 5.0))
    assert not result["reachable"]
    assert not result["trajectory_ok"]
    assert not result["grasp_proxy_ok"]
    assert result["ik_solution_rad"] is None


def test_trajectory_feasibility_boundary_conditions():
    ok = ig.check_trajectory_feasibility((0.0, 0.0, 0.0), (0.5, -0.3, 0.2))
    assert ok


def test_grasp_tolerance_threshold_behavior():
    rng = np.random.default_rng(2)
    x, y = sc.sample_reachable_object_xy(rng)
    just_inside = ig.run_pick_trial((x + ig.GRASP_TOLERANCE_M * 0.5, y), (x, y))
    just_outside = ig.run_pick_trial((x + ig.GRASP_TOLERANCE_M * 3.0, y), (x, y))
    assert just_inside["grasp_proxy_ok"]
    assert not just_outside["grasp_proxy_ok"]


def test_summarize_batch_rates_in_unit_interval():
    rng = np.random.default_rng(3)
    trials = []
    for _ in range(30):
        x, y = sc.sample_reachable_object_xy(rng)
        noisy = (x + rng.normal(0, 0.02), y + rng.normal(0, 0.02))
        trials.append(ig.run_pick_trial(noisy, (x, y)))
    summary = ig.summarize_batch(trials)
    for key in ("perception_ok_rate", "reachable_rate", "trajectory_ok_rate", "grasp_proxy_ok_rate"):
        assert 0.0 <= summary[key] <= 1.0
    assert summary["n"] == 30
