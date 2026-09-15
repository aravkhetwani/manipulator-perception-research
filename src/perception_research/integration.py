#!/usr/bin/env python3
"""
Connects perception output to the existing, UNCHANGED analytical robotics
stack: predicted (x, y) -> reachability check -> analytic IK
(kinematics.inverse_kinematics) -> elbow-branch selection (handled inside
inverse_kinematics itself) -> smooth cubic joint-space trajectory
(trajectory_generator.cubic_position/cubic_velocity) -> a geometric
pick-success proxy.

This module intentionally does NOT reimplement FK/IK/trajectory math --
it imports the real modules from
``manipulator_description/scripts/`` (both are pure Python/NumPy with no
rclpy import at module scope, see the docstring notes in those files), so
"analytical IK" here is the exact same code path the ROS 2 node uses.

GRASP-SUCCESS HONESTY NOTE: there is no physics engine in this environment
(Gazebo is not installed here, and even where it is, this project's own
docs/CHANGELOG.md and pick_and_place.py document that gz-sim's
DetachableJoint does not physically couple motion to the grasped object in
the installed build, requiring a kinematic pose-follow workaround). This
module therefore does NOT claim to simulate physical grasping. It reports
four DISTINCT, separately-tracked outcomes for every trial:

  perception_ok   : the predicted pixel was within a plausible localization
                     tolerance of the ground-truth pixel (sanity signal only)
  reachable        : IK found a joint solution within joint limits for the
                     predicted (back-projected) world position
  trajectory_ok    : the cubic trajectory generated for that IK solution
                     satisfies its own boundary conditions (position/zero-
                     velocity at start and end) -- a smoothness/feasibility
                     check, not a physics simulation
  grasp_proxy_ok   : a GEOMETRIC proxy only -- the predicted world position
                     is within GRASP_TOLERANCE_M of the true object position,
                     i.e. close enough that a real gripper of the project's
                     documented size could plausibly still close around the
                     object. This is explicitly a kinematic/geometric stand-
                     in for "would this pick have worked", not a claim about
                     simulated contact dynamics.
"""

import math
import os
import sys

import numpy as np

_SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "manipulator_description", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import kinematics as kin  # noqa: E402  (existing, unchanged analytical FK/IK)
from trajectory_generator import cubic_position, cubic_velocity  # noqa: E402

from perception_research import sim_camera as sc

# Half of the object's footprint (see sim_camera.OBJECT_RADIUS_M) plus a
# small margin matching pick_and_place.py's documented gripper geometry
# (GRIPPER_CLOSED leaves ~0.024 m clear width around a 0.03 m object) --
# i.e. a target position error below this is one a real gripper of this
# project's documented size could still plausibly close around.
GRASP_TOLERANCE_M = 0.02

HOME_Q = (0.0, 0.0, 0.0)
TRAJ_DURATION_S = 2.5
TRAJ_WAYPOINTS = 20


def check_trajectory_feasibility(q0, qf, duration_s=TRAJ_DURATION_S, num_waypoints=TRAJ_WAYPOINTS,
                                  tol=1e-9):
    """
    Re-derives the same cubic trajectory pick_and_place.py would send to
    the controller and checks its boundary conditions hold (position
    continuity, zero start/end velocity) -- mirrors
    trajectory_generator.py's own __main__ self-check, but for arbitrary
    q0/qf without needing ROS message types.
    """
    ok = True
    for a, b in zip(q0, qf):
        pos0 = cubic_position(a, b, 0.0, duration_s)
        posT = cubic_position(a, b, duration_s, duration_s)
        vel0 = cubic_velocity(a, b, 0.0, duration_s)
        velT = cubic_velocity(a, b, duration_s, duration_s)
        if abs(pos0 - a) > tol or abs(posT - b) > 1e-6 or abs(vel0) > 1e-6 or abs(velT) > 1e-6:
            ok = False
    return ok


def run_pick_trial(predicted_world_xy, true_world_xy, pixel_err_px=None,
                    perception_tol_px=6.0):
    """
    Runs one predicted-object pick attempt through the real IK/trajectory
    pipeline and returns a dict of the four outcomes described in the
    module docstring, plus intermediate values for analysis.
    """
    px, py = predicted_world_xy
    tx, ty = true_world_xy
    z = sc.TABLE_Z

    perception_ok = (pixel_err_px is None) or (pixel_err_px <= perception_tol_px)

    sol = kin.inverse_kinematics(px, py, z)
    reachable = sol is not None

    trajectory_ok = False
    achieved_xyz = None
    if reachable:
        trajectory_ok = check_trajectory_feasibility(HOME_Q, sol)
        achieved_xyz = kin.forward_kinematics(*sol)

    position_error_m = math.dist((px, py), (tx, ty))
    grasp_proxy_ok = reachable and trajectory_ok and (position_error_m <= GRASP_TOLERANCE_M)

    return dict(
        predicted_world_xy=(px, py),
        true_world_xy=(tx, ty),
        position_error_m=position_error_m,
        perception_ok=bool(perception_ok),
        reachable=bool(reachable),
        ik_solution_rad=None if sol is None else tuple(sol),
        trajectory_ok=bool(trajectory_ok),
        achieved_xyz=achieved_xyz,
        grasp_proxy_ok=bool(grasp_proxy_ok),
    )


def run_batch(predicted_world_xy_arr, true_world_xy_arr, pixel_err_arr=None,
              perception_tol_px=6.0):
    n = len(predicted_world_xy_arr)
    results = []
    for i in range(n):
        pe = None if pixel_err_arr is None else float(pixel_err_arr[i])
        results.append(run_pick_trial(
            tuple(predicted_world_xy_arr[i]), tuple(true_world_xy_arr[i]),
            pixel_err_px=pe, perception_tol_px=perception_tol_px))
    return results


def summarize_batch(results):
    n = len(results)
    if n == 0:
        return dict(n=0)
    rates = {}
    for key in ("perception_ok", "reachable", "trajectory_ok", "grasp_proxy_ok"):
        rates[f"{key}_rate"] = float(np.mean([r[key] for r in results]))
    pos_errs = np.array([r["position_error_m"] for r in results])
    rates["position_error_mean_m"] = float(pos_errs.mean())
    rates["position_error_rmse_m"] = float(np.sqrt((pos_errs ** 2).mean()))
    rates["n"] = n
    return rates


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    trials = []
    for _ in range(20):
        x, y = sc.sample_reachable_object_xy(rng)
        noisy = (x + rng.normal(0, 0.01), y + rng.normal(0, 0.01))
        trials.append(run_pick_trial(noisy, (x, y)))
    print(summarize_batch(trials))
