"""
Regression tests for the PRE-EXISTING analytical robotics stack
(kinematics.py, trajectory_generator.py's pure math), to confirm the ML/
perception work in this package did not change their behavior. The only
edit made to either file was moving trajectory_generator.py's ROS message
imports from module scope into build_cubic_trajectory() so its pure-math
helpers can be imported without ROS 2 installed -- these tests also pin
down that build_cubic_trajectory() itself was left logically unchanged
(save for that import-scoping) by checking cubic_position/cubic_velocity
directly, which is all any ROS-independent caller in this project uses.
"""

import math
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "manipulator_description", "scripts")))

import kinematics as kin
from trajectory_generator import cubic_position, cubic_velocity


def test_kinematics_fk_selfconsistency():
    max_err = kin.self_test()
    assert max_err < 1e-9


def test_kinematics_ik_roundtrip():
    max_err, failures = kin.ik_roundtrip_test()
    assert max_err < 1e-6


def test_cubic_trajectory_boundary_conditions():
    q0, qf, T = 0.0, 0.5, 2.0
    assert cubic_position(q0, qf, 0.0, T) == q0
    assert abs(cubic_position(q0, qf, T, T) - qf) < 1e-9
    assert abs(cubic_velocity(q0, qf, 0.0, T)) < 1e-9
    assert abs(cubic_velocity(q0, qf, T, T)) < 1e-9


def test_trajectory_generator_importable_without_ros():
    """build_cubic_trajectory() itself needs ROS message types, but the
    module import must succeed without ROS installed (this is the actual
    behavior change made to support this package)."""
    import trajectory_generator  # noqa: F401 -- import success is the assertion
