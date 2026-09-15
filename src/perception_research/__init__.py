"""
Perception/ML research extension for the 3-DOF R-R-R manipulator project.

This package is intentionally a PLAIN Python package (no ROS/ament
dependency, no rclpy import anywhere in it) so it can be developed, tested
and run on a machine that does not have ROS 2 / Gazebo installed. It reuses
the existing analytical kinematics from
``manipulator_description/scripts/kinematics.py`` (which is itself pure
NumPy/Python, see ``_kinematics_path()`` in ``integration.py``) rather than
re-deriving FK/IK.

See docs/RESEARCH.md at the repository root for the full write-up:
research question, simulated-camera model, dataset generation, ML
methodology, baselines, validation, robotics integration, experiments and
honest results/limitations.
"""
