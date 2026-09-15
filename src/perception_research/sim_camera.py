#!/usr/bin/env python3
"""
Lightweight, deterministic synthetic "camera" used to generate the
vision-based localization dataset.

HONESTY NOTE (read this first): this is NOT Gazebo, and it is NOT a photo-
realistic renderer. Gazebo Sim / gz-sim is not installed in this
development environment (see docs/RESEARCH.md, "Environment availability"),
so this module implements a small but *geometrically real* pinhole-camera
simulator directly in NumPy: real 3-D-to-2-D perspective projection, a
real (if simplified) object-appearance model, and real, controllable image
noise/lighting/background perturbations. It plays the same role Gazebo's
camera sensor + a scene randomizer would play (produce an image plus
automatically-known ground truth), but it is a mock, and every place in
this project that uses it says so.

Scene setup
-----------
A single top-down camera is mounted above the robot's base axis, looking
down at the table plane the object rests on (z = TABLE_Z, matching
``pick_and_place.OBJECT_POS``'s height). The object is a small rectangular
block at a random position within the arm's reachable annulus (derived
directly from ``kinematics.py``'s link lengths, see
``sample_reachable_object_xy``) and a random yaw.

Camera viewpoint variation is modeled as small, random rigid perturbations
(rotation + translation) of the camera pose around a NOMINAL pose. The
image is rendered with the perturbed ("actual") pose -- exactly as a real
camera bolted slightly off from its calibrated mount would produce. Pixel
ground truth is the true projected pixel location under that perturbed
pose. Back-projection to world coordinates (in ``integration.py``), like a
real system, only has access to the NOMINAL calibration -- so a
viewpoint-jitter-induced calibration error is a first-class, honestly
propagated error source in this project, not something hidden.

Ground truth is generated automatically (never hand-labeled): every image
comes from a call to ``render_scene`` and is paired with the exact
(pixel, world) coordinates used to draw it, plus the nuisance parameters
(orientation, size, brightness, background type, noise level, camera
jitter) that produced it.
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "manipulator_description", "scripts"))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import kinematics as kin  # noqa: E402  (existing, unchanged analytical IK; pure Python, no ROS)

# ---------------------------------------------------------------------------
# Robot/workspace geometry -- pulled from kinematics.py's link lengths so the
# sampled object positions are guaranteed reachable, without duplicating or
# hand-tuning the numbers here. See integration.py for the actual import.
# ---------------------------------------------------------------------------
BASE_HEIGHT = 0.15
LINK1_LENGTH = 0.30
LINK2_LENGTH = 0.30
EE_LENGTH = 0.10

TABLE_Z = 0.015  # matches pick_and_place.OBJECT_POS[2]
OBJECT_RADIUS_M = 0.045  # nominal object half-width, meters

# ---------------------------------------------------------------------------
# Camera model (pure pinhole, see module docstring)
# ---------------------------------------------------------------------------
IMG_SIZE = 64
SUPERSAMPLE = 4  # anti-aliasing factor for rendering the object blob

CAM_NOMINAL_POS = np.array([0.0, 0.0, 1.0])   # above the robot's base axis
FOCAL_PX = 43.68                               # chosen so the reachable
                                                # annulus fits the frame with
                                                # a small margin, see
                                                # docs/RESEARCH.md
PRINCIPAL_PX = IMG_SIZE / 2.0

# world -> nominal-camera-frame rotation: camera looks straight down
# (+Z_cam == -Z_world), +X_cam == +X_world, +Y_cam == -Y_world.
R_NOMINAL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])

CAM_ROT_JITTER_RAD = math.radians(3.0)   # +/- max tilt perturbation
CAM_POS_JITTER_XY_M = 0.03
CAM_POS_JITTER_Z_M = 0.02


def reachable_radius_bounds():
    """
    Returns (r_min, r_max): bounds on hypot(x, y) for which an object resting
    at z=TABLE_Z is reachable by the arm, derived from the same geometry
    kinematics.inverse_kinematics() uses (2-link planar reach annulus
    [|L2-L3|, L2+L3] measured from the shoulder point, offset by LINK1 and by
    the base->table height difference).
    """
    L2, L3 = LINK2_LENGTH, EE_LENGTH
    z_rel = TABLE_Z - BASE_HEIGHT
    dist_min, dist_max = abs(L2 - L3), (L2 + L3)
    # r^2 + z_rel^2 = dist^2  =>  r = sqrt(dist^2 - z_rel^2), needs dist>|z_rel|
    r_at_dist_min = math.sqrt(max(0.0, dist_min ** 2 - z_rel ** 2))
    r_at_dist_max = math.sqrt(max(0.0, dist_max ** 2 - z_rel ** 2))
    r_min = LINK1_LENGTH + r_at_dist_min
    r_max = LINK1_LENGTH + r_at_dist_max
    return r_min, r_max


def sample_reachable_object_xy(rng, margin=0.01, max_tries=200):
    """
    Uniform-in-annulus sample of an (x, y) at z=TABLE_Z, REJECTION-SAMPLED
    against the real analytic IK (kinematics.inverse_kinematics) so every
    returned point is actually reachable within joint limits -- the link-
    length-only annulus (reachable_radius_bounds) is necessary but not
    sufficient near its inner edge, where joint limits (q2/q3 ranges) cut
    into the geometric annulus.
    """
    r_min, r_max = reachable_radius_bounds()
    r_min += margin
    r_max -= margin
    for _ in range(max_tries):
        u = rng.uniform(0.0, 1.0)
        r = math.sqrt(u * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        theta = rng.uniform(-math.pi, math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        if kin.inverse_kinematics(x, y, TABLE_Z) is not None:
            return x, y
    raise RuntimeError("sample_reachable_object_xy: could not find a reachable point")


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class CameraPose:
    position: np.ndarray
    rotation: np.ndarray  # world -> camera


def nominal_camera_pose():
    return CameraPose(position=CAM_NOMINAL_POS.copy(), rotation=R_NOMINAL.copy())


def jittered_camera_pose(rng):
    """A random small perturbation of the nominal camera pose (see module docstring)."""
    dpos = np.array([
        rng.uniform(-CAM_POS_JITTER_XY_M, CAM_POS_JITTER_XY_M),
        rng.uniform(-CAM_POS_JITTER_XY_M, CAM_POS_JITTER_XY_M),
        rng.uniform(-CAM_POS_JITTER_Z_M, CAM_POS_JITTER_Z_M),
    ])
    rx = rng.uniform(-CAM_ROT_JITTER_RAD, CAM_ROT_JITTER_RAD)
    ry = rng.uniform(-CAM_ROT_JITTER_RAD, CAM_ROT_JITTER_RAD)
    rz = rng.uniform(-CAM_ROT_JITTER_RAD, CAM_ROT_JITTER_RAD)
    R_jit = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    return CameraPose(position=CAM_NOMINAL_POS + dpos, rotation=R_jit @ R_NOMINAL)


def project_point(world_xyz, cam_pose: CameraPose, focal_px=FOCAL_PX):
    """Pinhole-project a world point to (u, v) pixel coordinates, plus camera-frame depth."""
    p_rel = np.asarray(world_xyz, dtype=float) - cam_pose.position
    p_cam = cam_pose.rotation @ p_rel
    depth = p_cam[2]
    u = focal_px * p_cam[0] / depth + PRINCIPAL_PX
    v = focal_px * p_cam[1] / depth + PRINCIPAL_PX
    return u, v, depth


def backproject_nominal(u, v, world_z=TABLE_Z, focal_px=FOCAL_PX):
    """
    Inverse of project_point() using the NOMINAL (un-jittered) camera pose,
    for a known world_z plane. This is what a real system does at inference
    time: it only has the factory/calibrated camera pose, not the true
    (possibly perturbed) one -- see module docstring.
    """
    depth = CAM_NOMINAL_POS[2] - world_z  # R_NOMINAL's structure makes this exact
    x = CAM_NOMINAL_POS[0] + (u - PRINCIPAL_PX) * depth / focal_px
    y = CAM_NOMINAL_POS[1] - (v - PRINCIPAL_PX) * depth / focal_px
    return x, y


BACKGROUND_TYPES = ("flat", "gradient", "checker")


def _render_background(rng, bg_type, base_level):
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float64)
    if bg_type == "flat":
        bg = np.full((IMG_SIZE, IMG_SIZE), base_level)
    elif bg_type == "gradient":
        grad = (xx + yy) / (2 * IMG_SIZE)
        bg = base_level * (0.6 + 0.4 * grad)
    elif bg_type == "checker":
        cell = IMG_SIZE // 8
        checker = ((xx // cell).astype(int) + (yy // cell).astype(int)) % 2
        bg = base_level * (0.75 + 0.25 * checker)
    else:
        raise ValueError(f"unknown background type {bg_type!r}")
    return bg


@dataclass
class SceneSample:
    image: np.ndarray          # (IMG_SIZE, IMG_SIZE) float32 in [0, 1]
    pixel_uv: tuple            # ground-truth pixel centroid (actual, jittered camera)
    world_xy: tuple            # ground-truth world (x, y) at z=TABLE_Z
    orientation_rad: float
    size_scale: float
    brightness: float
    noise_level: float
    bg_type: str
    cam_jitter_deg: float       # magnitude of rotational jitter used, for analysis


def render_scene(rng, bg_types=BACKGROUND_TYPES, noise_range=(0.0, 0.06),
                  size_range=(0.8, 1.3), brightness_range=(0.6, 1.4)):
    """
    Draws one random scene and returns a SceneSample with image + full
    automatically-generated ground truth / nuisance parameters.
    """
    x, y = sample_reachable_object_xy(rng)
    theta = rng.uniform(-math.pi, math.pi)
    size_scale = rng.uniform(*size_range)
    brightness = rng.uniform(*brightness_range)
    noise_level = rng.uniform(*noise_range)
    bg_type = bg_types[rng.integers(0, len(bg_types))]
    bg_level = rng.uniform(0.15, 0.35)
    obj_level = rng.uniform(0.75, 0.95)

    cam_pose = jittered_camera_pose(rng)
    u, v, depth = project_point((x, y, TABLE_Z), cam_pose)
    if depth <= 0:
        # Degenerate (camera jittered below the table) -- extremely rare given
        # the jitter magnitudes above; resample once.
        return render_scene(rng, bg_types, noise_range, size_range, brightness_range)

    render_size = IMG_SIZE * SUPERSAMPLE
    jj, ii = np.meshgrid(np.arange(render_size), np.arange(render_size))
    du = (jj - u * SUPERSAMPLE)
    dv = (ii - v * SUPERSAMPLE)
    ct, st = math.cos(-theta), math.sin(-theta)
    du_r = du * ct - dv * st
    dv_r = du * st + dv * ct

    base_radius_px = size_scale * FOCAL_PX * OBJECT_RADIUS_M / max(depth, 1e-6)
    a = max(base_radius_px * SUPERSAMPLE * 1.4, 1.0)   # major axis
    b = max(base_radius_px * SUPERSAMPLE * 0.8, 1.0)   # minor axis
    blob_hi = np.exp(-((du_r / a) ** 2 + (dv_r / b) ** 2))

    blob = blob_hi.reshape(IMG_SIZE, SUPERSAMPLE, IMG_SIZE, SUPERSAMPLE).mean(axis=(1, 3))

    bg = _render_background(rng, bg_type, bg_level)
    image = bg + blob * obj_level
    image = image * brightness
    image = image + rng.normal(0.0, noise_level, size=image.shape)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    return SceneSample(
        image=image, pixel_uv=(float(u), float(v)), world_xy=(float(x), float(y)),
        orientation_rad=float(theta), size_scale=float(size_scale),
        brightness=float(brightness), noise_level=float(noise_level), bg_type=bg_type,
        cam_jitter_deg=float(math.degrees(
            math.acos(np.clip((np.trace(cam_pose.rotation @ R_NOMINAL.T) - 1) / 2, -1.0, 1.0)))),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    s = render_scene(rng)
    print(f"rendered image shape={s.image.shape}, pixel_uv={s.pixel_uv}, "
          f"world_xy={s.world_xy}, bg_type={s.bg_type}, brightness={s.brightness:.2f}, "
          f"noise_level={s.noise_level:.3f}")
    r_min, r_max = reachable_radius_bounds()
    print(f"reachable radius bounds: [{r_min:.4f}, {r_max:.4f}] m")
