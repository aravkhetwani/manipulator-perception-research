import math

import numpy as np
import pytest

from perception_research import sim_camera as sc


def test_reachable_bounds_positive_and_ordered():
    r_min, r_max = sc.reachable_radius_bounds()
    assert 0.0 < r_min < r_max


def test_sampled_points_are_actually_reachable():
    import kinematics as kin  # picked up via sim_camera's sys.path insert
    rng = np.random.default_rng(3)
    for _ in range(100):
        x, y = sc.sample_reachable_object_xy(rng)
        r = math.hypot(x, y)
        r_min, r_max = sc.reachable_radius_bounds()
        assert r_min - 1e-6 <= r <= r_max + 1e-6
        assert kin.inverse_kinematics(x, y, sc.TABLE_Z) is not None


def test_projection_backprojection_roundtrip_at_nominal_pose():
    """With NO camera jitter, projecting then back-projecting with the
    nominal calibration must recover the original (x, y) essentially
    exactly (this is the affine special case documented in the module)."""
    nominal = sc.nominal_camera_pose()
    rng = np.random.default_rng(11)
    for _ in range(50):
        x, y = sc.sample_reachable_object_xy(rng)
        u, v, depth = sc.project_point((x, y, sc.TABLE_Z), nominal)
        assert depth > 0
        x2, y2 = sc.backproject_nominal(u, v)
        assert math.isclose(x, x2, abs_tol=1e-9)
        assert math.isclose(y, y2, abs_tol=1e-9)


def test_render_scene_shape_and_range():
    rng = np.random.default_rng(42)
    s = sc.render_scene(rng)
    assert s.image.shape == (sc.IMG_SIZE, sc.IMG_SIZE)
    assert s.image.dtype == np.float32
    assert 0.0 <= s.image.min() and s.image.max() <= 1.0
    assert s.bg_type in sc.BACKGROUND_TYPES


def test_render_scene_is_reproducible_with_fixed_seed():
    s1 = sc.render_scene(np.random.default_rng(7))
    s2 = sc.render_scene(np.random.default_rng(7))
    np.testing.assert_array_equal(s1.image, s2.image)
    assert s1.pixel_uv == s2.pixel_uv
    assert s1.world_xy == s2.world_xy


def test_object_blob_is_near_ground_truth_pixel():
    """The brightest region of the rendered image should be close to the
    labeled ground-truth pixel (sanity check that labels aren't decoupled
    from the actual rendering)."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        s = sc.render_scene(rng, noise_range=(0.0, 0.0))
        idx = np.unravel_index(np.argmax(s.image), s.image.shape)
        peak_v, peak_u = idx
        u, v = s.pixel_uv
        # allow a few pixels of slack: peak search is on a coarse pooled grid
        assert abs(peak_u - u) <= 3
        assert abs(peak_v - v) <= 3
