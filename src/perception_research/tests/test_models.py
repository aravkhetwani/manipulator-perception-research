import numpy as np

from perception_research import sim_camera as sc
from perception_research.models import ClassicalCentroidBaseline, MLPCoordinateRegressor


def _small_dataset(n=40, seed=1):
    rng = np.random.default_rng(seed)
    images = np.empty((n, sc.IMG_SIZE, sc.IMG_SIZE), dtype=np.float32)
    uv = np.empty((n, 2))
    for i in range(n):
        s = sc.render_scene(rng, noise_range=(0.0, 0.02))
        images[i] = s.image
        uv[i] = s.pixel_uv
    return images, uv


def test_classical_baseline_predicts_within_object_neighborhood():
    images, uv = _small_dataset(n=30, seed=2)
    model = ClassicalCentroidBaseline().fit(images, uv)
    pred = model.predict(images)
    err = np.linalg.norm(pred - uv, axis=1)
    # A percentile-threshold centroid on these varied-background/brightness
    # scenes is a real, imperfect baseline (see results/metrics.json:
    # test_localization.classical_baseline, ~10px MAE on the full test
    # split) -- this just checks it's in the right neighborhood of the
    # image, not off by tens of pixels or picking a different blob entirely.
    assert err.mean() < 15.0


def test_classical_baseline_output_shape():
    images, uv = _small_dataset(n=5, seed=3)
    model = ClassicalCentroidBaseline()
    pred = model.predict(images)
    assert pred.shape == (5, 2)


def test_mlp_regressor_trains_and_predicts_right_shape():
    images, uv = _small_dataset(n=60, seed=4)
    model = MLPCoordinateRegressor(hidden_layer_sizes=(16,), max_iter=30, seed=0)
    model.fit(images, uv)
    pred = model.predict(images)
    assert pred.shape == uv.shape
    assert np.all(np.isfinite(pred))


def test_mlp_regressor_save_load_roundtrip(tmp_path):
    images, uv = _small_dataset(n=40, seed=6)
    model = MLPCoordinateRegressor(hidden_layer_sizes=(8,), max_iter=20, seed=0)
    model.fit(images, uv)
    path = tmp_path / "model.pkl"
    model.save(str(path))
    loaded = MLPCoordinateRegressor.load(str(path))
    pred_before = model.predict(images)
    pred_after = loaded.predict(images)
    np.testing.assert_allclose(pred_before, pred_after)
