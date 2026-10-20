import numpy as np
import pytest
from volumetric_light import LightFieldOptimizer, plane_metrics, propagate
from volumetric_light.optimizer import quantize_phase


def target(center):
    yy, xx = np.indices((32, 32))
    return (((xx-center[1])**2 + (yy-center[0])**2) < 25).astype(float)


def test_zero_distance_is_identity():
    field = np.exp(1j * np.linspace(0, 1, 1024).reshape(32, 32))
    assert np.allclose(propagate(field, 0, 532e-9, 8e-6), field)


def test_propagation_approximately_preserves_energy():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(32,32)) + 1j*rng.normal(size=(32,32))
    moved = propagate(field, 0.05, 532e-9, 8e-6)
    assert np.isclose(np.linalg.norm(field), np.linalg.norm(moved), rtol=1e-6)


def test_quantization_uses_bounded_number_of_levels():
    phase = np.linspace(-3, 10, 1000)
    assert len(np.unique(np.round(quantize_phase(phase, 16), 10))) <= 16


def test_seed_makes_optimization_reproducible():
    optimizer = LightFieldOptimizer(phase_levels=64)
    a = optimizer.optimize([target((10,10)), target((22,22))], [.04,.07], iterations=8, seed=2)
    b = optimizer.optimize([target((10,10)), target((22,22))], [.04,.07], iterations=8, seed=2)
    assert np.allclose(a.phase, b.phase)
    assert a.loss_history == b.loss_history


def test_adaptive_weights_are_probabilities():
    result = LightFieldOptimizer().optimize([target((10,10)), target((22,22))], [.04,.07], iterations=5)
    assert sum(result.plane_weights) == pytest.approx(1)
    assert all(weight > 0 for weight in result.plane_weights)


def test_metrics_are_finite_and_bounded_where_expected():
    field = np.ones((32,32), complex)
    metrics = plane_metrics(field, target((16,16)))
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["efficiency"] <= 1


def test_invalid_plane_configuration_is_rejected():
    with pytest.raises(ValueError):
        LightFieldOptimizer().optimize([target((10,10))], [.04,.07])
