"""Phase-only multi-depth hologram synthesis for reproducible research experiments."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _frequency_grid(shape: tuple[int, int], pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(shape[0], d=pitch_m)
    fx = np.fft.fftfreq(shape[1], d=pitch_m)
    return np.meshgrid(fx, fy)


def transfer_function(shape: tuple[int, int], z_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """Band-limited angular-spectrum transfer function."""
    fx, fy = _frequency_grid(shape, pitch_m)
    radicand = (1.0 / wavelength_m**2) - fx**2 - fy**2
    propagating = radicand >= 0
    phase = np.zeros(shape, dtype=float)
    phase[propagating] = 2 * np.pi * z_m * np.sqrt(radicand[propagating])
    return propagating * np.exp(1j * phase)


def propagate(field: np.ndarray, z_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    spectrum = np.fft.fft2(field, norm="ortho")
    return np.fft.ifft2(spectrum * transfer_function(field.shape, z_m, wavelength_m, pitch_m), norm="ortho")


def quantize_phase(phase: np.ndarray, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("phase quantization needs at least two levels")
    wrapped = np.mod(phase, 2 * np.pi)
    step = 2 * np.pi / levels
    return np.mod(np.round(wrapped / step) * step, 2 * np.pi)


def plane_metrics(reconstruction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intensity = np.abs(reconstruction) ** 2
    target = np.asarray(target, dtype=float)
    target = target / max(float(target.max()), 1e-12)
    normalized = intensity / max(float(intensity.max()), 1e-12)
    mse = float(np.mean((normalized - target) ** 2))
    roi = target > 0.35
    efficiency = float(intensity[roi].sum() / max(float(intensity.sum()), 1e-12))
    roi_values = intensity[roi]
    speckle = float(roi_values.std() / max(float(roi_values.mean()), 1e-12)) if roi_values.size else 0.0
    background = float(intensity[~roi].mean()) if (~roi).any() else 0.0
    signal = float(intensity[roi].mean()) if roi.any() else 0.0
    crosstalk = background / max(signal, 1e-12)
    return {"mse": mse, "efficiency": efficiency, "speckle": speckle, "crosstalk": crosstalk}


@dataclass(frozen=True)
class OptimizationResult:
    phase: np.ndarray
    loss_history: tuple[float, ...]
    plane_weights: tuple[float, ...]
    metrics: tuple[dict[str, float], ...]


class LightFieldOptimizer:
    """Adaptive weighted Gerchberg-Saxton optimization across target depths."""

    def __init__(self, wavelength_m: float = 532e-9, pixel_pitch_m: float = 8e-6, phase_levels: int = 256):
        self.wavelength_m = wavelength_m
        self.pixel_pitch_m = pixel_pitch_m
        self.phase_levels = phase_levels

    def optimize(self, targets: list[np.ndarray], depths_m: list[float], iterations: int = 40, seed: int = 7) -> OptimizationResult:
        if not targets or len(targets) != len(depths_m):
            raise ValueError("targets and depths must be non-empty and equally sized")
        shape = targets[0].shape
        if any(t.shape != shape for t in targets):
            raise ValueError("all target planes must have the same shape")
        normalized_targets = [np.sqrt(np.clip(t.astype(float), 0, None) / max(float(t.max()), 1e-12)) for t in targets]
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi, shape)
        weights = np.ones(len(targets), dtype=float) / len(targets)
        history: list[float] = []

        for _ in range(iterations):
            slm = np.exp(1j * phase)
            back_fields = []
            losses = []
            for target_amp, depth in zip(normalized_targets, depths_m):
                observed = propagate(slm, depth, self.wavelength_m, self.pixel_pitch_m)
                scale = np.linalg.norm(np.abs(observed)) / max(np.linalg.norm(target_amp), 1e-12)
                constrained = target_amp * scale * np.exp(1j * np.angle(observed))
                back_fields.append(propagate(constrained, -depth, self.wavelength_m, self.pixel_pitch_m))
                obs_norm = np.abs(observed) / max(float(np.abs(observed).max()), 1e-12)
                losses.append(float(np.mean((obs_norm - target_amp) ** 2)))
            combined = sum(w * field for w, field in zip(weights, back_fields))
            phase = quantize_phase(np.angle(combined), self.phase_levels)
            loss_array = np.asarray(losses)
            history.append(float(np.dot(weights, loss_array)))
            difficulty = loss_array + 1e-6
            weights = difficulty / difficulty.sum()

        slm = np.exp(1j * phase)
        metrics = tuple(plane_metrics(propagate(slm, z, self.wavelength_m, self.pixel_pitch_m), target) for target, z in zip(targets, depths_m))
        return OptimizationResult(phase, tuple(history), tuple(float(w) for w in weights), metrics)
