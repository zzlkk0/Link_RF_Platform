"""Masked multi-snapshot path recovery with held-tone predictive evidence.

This estimates propagation lengths, not geometric LOS guarantees. Independent
complex coefficients absorb snapshot phase offsets without oracle calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..constants import C0


@dataclass(frozen=True)
class SequenceRecoveryConfig:
    max_distance_m: float = 45.0
    grid_step_m: float = .25
    max_paths: int = 4
    max_refinement_evaluations: int = 25
    minimum_tones: int = 16
    minimum_amplitude_ratio: float = .025
    minimum_predictive_gain: float = .003
    minimum_frame_support: float = .625
    maximum_prediction_residual: float = .45
    complexity_weight: float = 1.0
    path_factor: float = 1.0
    receiver_ripple_period_hz: float | None = 20e6
    maximum_ripple_coefficient_ratio: float = .2

    def __post_init__(self):
        if not np.isfinite(self.max_distance_m) or self.max_distance_m <= 0:
            raise ValueError("distance prior must be positive and finite")
        if not np.isfinite(self.grid_step_m) or self.grid_step_m <= 0:
            raise ValueError("grid step must be positive and finite")
        if not 1 <= self.max_paths <= 6 or self.max_refinement_evaluations < 1:
            raise ValueError("invalid model size or refinement budget")
        if self.minimum_tones < 8 or self.path_factor <= 0 or not np.isfinite(self.path_factor):
            raise ValueError("invalid tone support or phase convention")
        for value in (self.minimum_amplitude_ratio, self.minimum_frame_support, self.maximum_prediction_residual):
            if not 0 < value <= 1:
                raise ValueError("relative thresholds must lie in (0, 1]")
        if self.minimum_predictive_gain < 0 or self.complexity_weight < 0:
            raise ValueError("penalties and predictive gain cannot be negative")
        if self.receiver_ripple_period_hz is not None and (not np.isfinite(self.receiver_ripple_period_hz) or self.receiver_ripple_period_hz <= 0):
            raise ValueError("receiver calibration period must be finite and positive")
        if not 0 <= self.maximum_ripple_coefficient_ratio < 1:
            raise ValueError("receiver calibration ripple bound must lie in [0, 1)")


@dataclass(frozen=True)
class SequenceRecoveryResult:
    distance_m: float | None
    available: bool
    path_distances_m: tuple[float, ...]
    amplitude_ratios: tuple[float, ...]
    predictive_gains: tuple[float, ...]
    frame_supports: tuple[float, ...]
    prediction_residual: float
    model_order: int
    selected_index: int | None
    training_bic: float
    received_fraction: float
    reason: str


def _steering(centered_frequencies, distances, factor):
    return np.exp(-2j * np.pi * centered_frequencies[:, None] * np.asarray(distances)[None, :] * factor / C0)


def _coefficients(steering, observations, mask, nuisance=None, ripple_bound=.2):
    design = steering if nuisance is None else np.concatenate((steering, steering * nuisance[:, :1], steering * nuisance[:, 1:]), axis=1)
    path_count = steering.shape[1]
    coefficients = []
    predictions = np.zeros_like(observations)
    for snapshot in range(len(observations)):
        used = mask[snapshot]
        coefficient = np.linalg.lstsq(design[used], observations[snapshot, used], rcond=1e-7)[0]
        if nuisance is not None:
            bound = np.tile(np.abs(coefficient[:path_count]), 2) * ripple_bound
            coefficient[path_count:] *= np.minimum(1, bound / np.maximum(np.abs(coefficient[path_count:]), 1e-12))
        coefficients.append(coefficient[:path_count])
        predictions[snapshot] = design @ coefficient
    return np.asarray(coefficients), predictions


def _refine(initial, frequencies, observations, fitting_mask, config, nuisance):
    def objective(distances):
        steering = _steering(frequencies, distances, config.path_factor)
        _, prediction = _coefficients(steering, observations, fitting_mask, nuisance, config.maximum_ripple_coefficient_ratio)
        residual = (observations - prediction)[fitting_mask]
        return np.concatenate((residual.real, residual.imag))

    result = least_squares(objective, np.clip(initial, 1e-7, config.max_distance_m - 1e-7),
                           bounds=(0, config.max_distance_m), max_nfev=config.max_refinement_evaluations,
                           diff_step=1e-4, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    distances = np.sort(result.x)
    steering = _steering(frequencies, distances, config.path_factor)
    coefficients, predictions = _coefficients(steering, observations, fitting_mask, nuisance, config.maximum_ripple_coefficient_ratio)
    error = float(np.sum(np.abs((observations - predictions)[fitting_mask]) ** 2))
    observations_count = 2 * int(fitting_mask.sum())
    parameters = len(distances) * ((6 if nuisance is not None else 2) * len(observations) + 1)
    bic = observations_count * np.log(max(error / observations_count, 1e-15))
    bic += config.complexity_weight * parameters * np.log(observations_count)
    return distances, coefficients, predictions, float(bic)


def recover_sequence(frequencies_hz, iq, mask=None, config=None):
    """Fit shared delays on received tones; score on interleaved held-out tones."""
    cfg = config or SequenceRecoveryConfig()
    frequencies = np.asarray(frequencies_hz, float)
    observations = np.asarray(iq, np.complex128)
    if observations.ndim == 1:
        observations = observations[None, :]
    if observations.ndim != 2 or frequencies.ndim != 1 or observations.shape[1] != len(frequencies):
        raise ValueError("expected [snapshot, frequency] measurements")
    if len(frequencies) < cfg.minimum_tones or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be finite, increasing and sufficiently supported")
    if not np.isfinite(observations).all():
        raise ValueError("I/Q must be finite")
    received = np.ones(observations.shape, bool) if mask is None else np.asarray(mask)
    if received.shape != observations.shape or received.dtype != bool:
        raise ValueError("mask must be a matching boolean matrix")
    if np.any(received.sum(-1) < cfg.minimum_tones):
        raise ValueError("insufficient received tone support")
    centered = frequencies - np.mean(frequencies)
    nuisance = None if cfg.receiver_ripple_period_hz is None else np.column_stack((
        np.cos(2 * np.pi * centered / cfg.receiver_ripple_period_hz),
        np.sin(2 * np.pi * centered / cfg.receiver_ripple_period_hz)))
    held_columns = np.arange(len(frequencies)) % 4 == 1
    fitting = received & ~held_columns[None, :]
    held = received & held_columns[None, :]
    if np.any(fitting.sum(-1) < 6) or np.any(held.sum(-1) < 3):
        raise ValueError("insufficient independent fitting or validation tones")
    scales = np.sqrt(np.sum(np.abs(observations) ** 2 * fitting, axis=1) / fitting.sum(-1))
    observations = np.where(received, observations, 0) / np.maximum(scales[:, None], 1e-12)
    grid = np.arange(0, cfg.max_distance_m + cfg.grid_step_m / 2, cfg.grid_step_m)
    dictionary = _steering(centered, grid, cfg.path_factor)
    residual = observations.copy()
    candidates = []
    distances = np.empty(0)
    for order in range(1, cfg.max_paths + 1):
        spectrum = np.abs((np.where(fitting, residual, 0) @ dictionary.conj())) ** 2
        spectrum /= np.maximum(fitting.sum(-1, keepdims=True), 1) ** 2
        score = np.mean(spectrum, axis=0)
        if len(distances):
            score[np.min(abs(grid[:, None] - distances[None, :]), axis=1) < .35] = -np.inf
        additional = grid[int(np.argmax(score))]
        initial = np.append(distances, additional)
        fits = [_refine(initial, centered, observations, fitting, cfg, nuisance)]
        if order == 2:
            peak = distances[0]
            for offset in (.6, 1.5):
                fits.append(_refine([max(0, peak - offset), min(cfg.max_distance_m, peak + offset)],
                                    centered, observations, fitting, cfg, nuisance))
        fitted = min(fits, key=lambda item: item[3])
        distances, coefficients, prediction, bic = fitted
        if len(distances) == 1 or np.min(np.diff(distances)) > .05:
            candidates.append(fitted)
        residual = observations - prediction
    distances, coefficients, prediction, bic = min(candidates, key=lambda item: item[3])
    held_energy = np.sum(np.abs(observations) ** 2 * held, axis=1)
    full_error = np.sum(np.abs(observations - prediction) ** 2 * held, axis=1)
    relative_error = full_error / np.maximum(held_energy, 1e-12)
    amplitudes = np.sqrt(np.mean(np.abs(coefficients) ** 2, axis=0))
    ratios = amplitudes / max(float(amplitudes.max()), 1e-12)
    gains, supports = [], []
    for path_index in range(len(distances)):
        remaining = np.delete(distances, path_index)
        if len(remaining):
            _, reduced = _coefficients(_steering(centered, remaining, cfg.path_factor), observations, fitting, nuisance, cfg.maximum_ripple_coefficient_ratio)
        else:
            reduced = np.zeros_like(observations)
        reduced_error = np.sum(np.abs(observations - reduced) ** 2 * held, axis=1)
        gain = (reduced_error - full_error) / np.maximum(held_energy, 1e-12)
        gains.append(float(np.mean(gain)))
        supports.append(float(np.mean(gain > cfg.minimum_predictive_gain)))
    significant = (ratios >= cfg.minimum_amplitude_ratio) & (np.asarray(gains) >= cfg.minimum_predictive_gain)
    significant &= np.asarray(supports) >= cfg.minimum_frame_support
    selected = int(np.flatnonzero(significant)[0]) if significant.any() else None
    prediction_residual = float(np.sqrt(np.mean(relative_error)))
    available = selected is not None and prediction_residual <= cfg.maximum_prediction_residual
    return SequenceRecoveryResult(
        distance_m=float(distances[selected]) if selected is not None else None,
        available=bool(available), path_distances_m=tuple(float(value) for value in distances),
        amplitude_ratios=tuple(float(value) for value in ratios), predictive_gains=tuple(gains),
        frame_supports=tuple(supports), prediction_residual=prediction_residual,
        model_order=len(distances), selected_index=selected, training_bic=bic,
        received_fraction=float(received.mean()),
        reason="supported_propagation_length_not_los_certificate" if available else "insufficient_predictive_path_evidence",
    )
