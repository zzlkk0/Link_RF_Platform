"""Independent acceptance suite for the RF ranging research platform.

This suite deliberately does not modify the implementation under test. It
separates numerical evidence from standards coverage and from measurements
that require external hardware or datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_ranging_model import C0, PropagationPath, RangingPipeline
from rf_ranging_model.ble.channel_sounding import allowed_cs_channel_indices
from rf_ranging_model.ble.pbr import estimate_distance
from rf_ranging_model.ble.robust import RobustPBRConfig, estimate_distance_robust
from rf_ranging_model.channel.free_space import free_space_response
from rf_ranging_model.channel.multipath import frequency_response
from rf_ranging_model.channel.stochastic_channel import (
    SalehValenzuelaConfig,
    generate_saleh_valenzuela,
)
from rf_ranging_model.rfic.impairments import (
    RFICImpairmentConfig,
    apply_rfic_impairments,
)
from rf_ranging_model.rfic.calibration import (
    CalibrationResult,
    apply_calibration,
    build_calibration_profile,
)
from rf_ranging_model.rfic.sparameter import load_s2p_s21
from rf_ranging_model.uwb.cir import (
    frequency_response_frames_to_cir,
    frequency_response_to_cir,
)
from rf_ranging_model.uwb.first_path import FirstPathStatus, detect_first_path
from rf_ranging_model.uwb.tof import estimate_distance_from_cir
from rf_ranging_model import webapp as webapp_module


STATUS_PASS = "PASS"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAIL = "FAIL"
STATUS_NOT_TESTED = "NOT TESTED"

STANDARD_REFERENCES = {
    "IEEE 802.15.4-2024": "https://standards.ieee.org/ieee/802.15.4/11041/",
    "IEEE 802.15.4a channel model": (
        "https://mentor.ieee.org/802.15/dcn/04/"
        "15-04-0662-03-004a-channel-model-final-report-r1.pdf"
    ),
    "Bluetooth Core 6.3": (
        "https://www.bluetooth.com/specifications/specs/core-specification-6-3/"
    ),
    "3GPP TR 38.901": (
        "https://portal.3gpp.org/desktopmodules/Specifications/"
        "SpecificationDetails.aspx?specificationId=3173"
    ),
    "ITU-R P.1238-13": (
        "https://www.itu.int/dms_pubrec/itu-r/rec/p/"
        "R-REC-P.1238-13-202509-I%21%21PDF-E.pdf"
    ),
    "ISO/IEC 18305:2016": "https://www.iso.org/standard/62090.html",
    "Touchstone 2.1": "https://ibis.org/specs/",
}


@dataclass(slots=True)
class CheckResult:
    check_id: str
    title: str
    status: str
    criterion: str
    evidence: dict[str, Any]
    note: str = ""


EXPECTED_CS_CHANNEL_INDICES = np.asarray(
    [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, 73, 74, 75, 76,
    ],
    dtype=int,
)


def bluetooth_cs_frequencies_hz() -> np.ndarray:
    """Return the 72 allowed Bluetooth Channel Sounding tone centers.

    Bluetooth Core defines indices 0..78 at 1 MHz spacing from 2402 MHz.
    Seven indices are not allowed for CS communication.
    """

    return 2.402e9 + EXPECTED_CS_CHANNEL_INDICES.astype(float) * 1.0e6


def _run_check(
    check_id: str,
    title: str,
    criterion: str,
    operation: Callable[[], tuple[bool, dict[str, Any], str]],
) -> CheckResult:
    try:
        passed, evidence, note = operation()
        return CheckResult(
            check_id,
            title,
            STATUS_PASS if passed else STATUS_FAIL,
            criterion,
            evidence,
            note,
        )
    except Exception as exc:  # acceptance harness must report, not abort
        return CheckResult(
            check_id,
            title,
            STATUS_FAIL,
            criterion,
            {"exception_type": type(exc).__name__, "exception": str(exc)},
            "验收操作发生异常。",
        )


def _float(value: float) -> float:
    return float(np.asarray(value))


def check_ble_los_sweep() -> tuple[bool, dict[str, Any], str]:
    frequencies = bluetooth_cs_frequencies_hz()
    distances = np.unique(
        np.r_[
            np.arange(0.5, 149.0 + 0.25, 0.5),
            [37.49, 37.50, 37.51, 74.94, 74.95, 74.96, 149.0],
        ]
    )
    estimates: list[float] = []
    residuals: list[float] = []
    peak_ratios: list[float] = []
    coarse_errors: list[float] = []
    for distance in distances:
        delay_s = float(distance / C0)
        independent_response = np.exp(
            -1j * 2.0 * np.pi * frequencies * delay_s
        )
        result = estimate_distance(
            frequencies,
            independent_response,
            max_distance_m=149.0,
        )
        estimates.append(result.distance_m)
        residuals.append(result.residual_rms_rad)
        peak_ratios.append(float(result.ambiguity_peak_ratio or 0.0))
        coarse_errors.append(
            C0 * float(result.coarse_delay_s or 0.0) - float(distance)
        )
    estimates_array = np.asarray(estimates)
    errors = estimates_array - distances
    maximum_error = _float(np.max(np.abs(errors)))
    finite = bool(np.all(np.isfinite(estimates_array)))
    production_map_matches = bool(
        np.array_equal(allowed_cs_channel_indices(), EXPECTED_CS_CHANNEL_INDICES)
    )
    worst_index = int(np.argmax(np.abs(errors)))
    return (
        frequencies.size == 72
        and production_map_matches
        and finite
        and maximum_error < 1.0e-8,
        {
            "channel_count": int(frequencies.size),
            "full_channel_index_map_matches_golden": production_map_matches,
            "channel_indices": EXPECTED_CS_CHANNEL_INDICES.tolist(),
            "first_frequency_mhz": _float(frequencies[0] / 1.0e6),
            "last_frequency_mhz": _float(frequencies[-1] / 1.0e6),
            "distance_min_m": _float(distances[0]),
            "distance_max_m": _float(distances[-1]),
            "sample_count": int(distances.size),
            "availability": float(np.mean(np.isfinite(estimates_array))),
            "max_abs_error_m": maximum_error,
            "worst_distance_m": float(distances[worst_index]),
            "worst_error_m": float(errors[worst_index]),
            "max_abs_coarse_error_m": float(np.max(np.abs(coarse_errors))),
            "max_phase_residual_rms_rad": float(np.max(residuals)),
            "minimum_ambiguity_peak_ratio": float(np.min(peak_ratios)),
        },
        "独立公式生成测试向量；验证全局模糊度解析，不等于 Bluetooth CS 协议一致性。",
    )


def check_ble_round_trip_factor() -> tuple[bool, dict[str, Any], str]:
    frequencies = bluetooth_cs_frequencies_hz()
    distances = np.unique(
        np.r_[
            np.arange(0.5, 100.0 + 0.25, 0.5),
            [37.49, 37.50, 37.51, 74.94, 74.95, 74.96, 100.0],
        ]
    )
    estimates: list[float] = []
    peak_ratios: list[float] = []
    for distance in distances:
        round_trip_delay = 2.0 * distance / C0
        response = np.exp(-1j * 2.0 * np.pi * frequencies * round_trip_delay)
        result = estimate_distance(
            frequencies,
            response,
            path_factor=2.0,
            max_distance_m=100.0,
        )
        estimates.append(result.distance_m)
        peak_ratios.append(float(result.ambiguity_peak_ratio or 0.0))
    errors = np.asarray(estimates) - distances
    maximum_error = _float(np.max(np.abs(errors)))
    worst_index = int(np.argmax(np.abs(errors)))
    return (
        bool(np.all(np.isfinite(estimates))) and maximum_error < 1.0e-8,
        {
            "distance_min_m": _float(distances[0]),
            "distance_max_m": _float(distances[-1]),
            "sample_count": int(distances.size),
            "max_search_distance_m": 100.0,
            "theoretical_unambiguous_distance_m": C0 / (2.0e6),
            "availability": float(np.mean(np.isfinite(estimates))),
            "max_abs_error_m": maximum_error,
            "worst_distance_m": float(distances[worst_index]),
            "worst_error_m": float(errors[worst_index]),
            "minimum_ambiguity_peak_ratio": float(np.min(peak_ratios)),
        },
        "仅验证 PBR 往返相位/path_factor=2，不是 Bluetooth Mode-1 RTT 时间戳算法。",
    )


def check_uwb_los_resolution() -> tuple[bool, dict[str, Any], str]:
    uwb_frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    bandwidth_hz = float(np.ptp(uwb_frequencies))
    one_way_resolution_m = C0 / bandwidth_hz
    distances = np.unique(
        np.r_[np.arange(0.5, 149.0 + 0.25, 0.5), [1.125, 7.375, 37.625]]
    )
    estimates: list[float] = []
    crossing_estimates: list[float] = []
    distance_bin_m: float | None = None
    for distance in distances:
        independent_response = np.exp(
            -1j * 2.0 * np.pi * uwb_frequencies * float(distance / C0)
        )
        cir_delay_s, cir = frequency_response_to_cir(
            uwb_frequencies,
            independent_response,
            zero_pad_factor=8,
        )
        estimate, first_path = estimate_distance_from_cir(cir_delay_s, cir)
        estimates.append(estimate)
        crossing_estimates.append(C0 * first_path.crossing_delay_s)
        if distance_bin_m is None:
            distance_bin_m = C0 * float(np.median(np.diff(cir_delay_s)))
    errors = np.asarray(estimates) - distances
    crossing_errors = np.asarray(crossing_estimates) - distances
    maximum_error = _float(np.max(np.abs(errors)))
    assert distance_bin_m is not None
    worst_index = int(np.argmax(np.abs(errors)))
    return (
        bool(np.all(np.isfinite(estimates))) and maximum_error < distance_bin_m,
        {
            "bandwidth_hz": bandwidth_hz,
            "one_way_resolution_m": one_way_resolution_m,
            "zero_pad_factor": 8,
            "distance_grid_bin_m": distance_bin_m,
            "distance_min_m": float(distances[0]),
            "distance_max_m": float(distances[-1]),
            "sample_count": int(distances.size),
            "availability": float(np.mean(np.isfinite(estimates))),
            "max_abs_error_m": maximum_error,
            "mean_bias_m": _float(np.mean(errors)),
            "worst_distance_m": float(distances[worst_index]),
            "worst_error_m": float(errors[worst_index]),
            "legacy_crossing_mean_bias_m": float(np.mean(crossing_errors)),
        },
        "门槛收紧为一个 8 倍零填充距离格；测试向量由独立相位公式生成。",
    )


def check_group_delay_mapping() -> tuple[bool, dict[str, Any], str]:
    frequencies = bluetooth_cs_frequencies_hz()
    true_distance = 7.0
    channel = frequency_response(
        frequencies,
        [PropagationPath(true_distance / C0, 1.0 + 0.0j, "LOS")],
    )
    delays = np.asarray([-0.8e-9, 0.0, 0.2e-9, 1.0e-9, 3.0e-9])
    errors: list[float] = []
    for delay in delays:
        measured = apply_rfic_impairments(
            frequencies,
            channel,
            RFICImpairmentConfig(base_group_delay_s=float(delay)),
        )
        estimate = estimate_distance(frequencies, measured).distance_m
        expected = true_distance + C0 * delay
        errors.append(estimate - expected)
    maximum_error = float(np.max(np.abs(errors)))
    return (
        maximum_error < 1.0e-8,
        {
            "delay_min_ns": _float(np.min(delays) * 1.0e9),
            "delay_max_ns": _float(np.max(delays) * 1.0e9),
            "max_abs_distance_mapping_error_m": maximum_error,
        },
        "",
    )


def check_group_delay_calibration() -> tuple[bool, dict[str, Any], str]:
    """Verify exact scalar mapping and reject a grossly wrong calibration."""

    ble_frequencies = bluetooth_cs_frequencies_hz()
    uwb_frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    true_distance_m = 7.0
    injected_delays_s = np.asarray(
        [-0.8e-9, 0.0, 0.2e-9, 1.0e-9, 3.0e-9]
    )
    records: list[dict[str, float]] = []
    ble_corrected_errors: list[float] = []
    uwb_corrected_errors: list[float] = []
    raw_bias_mapping_errors: list[float] = []
    uwb_bins_m: list[float] = []
    for path_factor in (1.0, 2.0):
        for injected_delay_s in injected_delays_s:
            propagation_delay_s = (
                path_factor * true_distance_m / C0 + injected_delay_s
            )
            ble_response = np.exp(
                -1j
                * 2.0
                * np.pi
                * ble_frequencies
                * propagation_delay_s
            )
            ble_result = estimate_distance(
                ble_frequencies,
                ble_response,
                path_factor=path_factor,
                max_distance_m=100.0 if path_factor == 2.0 else 149.0,
            )
            expected_raw_bias_m = C0 * injected_delay_s / path_factor
            ble_corrected_m = (
                ble_result.distance_m - expected_raw_bias_m
            )

            uwb_response = np.exp(
                -1j
                * 2.0
                * np.pi
                * uwb_frequencies
                * propagation_delay_s
            )
            cir_delay_s, cir = frequency_response_to_cir(
                uwb_frequencies,
                uwb_response,
                zero_pad_factor=8,
            )
            uwb_corrected_m, first_path = estimate_distance_from_cir(
                cir_delay_s,
                cir,
                path_factor=path_factor,
                calibration_delay_s=float(injected_delay_s),
            )
            uwb_raw_m = C0 * first_path.delay_s / path_factor
            uwb_bins_m.append(
                C0
                * float(np.median(np.diff(cir_delay_s)))
                / path_factor
            )
            ble_error_m = ble_corrected_m - true_distance_m
            uwb_error_m = uwb_corrected_m - true_distance_m
            raw_mapping_error_m = (
                ble_result.distance_m
                - true_distance_m
                - expected_raw_bias_m
            )
            ble_corrected_errors.append(ble_error_m)
            uwb_corrected_errors.append(uwb_error_m)
            raw_bias_mapping_errors.append(raw_mapping_error_m)
            records.append(
                {
                    "path_factor": path_factor,
                    "injected_delay_ns": float(injected_delay_s * 1.0e9),
                    "calibration_delay_ns": float(injected_delay_s * 1.0e9),
                    "residual_calibration_delay_ns": 0.0,
                    "expected_raw_bias_m": float(expected_raw_bias_m),
                    "ble_raw_distance_m": float(ble_result.distance_m),
                    "ble_corrected_error_m": float(ble_error_m),
                    "uwb_raw_distance_m": float(uwb_raw_m),
                    "uwb_corrected_error_m": float(uwb_error_m),
                }
            )

    uwb_bin_m = float(np.min(uwb_bins_m))
    max_ble_error_m = float(np.max(np.abs(ble_corrected_errors)))
    max_uwb_error_m = float(np.max(np.abs(uwb_corrected_errors)))
    max_mapping_error_m = float(np.max(np.abs(raw_bias_mapping_errors)))
    guard_result = RangingPipeline().run(
        [PropagationPath(0.5 / C0, 1.0 + 0.0j, "LOS")],
        ble_frequencies,
        uwb_frequencies,
        calibrate_rfic_group_delay=True,
        rfic_group_delay_calibration_s=20.0e-9,
    )
    wrong_calibration_guard_ok = bool(
        not guard_result.ble_robust.availability
        and guard_result.ble_robust.rejection_reason
        == "calibrated_distance_out_of_range"
        and np.isnan(guard_result.uwb_distance_m)
        and guard_result.uwb_first_path.status is FirstPathStatus.NO_PATH
        and guard_result.uwb_first_path.rejection_reason
        == "CALIBRATION_EXCEEDS_DETECTED_DELAY"
    )
    passed = (
        max_mapping_error_m < 1.0e-8
        and max_ble_error_m < 1.0e-8
        and max_uwb_error_m < uwb_bin_m
        and wrong_calibration_guard_ok
    )
    return (
        passed,
        {
            "true_distance_m": true_distance_m,
            "path_factors": [1.0, 2.0],
            "max_raw_bias_mapping_error_m": max_mapping_error_m,
            "max_ble_corrected_error_m": max_ble_error_m,
            "max_uwb_corrected_error_m": max_uwb_error_m,
            "uwb_distance_grid_bin_m_conservative": uwb_bin_m,
            "gross_wrong_calibration_guard": {
                "calibration_delay_ns": 20.0,
                "ble_status": guard_result.ble_robust.status,
                "ble_reason": guard_result.ble_robust.rejection_reason,
                "uwb_status": guard_result.uwb_first_path.status.value,
                "uwb_reason": guard_result.uwb_first_path.rejection_reason,
                "passed": wrong_calibration_guard_ok,
            },
            "records": records,
        },
        "精确映射向量使用显式合成校准值；一般多径回归另用独立参考扫频，错误校准会被状态机拒绝。",
    )


def check_phase_offset_invariance() -> tuple[bool, dict[str, Any], str]:
    frequencies = bluetooth_cs_frequencies_hz()
    true_distance = 9.0
    offsets = np.asarray([-2.5, -0.3, 0.0, 0.8, 2.9])
    errors: list[float] = []
    for offset in offsets:
        response = free_space_response(
            frequencies, true_distance, phase_offset_rad=float(offset)
        )
        errors.append(
            estimate_distance(frequencies, response).distance_m - true_distance
        )
    maximum_error = float(np.max(np.abs(errors)))
    return (
        maximum_error < 1.0e-8,
        {
            "phase_offset_min_rad": _float(np.min(offsets)),
            "phase_offset_max_rad": _float(np.max(offsets)),
            "max_abs_distance_error_m": maximum_error,
        },
        "",
    )


def check_stochastic_arrival_rates() -> tuple[bool, dict[str, Any], str]:
    config = SalehValenzuelaConfig(
        cluster_arrival_rate_hz=25.0e6,
        ray_arrival_rate_hz=180.0e6,
        cluster_count=4,
        rays_per_cluster=5,
    )
    cluster_intervals: list[float] = []
    ray_intervals: list[float] = []
    for seed in range(1000):
        paths = generate_saleh_valenzuela(
            5.0, config=config, seed=seed, include_los=False
        )
        by_label = {path.label: path.delay_s for path in paths}
        cluster_starts = [
            by_label[f"cluster_{cluster}_ray_0"]
            for cluster in range(config.cluster_count)
        ]
        cluster_intervals.extend(np.diff(cluster_starts))
        for cluster in range(config.cluster_count):
            ray_times = [
                by_label[f"cluster_{cluster}_ray_{ray}"]
                for ray in range(config.rays_per_cluster)
            ]
            ray_intervals.extend(np.diff(ray_times))

    cluster_mean = float(np.mean(cluster_intervals))
    ray_mean = float(np.mean(ray_intervals))
    cluster_expected = 1.0 / config.cluster_arrival_rate_hz
    ray_expected = 1.0 / config.ray_arrival_rate_hz
    cluster_relative_error = abs(cluster_mean / cluster_expected - 1.0)
    ray_relative_error = abs(ray_mean / ray_expected - 1.0)
    return (
        cluster_relative_error < 0.05 and ray_relative_error < 0.05,
        {
            "realizations": 1000,
            "cluster_expected_mean_ns": cluster_expected * 1.0e9,
            "cluster_empirical_mean_ns": cluster_mean * 1.0e9,
            "cluster_relative_error": cluster_relative_error,
            "ray_expected_mean_ns": ray_expected * 1.0e9,
            "ray_empirical_mean_ns": ray_mean * 1.0e9,
            "ray_relative_error": ray_relative_error,
        },
        "只验证内部参数生成器；其默认参数尚未追溯到 IEEE 802.15.4a 场景表。",
    )


def check_noise_scaling() -> tuple[bool, dict[str, Any], str]:
    frequencies = bluetooth_cs_frequencies_hz()
    true_distance = 8.0
    channel = frequency_response(
        frequencies, [PropagationPath(true_distance / C0, 1.0 + 0.0j)]
    )
    snr_values = [10.0, 20.0, 30.0, 40.0]
    rmse_by_snr: dict[str, float] = {}
    for snr_db in snr_values:
        errors: list[float] = []
        for seed in range(120):
            measured = apply_rfic_impairments(
                frequencies,
                channel,
                RFICImpairmentConfig(snr_db=snr_db),
                rng=np.random.default_rng(seed),
            )
            errors.append(
                estimate_distance(frequencies, measured).distance_m - true_distance
            )
        rmse_by_snr[f"{snr_db:.0f} dB"] = float(
            np.sqrt(np.mean(np.square(errors)))
        )
    rmse = list(rmse_by_snr.values())
    monotonic = all(left > right for left, right in zip(rmse, rmse[1:]))
    return (
        monotonic,
        {"distance_rmse_m_by_snr": rmse_by_snr},
        "验证噪声越低时估计误差单调改善，不代表具体接收机灵敏度。",
    )


def check_touchstone_v1_parser() -> tuple[bool, dict[str, Any], str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.s2p"
        path.write_text(
            "\n".join(
                [
                    "! synthetic acceptance vector",
                    "# GHz S DB R 50",
                    "2.4 -20 0 -3 10 -30 0 -10 0",
                    "2.5 -20 0 -6 20 -30 0 -10 0",
                ]
            ),
            encoding="utf-8",
        )
        target = np.asarray([2.4e9, 2.45e9, 2.5e9])
        frequencies, s21 = load_s2p_s21(path, target)
        expected_endpoints = np.asarray([10.0 ** (-3.0 / 20.0), 10.0 ** (-6.0 / 20.0)])
        endpoint_error = float(
            np.max(np.abs(np.abs(s21[[0, -1]]) - expected_endpoints))
        )
        frequency_match = bool(np.array_equal(frequencies, target))
    return (
        endpoint_error < 1.0e-12 and frequency_match,
        {
            "endpoint_magnitude_error": endpoint_error,
            "target_frequency_match": frequency_match,
        },
        "仅验证平台声明支持的单行 Touchstone v1；未验证完整 Touchstone 2.1。",
    )


def check_ble_robust_acceptance() -> tuple[bool, dict[str, Any], str]:
    """Gate BLE quality decisions separately from their numeric candidates."""

    frequencies = bluetooth_cs_frequencies_hz()
    clean_records: list[dict[str, Any]] = []
    clean_ok = True
    for path_factor, distances, maximum_distance in (
        (1.0, (0.5, 50.0, 149.0), 149.0),
        (2.0, (100.0,), 120.0),
    ):
        for distance_m in distances:
            response = np.exp(
                -1j
                * 2.0
                * np.pi
                * frequencies
                * (path_factor * distance_m / C0)
            )
            decision = estimate_distance_robust(
                frequencies,
                response,
                config=RobustPBRConfig(
                    path_factor=path_factor,
                    max_distance_m=maximum_distance,
                ),
            )
            error_m = (
                float(decision.distance_m - distance_m)
                if decision.distance_m is not None
                else None
            )
            record = {
                "path_factor": path_factor,
                "distance_m": distance_m,
                "status": decision.status,
                "error_m": error_m,
                "confidence": decision.confidence,
                "peak_ratio": decision.peak_ratio,
            }
            clean_records.append(record)
            clean_ok &= bool(
                decision.availability
                and error_m is not None
                and abs(error_m) < 1.0e-8
                and decision.confidence > 0.8
                and decision.peak_ratio is not None
                and decision.peak_ratio > 4.0
            )

    true_distance_m = 20.0
    base_response = np.exp(
        -1j * 2.0 * np.pi * frequencies * (true_distance_m / C0)
    )
    rng = np.random.default_rng(20260822)
    random_indices = np.sort(
        rng.choice(frequencies.size, size=frequencies.size // 2, replace=False)
    )
    notch_mask = np.ones(frequencies.size, dtype=bool)
    notch_mask[20:41] = False
    tone_map_records: list[dict[str, Any]] = []
    tone_map_ok = True
    for name, indices in (
        ("random_half", random_indices),
        ("continuous_notch", np.flatnonzero(notch_mask)),
        ("unsorted_reverse", np.arange(frequencies.size - 1, -1, -1)),
    ):
        decision = estimate_distance_robust(
            frequencies[indices],
            base_response[indices],
            config=RobustPBRConfig(max_distance_m=120.0),
        )
        error_m = (
            float(decision.distance_m - true_distance_m)
            if decision.distance_m is not None
            else None
        )
        tone_map_records.append(
            {
                "case": name,
                "tone_count": int(indices.size),
                "status": decision.status,
                "error_m": error_m,
                "grid_spacing_hz": decision.frequency_grid_spacing_hz,
            }
        )
        tone_map_ok &= bool(
            decision.availability
            and error_m is not None
            and abs(error_m) < 1.0e-8
        )

    sparse_frequencies = 2.404e9 + np.arange(16, dtype=float) * 4.0e6
    sparse_response = np.exp(
        -1j * 2.0 * np.pi * sparse_frequencies * (2.0 * 20.0 / C0)
    )
    sparse_decision = estimate_distance_robust(
        sparse_frequencies,
        sparse_response,
        config=RobustPBRConfig(path_factor=2.0, max_distance_m=100.0),
    )
    expected_sparse_range_m = C0 / (2.0 * 4.0e6)
    sparse_ok = bool(
        not sparse_decision.availability
        and sparse_decision.status == "rejected"
        and sparse_decision.frequency_grid_spacing_hz == 4.0e6
        and sparse_decision.unambiguous_distance_m is not None
        and abs(
            sparse_decision.unambiguous_distance_m
            - expected_sparse_range_m
        )
        < 1.0e-9
    )

    rtt_distance_m = 100.0
    rtt_response = np.exp(
        -1j * 2.0 * np.pi * frequencies * (2.0 * rtt_distance_m / C0)
    )
    accepted_errors: list[float] = []
    rejection_reasons: dict[str, int] = {}
    for seed in range(200):
        noisy = apply_rfic_impairments(
            frequencies,
            rtt_response,
            RFICImpairmentConfig(snr_db=5.0),
            rng=np.random.default_rng(seed),
        )
        decision = estimate_distance_robust(
            frequencies,
            noisy,
            config=RobustPBRConfig(path_factor=2.0, max_distance_m=120.0),
        )
        if decision.availability and decision.distance_m is not None:
            accepted_errors.append(decision.distance_m - rtt_distance_m)
        else:
            reason = decision.rejection_reason or "unspecified"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    accepted_absolute = np.abs(np.asarray(accepted_errors, dtype=float))
    availability = len(accepted_errors) / 200.0
    accepted_p95_m = (
        float(np.percentile(accepted_absolute, 95))
        if accepted_absolute.size
        else math.inf
    )
    catastrophic_count = int(np.sum(accepted_absolute > 1.0))
    noisy_ok = bool(
        availability >= 0.90
        and accepted_p95_m < 0.15
        and catastrophic_count == 0
        and len(accepted_errors) < 200
    )

    noise_available_count = 0
    for seed in range(512):
        noise_rng = np.random.default_rng(50_000 + seed)
        pure_noise = (
            noise_rng.normal(size=frequencies.size)
            + 1j * noise_rng.normal(size=frequencies.size)
        ) / np.sqrt(2.0)
        noise_available_count += int(
            estimate_distance_robust(
                frequencies,
                pure_noise,
                config=RobustPBRConfig(max_distance_m=120.0),
            ).availability
        )

    first_distance_m = 10.0
    second_distance_m = 22.0
    equal_two_path = np.exp(
        -1j * 2.0 * np.pi * frequencies * (first_distance_m / C0)
    ) + np.exp(
        -1j * 2.0 * np.pi * frequencies * (second_distance_m / C0)
    )
    ambiguous = estimate_distance_robust(
        frequencies,
        equal_two_path,
        config=RobustPBRConfig(max_distance_m=120.0),
    )
    invalid_vectors = (
        estimate_distance_robust(
            frequencies,
            np.full(frequencies.size, np.nan + 0j),
        ),
        estimate_distance_robust(
            np.r_[frequencies[:-1], frequencies[-2]],
            base_response,
        ),
        estimate_distance_robust(frequencies, np.zeros(frequencies.size)),
        estimate_distance_robust(frequencies[:4], base_response[:4]),
    )
    explicit_rejection_ok = bool(
        noise_available_count == 0
        and not ambiguous.availability
        and ambiguous.status == "rejected"
        and all(item.status == "invalid_input" for item in invalid_vectors)
    )

    cfo_hz = 650.0
    tone_interval_s = 8.0e-6
    tone_time_s = np.arange(frequencies.size, dtype=float) * tone_interval_s
    cfo_corrupted = base_response * np.exp(
        1j * 2.0 * np.pi * cfo_hz * tone_time_s
    )
    independently_compensated = cfo_corrupted * np.exp(
        -1j * 2.0 * np.pi * cfo_hz * tone_time_s
    )
    cfo_decision = estimate_distance_robust(
        frequencies,
        independently_compensated,
        config=RobustPBRConfig(max_distance_m=120.0),
    )
    cfo_ok = bool(
        cfo_decision.availability
        and cfo_decision.distance_m is not None
        and abs(cfo_decision.distance_m - true_distance_m) < 1.0e-8
    )

    passed = bool(
        clean_ok
        and tone_map_ok
        and sparse_ok
        and noisy_ok
        and explicit_rejection_ok
        and cfo_ok
    )
    return (
        passed,
        {
            "clean_vectors": clean_records,
            "tone_map_vectors": tone_map_records,
            "sparse_map": {
                "status": sparse_decision.status,
                "reason": sparse_decision.rejection_reason,
                "grid_spacing_hz": sparse_decision.frequency_grid_spacing_hz,
                "reported_unambiguous_distance_m": (
                    sparse_decision.unambiguous_distance_m
                ),
                "expected_unambiguous_distance_m": expected_sparse_range_m,
            },
            "rtt_100m_5db_200": {
                "availability": availability,
                "accepted_count": len(accepted_errors),
                "rejected_count": 200 - len(accepted_errors),
                "accepted_p95_abs_error_m": accepted_p95_m,
                "accepted_error_gt_1m_count": catastrophic_count,
                "rejection_reasons": rejection_reasons,
            },
            "pure_noise_512_available_count": noise_available_count,
            "equal_two_path_status": ambiguous.status,
            "equal_two_path_reason": ambiguous.rejection_reason,
            "invalid_vector_statuses": [item.status for item in invalid_vectors],
            "known_cfo_independent_compensation_status": cfo_decision.status,
        },
        (
            "通过门槛只证明洁净/缺频/噪声 LOS 的选择性可靠度。"
            "一般复杂多径和未补偿 CFO 仍不得宣称厘米级精度。"
        ),
    )


def check_uwb_cfar_acceptance() -> tuple[bool, dict[str, Any], str]:
    frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    bandwidth_hz = float(np.ptp(frequencies))
    ideal_distances_m = np.asarray(
        [0.5, 0.75, 1.0, 5.0, 8.123, 25.0, 74.25, 149.0]
    )
    ideal_errors: list[float] = []
    ideal_statuses: list[str] = []
    ideal_ok = True
    for distance_m in ideal_distances_m:
        response = np.exp(
            -1j * 2.0 * np.pi * frequencies * (distance_m / C0)
        )
        delays, cir = frequency_response_to_cir(
            frequencies, response, zero_pad_factor=8
        )
        decision = detect_first_path(delays, cir)
        error_m = C0 * decision.delay_s - distance_m
        ideal_errors.append(float(error_m))
        ideal_statuses.append(decision.status.value)
        ideal_ok &= bool(
            decision.available
            and abs(error_m) < 0.04
            and (
                distance_m < 0.75
                or decision.status == FirstPathStatus.PASS
            )
        )

    noise_delays = np.arange(4096, dtype=float) * 0.125e-9
    noise_false_ranges = 0
    noise_status_ok = True
    for seed in range(512):
        noise_rng = np.random.default_rng(60_000 + seed)
        noise = (
            noise_rng.normal(size=noise_delays.size)
            + 1j * noise_rng.normal(size=noise_delays.size)
        ) / np.sqrt(2.0)
        decision = detect_first_path(noise_delays, noise)
        noise_false_ranges += int(decision.available)
        noise_status_ok &= bool(
            decision.status == FirstPathStatus.NO_PATH
            and not decision.available
            and decision.confidence == 0.0
            and math.isnan(decision.delay_s)
        )
    zero_decision = detect_first_path(
        noise_delays, np.zeros(noise_delays.size, dtype=np.complex128)
    )
    spike = np.zeros(noise_delays.size, dtype=np.complex128)
    spike[1000] = 1.0 + 0.0j
    spike_decision = detect_first_path(noise_delays, spike)

    def noisy_los_stats(snr_db: float, seed_offset: int) -> dict[str, Any]:
        true_distance_m = 5.0
        clean = np.exp(
            -1j * 2.0 * np.pi * frequencies * (true_distance_m / C0)
        )
        errors: list[float] = []
        statuses: dict[str, int] = {
            item.value: 0 for item in FirstPathStatus
        }
        for seed in range(200):
            measured = apply_rfic_impairments(
                frequencies,
                clean,
                RFICImpairmentConfig(snr_db=snr_db),
                rng=np.random.default_rng(seed_offset + seed),
            )
            delays, cir = frequency_response_to_cir(
                frequencies, measured, zero_pad_factor=8
            )
            decision = detect_first_path(delays, cir)
            statuses[decision.status.value] += 1
            if decision.available:
                errors.append(C0 * decision.delay_s - true_distance_m)
        absolute = np.abs(np.asarray(errors, dtype=float))
        return {
            "snr_db": snr_db,
            "availability": len(errors) / 200.0,
            "p95_abs_error_m": (
                float(np.percentile(absolute, 95)) if absolute.size else None
            ),
            "error_gt_c_over_b_count": int(
                np.sum(absolute > C0 / bandwidth_hz)
            ),
            "statuses": statuses,
        }

    minus12 = noisy_los_stats(-12.0, 70_000)
    minus18 = noisy_los_stats(-18.0, 80_000)
    snr_ok = bool(
        minus12["availability"] >= 0.95
        and minus12["p95_abs_error_m"] is not None
        and minus12["p95_abs_error_m"] < 0.08
        and minus12["error_gt_c_over_b_count"] == 0
        and minus18["availability"] <= 0.05
    )

    weak_errors: list[float] = []
    weak_statuses: dict[str, int] = {item.value: 0 for item in FirstPathStatus}
    weak_confidence_consistent = True
    weak_distance_m = 5.0
    weak_clean = frequency_response(
        frequencies,
        [
            PropagationPath(weak_distance_m / C0, 0.10 + 0.0j, "weak_first"),
            PropagationPath(
                weak_distance_m / C0 + 10.0e-9,
                1.0 + 0.0j,
                "strong_late",
            ),
        ],
    )
    for seed in range(200):
        measured = apply_rfic_impairments(
            frequencies,
            weak_clean,
            RFICImpairmentConfig(snr_db=10.0),
            rng=np.random.default_rng(90_000 + seed),
        )
        delays, cir = frequency_response_to_cir(
            frequencies, measured, zero_pad_factor=8
        )
        decision = detect_first_path(delays, cir)
        weak_statuses[decision.status.value] += 1
        if decision.available:
            weak_errors.append(C0 * decision.delay_s - weak_distance_m)
        weak_confidence_consistent &= bool(
            decision.status == FirstPathStatus.LOW_CONFIDENCE
            and decision.rejection_reason
            and decision.confidence < 1.0
        )
    weak_absolute = np.abs(np.asarray(weak_errors, dtype=float))
    weak_p95_m = (
        float(np.percentile(weak_absolute, 95))
        if weak_absolute.size
        else math.inf
    )
    weak_ok = bool(
        len(weak_errors) == 200
        and weak_statuses[FirstPathStatus.LOW_CONFIDENCE.value] == 200
        and weak_p95_m < 0.04
        and weak_confidence_consistent
    )

    equal_response = frequency_response(
        frequencies,
        [
            PropagationPath(5.0 / C0, 1.0 + 0.0j, "first"),
            PropagationPath(5.0 / C0 + 2.0e-9, 1.0 + 0.0j, "equal_late"),
        ],
    )
    equal_delays, equal_cir = frequency_response_to_cir(
        frequencies, equal_response, zero_pad_factor=8
    )
    equal_decision = detect_first_path(equal_delays, equal_cir)
    equal_ok = bool(
        equal_decision.status == FirstPathStatus.LOW_CONFIDENCE
        and equal_decision.rejection_reason
        and "COMPETING_OR_OVERLAPPING_LOBE"
        in equal_decision.rejection_reason
        and equal_decision.confidence < 1.0
    )

    no_path_contract_ok = bool(
        noise_false_ranges == 0
        and noise_status_ok
        and zero_decision.status == FirstPathStatus.NO_PATH
        and spike_decision.status == FirstPathStatus.NO_PATH
        and webapp_module._finite_or_none(float("nan")) is None
        and json.dumps(
            {"distance_m": webapp_module._finite_or_none(float("nan"))},
            allow_nan=False,
        )
        == '{"distance_m": null}'
    )
    passed = bool(
        ideal_ok and no_path_contract_ok and snr_ok and weak_ok and equal_ok
    )
    return (
        passed,
        {
            "ideal": {
                "distances_m": ideal_distances_m.tolist(),
                "statuses": ideal_statuses,
                "max_abs_error_m": float(np.max(np.abs(ideal_errors))),
            },
            "pure_noise_512_false_ranges": noise_false_ranges,
            "zero_status": zero_decision.status.value,
            "isolated_spike_status": spike_decision.status.value,
            "single_los_minus12db_200": minus12,
            "single_los_minus18db_200": minus18,
            "weak_first_0p1_late_1p0_10db_200": {
                "availability": len(weak_errors) / 200.0,
                "p95_abs_error_m": weak_p95_m,
                "statuses": weak_statuses,
                "confidence_consistent": weak_confidence_consistent,
            },
            "equal_2ns_path": {
                "status": equal_decision.status.value,
                "reason": equal_decision.rejection_reason,
                "confidence": equal_decision.confidence,
                "ambiguity_ratio": equal_decision.ambiguity_ratio,
            },
            "no_path_json_value": None,
        },
        (
            "5 dB 弱首径/强晚径可物理不可辨识并可能锁晚径，明确列为边界而不纳入 PASS；"
            "本门槛的弱首径压力向量固定为 10 dB。"
        ),
    )


def check_uwb_weak_late_multiframe() -> tuple[bool, dict[str, Any], str]:
    """Gate the explicit four-frame remedy for the 5 dB late-lock case."""

    frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    true_distance_m = 5.0
    direct_delay_s = true_distance_m / C0
    frame_count = 4
    trial_count = 200
    accepted_errors: list[float] = []
    status_counts = {item.value: 0 for item in FirstPathStatus}
    single_frame_false_late_pass_count = 0
    multiframe_false_late_pass_count = 0
    late_lock_count = 0
    coherences: list[float] = []
    pipeline = RangingPipeline()
    ble_frequencies = bluetooth_cs_frequencies_hz()
    pipeline_default_errors: list[float] = []
    pipeline_default_late_pass_count = 0
    pipeline_default_status_counts = {
        item.value: 0 for item in FirstPathStatus
    }
    pipeline_default_frame_counts: set[int] = set()

    for trial in range(trial_count):
        late_phase_rad = 2.0 * np.pi * (trial % 16) / 16.0
        paths = [
            PropagationPath(
                direct_delay_s,
                0.10 + 0.0j,
                "weak_first",
            ),
            PropagationPath(
                direct_delay_s + 10.0e-9,
                np.exp(1j * late_phase_rad),
                "strong_late",
            ),
        ]
        clean_response = frequency_response(frequencies, paths)
        frames: list[np.ndarray] = []
        for frame_index in range(frame_count):
            measured = apply_rfic_impairments(
                frequencies,
                clean_response,
                RFICImpairmentConfig(snr_db=5.0),
                rng=np.random.default_rng(
                    100_000 + trial * frame_count + frame_index
                ),
            )
            # A receiver restart may add a common phase. The combiner is
            # allowed to remove only this scalar phase, never a delay slope.
            common_phase_rad = 0.83 * frame_index + 0.17 * trial
            frames.append(measured * np.exp(1j * common_phase_rad))

        single_delays, single_cir = frequency_response_to_cir(
            frequencies,
            frames[0],
            zero_pad_factor=8,
        )
        single_decision = detect_first_path(
            single_delays,
            single_cir,
            early_path_enhancement=True,
        )
        if single_decision.available:
            single_error_m = (
                C0 * single_decision.delay_s - true_distance_m
            )
            single_frame_false_late_pass_count += int(
                abs(single_error_m) > 1.0
                and single_decision.status is FirstPathStatus.PASS
            )

        delays, cir, coherence = frequency_response_frames_to_cir(
            frequencies,
            np.asarray(frames),
            combining="phase_aligned",
            zero_pad_factor=8,
        )
        decision = detect_first_path(delays, cir)
        coherences.append(coherence)
        status_counts[decision.status.value] += 1
        if not decision.available:
            continue
        error_m = C0 * decision.delay_s - true_distance_m
        accepted_errors.append(error_m)
        is_late_lock = abs(error_m) > 1.0
        late_lock_count += int(is_late_lock)
        multiframe_false_late_pass_count += int(
            is_late_lock and decision.status is FirstPathStatus.PASS
        )

        pipeline_result = pipeline.run(
            paths,
            ble_frequencies,
            frequencies,
            rfic_config=RFICImpairmentConfig(snr_db=5.0),
            seed=trial,
        )
        pipeline_default_frame_counts.add(pipeline_result.uwb_frame_count)
        pipeline_status = pipeline_result.uwb_first_path.status
        pipeline_default_status_counts[pipeline_status.value] += 1
        if pipeline_result.uwb_first_path.available:
            pipeline_error_m = (
                pipeline_result.uwb_distance_m - true_distance_m
            )
            pipeline_default_errors.append(pipeline_error_m)
            pipeline_default_late_pass_count += int(
                abs(pipeline_error_m) > 1.0
                and pipeline_status is FirstPathStatus.PASS
            )

    absolute_errors = np.abs(np.asarray(accepted_errors, dtype=float))
    availability = len(accepted_errors) / trial_count
    correct_first_path_rate = float(
        np.mean(absolute_errors <= 0.10)
    ) if absolute_errors.size else 0.0
    p95_abs_error_m = (
        float(np.percentile(absolute_errors, 95))
        if absolute_errors.size
        else math.inf
    )
    median_coherence = float(np.median(coherences))
    pipeline_default_absolute_errors = np.abs(
        np.asarray(pipeline_default_errors, dtype=float)
    )
    pipeline_default_correct_rate = (
        float(np.mean(pipeline_default_absolute_errors <= 0.10))
        if pipeline_default_absolute_errors.size
        else 0.0
    )
    pipeline_default_p95_m = (
        float(np.percentile(pipeline_default_absolute_errors, 95))
        if pipeline_default_absolute_errors.size
        else math.inf
    )

    pure_noise_false_ranges = 0
    for trial in range(128):
        noise_frames = []
        for frame_index in range(frame_count):
            noise_rng = np.random.default_rng(
                200_000 + trial * frame_count + frame_index
            )
            noise_frames.append(
                (
                    noise_rng.normal(size=frequencies.size)
                    + 1j * noise_rng.normal(size=frequencies.size)
                )
                / np.sqrt(2.0)
            )
        noise_delays, noise_cir, _ = frequency_response_frames_to_cir(
            frequencies,
            np.asarray(noise_frames),
            combining="phase_aligned",
            zero_pad_factor=8,
        )
        pure_noise_false_ranges += int(
            detect_first_path(noise_delays, noise_cir).available
        )

    passed = bool(
        availability >= 0.95
        and correct_first_path_rate >= 0.95
        and p95_abs_error_m <= 0.10
        and late_lock_count / trial_count <= 0.01
        and multiframe_false_late_pass_count == 0
        and single_frame_false_late_pass_count == 0
        and median_coherence >= 0.75
        and pure_noise_false_ranges == 0
        and pipeline_default_frame_counts == {4}
        and len(pipeline_default_errors) / trial_count >= 0.95
        and pipeline_default_correct_rate >= 0.95
        and pipeline_default_p95_m <= 0.10
        and pipeline_default_late_pass_count == 0
    )
    return (
        passed,
        {
            "snr_db": 5.0,
            "first_to_late_amplitude_ratio": 0.10,
            "late_excess_delay_ns": 10.0,
            "frame_count": frame_count,
            "trial_count": trial_count,
            "availability": availability,
            "correct_first_path_rate_within_0p10m": (
                correct_first_path_rate
            ),
            "p95_abs_error_m": p95_abs_error_m,
            "late_lock_gt_1m_count": late_lock_count,
            "single_frame_false_late_pass_count": (
                single_frame_false_late_pass_count
            ),
            "multiframe_false_late_pass_count": (
                multiframe_false_late_pass_count
            ),
            "status_counts": status_counts,
            "median_frame_coherence": median_coherence,
            "four_frame_pure_noise_128_false_ranges": (
                pure_noise_false_ranges
            ),
            "pipeline_default_four_frame_200": {
                "frame_counts": sorted(pipeline_default_frame_counts),
                "availability": len(pipeline_default_errors) / trial_count,
                "correct_first_path_rate_within_0p10m": (
                    pipeline_default_correct_rate
                ),
                "p95_abs_error_m": pipeline_default_p95_m,
                "late_path_pass_count": pipeline_default_late_pass_count,
                "status_counts": pipeline_default_status_counts,
            },
        },
        (
            "四帧处理假设重复 CFR 已完成公共相位对齐且信道准静态；"
            "它不是零填充增益，也不能外推到运动或未补偿 CFO 场景。"
        ),
    )


def check_frequency_dependent_calibration() -> tuple[bool, dict[str, Any], str]:
    """Use an off-grid calibration sweep, never the DUT response quotient."""

    base_delay_s = 2.0e-9
    delay_slope_s_per_hz = 0.40e-18
    phase_offset_rad = -0.31

    def dut_transfer(
        frequencies_hz: np.ndarray, reference_hz: float
    ) -> np.ndarray:
        delta = frequencies_hz - reference_hz
        normalised = delta / max(float(np.ptp(frequencies_hz)), 1.0)
        log_magnitude = 0.06 * normalised + 0.015 * normalised**2
        phase = phase_offset_rad - 2.0 * np.pi * (
            base_delay_s * delta
            + 0.5 * delay_slope_s_per_hz * delta**2
        )
        return np.exp(log_magnitude + 1j * phase)

    def independent_fixture_sweep(
        frequencies_hz: np.ndarray, reference_hz: float, span_hz: float
    ) -> np.ndarray:
        # Algebraically independent fixture generator on a different grid.
        x = frequencies_hz - reference_hz
        amplitude_db = (20.0 / np.log(10.0)) * (
            0.06 * (x / span_hz) + 0.015 * (x / span_hz) ** 2
        )
        integrated_group_delay = (
            base_delay_s * x + delay_slope_s_per_hz * x * x / 2.0
        )
        fixture_phase = (
            phase_offset_rad
            - 2.0 * np.pi * integrated_group_delay
            + 0.07
        )
        return 1.03 * 10.0 ** (amplitude_db / 20.0) * np.exp(
            1j * fixture_phase
        )

    records: list[dict[str, Any]] = []
    all_accuracy_ok = True
    for radio, target_frequencies, reference_hz in (
        ("BLE", bluetooth_cs_frequencies_hz(), 2.441e9),
        ("UWB", np.linspace(3.5e9, 4.5e9, 1001), 4.0e9),
    ):
        calibration_step_hz = 0.5e6
        calibration_frequencies = np.arange(
            target_frequencies[0] - 0.25e6,
            target_frequencies[-1] + 0.25e6 + calibration_step_hz / 2.0,
            calibration_step_hz,
        )
        span_hz = float(np.ptp(target_frequencies))
        measured_fixture = independent_fixture_sweep(
            calibration_frequencies, reference_hz, span_hz
        )
        profile = build_calibration_profile(
            calibration_frequencies,
            measured_fixture,
            reference_plane="independent fixture plane to estimator input",
            source=f"acceptance off-grid {radio} synthetic fixture sweep",
            metadata={
                "dut_seed": 2718,
                "calibration_seed": 3141,
                "shared_response_samples": False,
            },
            maximum_interpolation_gap_hz=0.75e6,
            maximum_expected_group_delay_s=4.0e-9,
        )
        for path_factor in (1.0, 2.0):
            true_distance_m = 7.0
            channel = np.exp(
                -1j
                * 2.0
                * np.pi
                * target_frequencies
                * (path_factor * true_distance_m / C0)
            )
            raw = channel * dut_transfer(target_frequencies, reference_hz)
            calibrated = apply_calibration(
                target_frequencies, raw, profile
            )
            if radio == "BLE":
                maximum_distance_m = 100.0 if path_factor == 2.0 else 149.0
                raw_distance_m = estimate_distance(
                    target_frequencies,
                    raw,
                    path_factor=path_factor,
                    max_distance_m=maximum_distance_m,
                ).distance_m
                calibrated_distance_m = estimate_distance(
                    target_frequencies,
                    calibrated.calibrated_response,
                    path_factor=path_factor,
                    max_distance_m=maximum_distance_m,
                ).distance_m
                gate_m = 1.0e-8
            else:
                raw_delay_s, raw_cir = frequency_response_to_cir(
                    target_frequencies, raw, zero_pad_factor=8
                )
                raw_distance_m, _ = estimate_distance_from_cir(
                    raw_delay_s, raw_cir, path_factor=path_factor
                )
                cal_delay_s, cal_cir = frequency_response_to_cir(
                    target_frequencies,
                    calibrated.calibrated_response,
                    zero_pad_factor=8,
                )
                calibrated_distance_m, _ = estimate_distance_from_cir(
                    cal_delay_s, cal_cir, path_factor=path_factor
                )
                gate_m = (
                    C0
                    * float(np.median(np.diff(cal_delay_s)))
                    / path_factor
                )
            raw_error_m = raw_distance_m - true_distance_m
            calibrated_error_m = calibrated_distance_m - true_distance_m
            improvement = abs(raw_error_m) / max(
                abs(calibrated_error_m), np.finfo(float).tiny
            )
            case_ok = bool(
                abs(raw_error_m) > 0.20
                and abs(calibrated_error_m) < gate_m
                and improvement >= 20.0
                and calibrated.profile_id == profile.profile_id
                and not np.shares_memory(raw, measured_fixture)
            )
            all_accuracy_ok &= case_ok
            records.append(
                {
                    "radio": radio,
                    "path_factor": path_factor,
                    "profile_id": profile.profile_id,
                    "target_grid_is_off_calibration_grid": bool(
                        not np.all(
                            np.isin(target_frequencies, calibration_frequencies)
                        )
                    ),
                    "raw_error_m": float(raw_error_m),
                    "calibrated_error_m": float(calibrated_error_m),
                    "gate_m": float(gate_m),
                    "improvement_factor": float(improvement),
                    "profile_source": profile.source,
                    "reference_plane": profile.reference_plane,
                    "raw_calibrated_evidence": calibrated.evidence(),
                }
            )

    guard_results: dict[str, bool] = {}

    def expect_value_error(name: str, operation: Callable[[], object]) -> None:
        try:
            operation()
        except ValueError:
            guard_results[name] = True
        else:
            guard_results[name] = False

    base_frequencies = np.asarray([1.0e9, 1.5e9, 2.0e9])
    base_transfer = np.ones(3, dtype=np.complex128)
    base_profile = build_calibration_profile(
        base_frequencies,
        base_transfer,
        reference_plane="guard plane",
        source="guard vector",
        maximum_interpolation_gap_hz=0.6e9,
    )
    expect_value_error(
        "extrapolation",
        lambda: base_profile.correction_at(
            np.asarray([0.9e9, 1.2e9, 1.8e9])
        ),
    )
    expect_value_error(
        "large_unmeasured_gap",
        lambda: build_calibration_profile(
            np.asarray([1.0e9, 1.1e9, 2.0e9]),
            np.ones(3, dtype=np.complex128),
            reference_plane="gap plane",
            source="gap vector",
            maximum_interpolation_gap_hz=0.2e9,
        ),
    )
    expect_value_error(
        "phase_unwrap_ambiguity",
        lambda: build_calibration_profile(
            np.asarray([1.0e9, 1.01e9, 1.02e9]),
            np.ones(3, dtype=np.complex128),
            reference_plane="phase plane",
            source="phase vector",
            maximum_expected_group_delay_s=60.0e-9,
        ),
    )
    expect_value_error(
        "deep_null",
        lambda: build_calibration_profile(
            np.asarray([1.0e9, 1.1e9, 1.2e9]),
            np.asarray([1.0 + 0j, 0.0 + 0j, 1.0 + 0j]),
            reference_plane="null plane",
            source="null vector",
        ),
    )
    expect_value_error(
        "excessive_inverse_gain",
        lambda: build_calibration_profile(
            np.asarray([1.0e9, 1.1e9, 1.2e9]),
            np.asarray([1.0 + 0j, 1.0e-4 + 0j, 1.0 + 0j]),
            reference_plane="gain plane",
            source="gain vector",
            maximum_correction_magnitude=1.0e3,
        ),
    )
    first_application = apply_calibration(
        base_frequencies,
        np.ones(3, dtype=np.complex128),
        base_profile,
    )
    expect_value_error(
        "repeat_application",
        lambda: apply_calibration(
            base_frequencies, first_application, base_profile
        ),
    )
    guards_ok = all(guard_results.values()) and len(guard_results) == 6
    return (
        bool(all_accuracy_ok and guards_ok),
        {
            "calibration_input_independent_from_dut_response": True,
            "off_grid_records": records,
            "safety_guards": guard_results,
        },
        (
            "Profile 来自独立函数和异频网格的显式扫频证据；这仍是合成校准，"
            "不替代真实 VNA/环回文件。"
        ),
    )


def check_web_ui_contract() -> tuple[bool, dict[str, Any], str]:
    """Exercise the local UI over HTTP and verify stale/result contracts."""

    class QuietRangingRequestHandler(webapp_module.RangingRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    server = webapp_module.ThreadingHTTPServer(
        ("127.0.0.1", 0), QuietRangingRequestHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    def request(
        path: str,
        *,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> tuple[int, bytes]:
        headers = {"Content-Type": content_type} if body is not None else {}
        call = urllib.request.Request(
            base_url + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(call, timeout=15.0) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()

    try:
        status_codes: dict[str, int] = {}
        status_codes["health"] = request("/api/health")[0]
        status_codes["index"] = request("/")[0]
        status_codes["javascript"] = request("/app.js")[0]
        status_codes["stylesheet"] = request("/styles.css")[0]
        status_codes["not_found"] = request("/missing")[0]
        valid_status, valid_body = request(
            "/api/run",
            body=json.dumps(
                {
                    "reflection_count": 0,
                    "snr_db": 80,
                    "phase_noise_deg": 0,
                    "cfo_hz": 0,
                    "calibration_mode": "frequency_profile",
                }
            ).encode("utf-8"),
        )
        status_codes["valid_run"] = valid_status
        valid_payload = json.loads(valid_body)
        bad_json_status, _ = request("/api/run", body=b"{")
        invalid_field_status, _ = request(
            "/api/run", body=b'{"distance_m":0}'
        )
        oversized_connection = HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=15.0
        )
        try:
            oversized_connection.putrequest("POST", "/api/run")
            oversized_connection.putheader(
                "Content-Length",
                str(webapp_module.MAX_REQUEST_BYTES + 1),
            )
            oversized_connection.endheaders()
            oversized_response = oversized_connection.getresponse()
            oversized_status = int(oversized_response.status)
            oversized_response.read()
        finally:
            oversized_connection.close()
        status_codes.update(
            {
                "bad_json": bad_json_status,
                "invalid_field": invalid_field_status,
                "oversized": oversized_status,
            }
        )
        rejected_status, rejected_body = request(
            "/api/run",
            body=json.dumps(
                {
                    "distance_m": 20,
                    "reflection_count": 5,
                    "reflection_gain": 1.5,
                    "reflection_delay_ns": 13.5,
                    "reflection_phase_rad": 3.14,
                    "snr_db": 10,
                    "seed": 2,
                    "calibration_mode": "frequency_profile",
                }
            ).encode("utf-8"),
        )
        rejected_payload = json.loads(rejected_body)

        example_common = {
            "base_group_delay_ns": 0,
            "group_delay_ripple_ns": 0,
            "amplitude_ripple_db": 0,
            "calibration_mode": "none",
            "calibration_delay_ns": 0,
            "calibration_ripple_ns": 0,
            "calibration_amplitude_ripple_db": 0,
            "reflection_phase_rad": 0,
            "cfo_hz": 0,
        }
        example_requests = {
            "uwb_late_single": {
                **example_common,
                "distance_m": 8,
                "direct_path_gain": 0.1,
                "reflection_count": 1,
                "reflection_gain": 1,
                "reflection_delay_ns": 10,
                "snr_db": 5,
                "phase_noise_deg": 0,
                "uwb_frame_count": 1,
                "seed": 0,
            },
            "uwb_late_four": {
                **example_common,
                "distance_m": 8,
                "direct_path_gain": 0.1,
                "reflection_count": 1,
                "reflection_gain": 1,
                "reflection_delay_ns": 10,
                "snr_db": 5,
                "phase_noise_deg": 0,
                "uwb_frame_count": 4,
                "seed": 0,
            },
            "low_coherence": {
                **example_common,
                "distance_m": 8,
                "direct_path_gain": 1,
                "reflection_count": 0,
                "reflection_gain": 0,
                "reflection_delay_ns": 10,
                "snr_db": -10,
                "phase_noise_deg": 0,
                "uwb_frame_count": 4,
                "seed": 41,
            },
        }
        example_status_codes: dict[str, int] = {}
        example_payloads: dict[str, dict[str, Any]] = {}
        for example_name, example_request in example_requests.items():
            example_status, example_body = request(
                "/api/run",
                body=json.dumps(example_request).encode("utf-8"),
            )
            example_status_codes[example_name] = example_status
            example_payloads[example_name] = json.loads(example_body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    strict_json_ok = bool(
        valid_status == 200
        and b"NaN" not in valid_body
        and b"Infinity" not in valid_body
        and json.dumps(valid_payload, allow_nan=False)
    )
    ble = valid_payload["summary"]["ble"]
    uwb = valid_payload["summary"]["uwb"]
    calibration = valid_payload["summary"]["rfic_calibration"]
    result_contract_ok = bool(
        {"available", "status", "raw_distance_m", "candidate_distance_m", "calibrated_distance_m"}
        <= set(ble)
        and {"available", "status", "raw_distance_m", "candidate_distance_m", "calibrated_distance_m"}
        <= set(uwb)
        and calibration["mode"] == "frequency_profile"
        and calibration["ble_profile"]["profile_id"]
        and calibration["uwb_profile"]["profile_id"]
        and rejected_status == 200
        and rejected_payload["summary"]["ble"]["status"] == "REJECTED"
        and not rejected_payload["summary"]["ble"]["available"]
        and rejected_payload["summary"]["ble"]["candidate_distance_m"]
        is not None
        and rejected_payload["summary"]["ble"]["calibrated_distance_m"]
        is None
        and len(valid_payload["paths"])
        == valid_payload["scenario"]["path_count"]
        and isinstance(valid_payload["ble_estimated_paths"], list)
    )

    single_example = example_payloads["uwb_late_single"]
    four_example = example_payloads["uwb_late_four"]
    coherence_example = example_payloads["low_coherence"]
    visual_examples_ok = bool(
        all(status == 200 for status in example_status_codes.values())
        and single_example["scenario"]["direct_path_gain"] == 0.1
        and single_example["summary"]["uwb"]["status"] == "PASS"
        and abs(single_example["summary"]["uwb"]["error_m"]) > 2.5
        and four_example["summary"]["uwb"]["status"]
        == "LOW_CONFIDENCE"
        and abs(four_example["summary"]["uwb"]["error_m"]) < 0.1
        and coherence_example["scenario"]["uwb_frame_coherence"] < 0.75
        and coherence_example["summary"]["uwb"]["rejection_reason"]
        == "LOW_FRAME_COHERENCE"
    )

    html = (webapp_module.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (webapp_module.STATIC_ROOT / "app.js").read_text(
        encoding="utf-8"
    )
    ids = re.findall(r'\bid="([^"]+)"', html)
    js_id_references = set(
        re.findall(r'getElementById\("([^"]+)"\)', javascript)
    )
    example_names = set(re.findall(r'data-example="([^"]+)"', html))
    required_visual_ids = {
        "exampleGrid",
        "exampleName",
        "exampleExpected",
        "exampleObserved",
        "distanceChart",
        "pathChart",
        "bleConfidence",
    }
    dom_contract_ok = bool(
        len(ids) == len(set(ids))
        and js_id_references <= set(ids)
        and required_visual_ids <= set(ids)
        and example_names
        == {
            "los",
            "ble_multipath",
            "uwb_late_single",
            "uwb_late_four",
            "low_coherence",
            "bad_calibration",
        }
        and all(
            asset in html
            for asset in ('href="/styles.css"', 'src="/app.js"')
        )
    )

    acceptance_payload = {
        "schema_version": webapp_module.ACCEPTANCE_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "overall_status": "PARTIAL",
        "software_gate_status": "PASS",
        "standards_status": "PARTIAL",
        "status_counts": {"PASS": 1},
        "checks": [],
    }
    original_acceptance_path = webapp_module.ACCEPTANCE_JSON
    with tempfile.TemporaryDirectory() as directory:
        temporary_acceptance = Path(directory) / "acceptance.json"
        temporary_acceptance.write_text(
            json.dumps(acceptance_payload), encoding="utf-8"
        )
        try:
            webapp_module.ACCEPTANCE_JSON = temporary_acceptance
            latest_mtime, _ = webapp_module._latest_acceptance_input_mtime()
            stale_mtime = max(315_532_800.0, latest_mtime - 60.0)
            os.utime(
                temporary_acceptance, (stale_mtime, stale_mtime)
            )
            stale_summary = webapp_module.load_acceptance_summary()
            future_mtime = max(
                latest_mtime + 60.0,
                datetime.now().timestamp() + 60.0,
            )
            os.utime(
                temporary_acceptance, (future_mtime, future_mtime)
            )
            fresh_summary = webapp_module.load_acceptance_summary()
        finally:
            webapp_module.ACCEPTANCE_JSON = original_acceptance_path
    freshness_ok = bool(
        stale_summary["available"]
        and not stale_summary["fresh"]
        and stale_summary["software_gate_status"] == "STALE"
        and fresh_summary["fresh"]
        and fresh_summary["schema_valid"]
        and fresh_summary["software_gate_status"] == "PASS"
    )
    status_codes_ok = status_codes == {
        "health": 200,
        "index": 200,
        "javascript": 200,
        "stylesheet": 200,
        "not_found": 404,
        "valid_run": 200,
        "bad_json": 400,
        "invalid_field": 422,
        "oversized": 413,
    }
    no_path_null_ok = bool(
        json.loads(
            json.dumps(
                {
                    "distance_m": webapp_module._finite_or_none(
                        float("nan")
                    )
                },
                allow_nan=False,
            )
        )["distance_m"]
        is None
    )
    passed = bool(
        status_codes_ok
        and strict_json_ok
        and result_contract_ok
        and visual_examples_ok
        and dom_contract_ok
        and freshness_ok
        and no_path_null_ok
    )
    return (
        passed,
        {
            "http_status_codes": status_codes,
            "strict_json_without_nan": strict_json_ok,
            "dynamic_rejected_ble_status": (
                rejected_payload["summary"]["ble"]["status"]
            ),
            "raw_candidate_calibrated_contract": result_contract_ok,
            "guided_visual_examples": {
                "status_codes": example_status_codes,
                "single_frame_error_m": single_example["summary"]["uwb"][
                    "error_m"
                ],
                "four_frame_error_m": four_example["summary"]["uwb"][
                    "error_m"
                ],
                "low_coherence": coherence_example["scenario"][
                    "uwb_frame_coherence"
                ],
                "passed": visual_examples_ok,
            },
            "frequency_profile_id": calibration["ble_profile"]["profile_id"],
            "unique_html_ids": len(ids) == len(set(ids)),
            "javascript_id_references_missing": sorted(
                js_id_references - set(ids)
            ),
            "guided_example_names": sorted(example_names),
            "stale_result_detected": not stale_summary["fresh"],
            "fresh_result_detected": fresh_summary["fresh"],
            "no_path_json_value": None,
        },
        "本地无外部依赖 UI；真实浏览器视觉检查另作为交付证据，不替代 API/DOM 门槛。",
    )


def _error_metrics(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=float)
    finite_mask = np.isfinite(errors)
    finite_errors = errors[finite_mask]
    metrics: dict[str, Any] = {
        "count": int(errors.size),
        "finite_count": int(finite_errors.size),
        "availability": float(np.mean(finite_mask)) if errors.size else 0.0,
    }
    if finite_errors.size == 0:
        metrics.update(
            {
                "bias_m": None,
                "mae_m": None,
                "rmse_m": None,
                "p50_abs_error_m": None,
                "p90_abs_error_m": None,
                "p95_abs_error_m": None,
            }
        )
        return metrics
    absolute = np.abs(finite_errors)
    metrics.update(
        {
            "bias_m": float(np.mean(finite_errors)),
            "mae_m": float(np.mean(absolute)),
            "rmse_m": float(np.sqrt(np.mean(finite_errors**2))),
            "p50_abs_error_m": float(np.percentile(absolute, 50)),
            "p90_abs_error_m": float(np.percentile(absolute, 90)),
            "p95_abs_error_m": float(np.percentile(absolute, 95)),
        }
    )
    return metrics


def _measure_scalar_group_delay_from_reference_sweep(
    frequencies_hz: np.ndarray,
    dut_config: RFICImpairmentConfig,
    rng: np.random.Generator,
    *,
    reference_plane_offset_s: float = 0.0,
) -> tuple[float, float]:
    """Estimate scalar RFIC delay from an independent known-through sweep.

    The estimator receives only the simulated reference response.  Injected
    DUT delay is used by the simulator to create that response, but is never
    copied into the calibration value supplied to the ranging pipeline.
    Dynamic CFO is intentionally excluded because it must be compensated per
    acquisition rather than hidden inside a static hardware calibration.
    """

    calibration_config = RFICImpairmentConfig(
        phase_offset_rad=dut_config.phase_offset_rad,
        base_group_delay_s=(
            dut_config.base_group_delay_s + reference_plane_offset_s
        ),
        group_delay_ripple_s=dut_config.group_delay_ripple_s,
        group_delay_ripple_period_hz=dut_config.group_delay_ripple_period_hz,
        amplitude_ripple_db=dut_config.amplitude_ripple_db,
        iq_gain_mismatch_db=dut_config.iq_gain_mismatch_db,
        iq_phase_mismatch_deg=dut_config.iq_phase_mismatch_deg,
        phase_noise_std_deg=0.2,
        cfo_hz=0.0,
        tone_interval_s=0.0,
        snr_db=40.0,
    )
    reference_response = apply_rfic_impairments(
        frequencies_hz,
        np.ones(frequencies_hz.size, dtype=np.complex128),
        calibration_config,
        rng=rng,
    )
    weights = np.maximum(
        np.square(np.abs(reference_response)), np.finfo(float).tiny
    )
    centered_frequency = frequencies_hz - np.average(
        frequencies_hz, weights=weights
    )
    phase = np.unwrap(np.angle(reference_response))
    centered_phase = phase - np.average(phase, weights=weights)
    denominator = float(np.sum(weights * centered_frequency**2))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("reference sweep has insufficient frequency span")
    slope = float(
        np.sum(weights * centered_frequency * centered_phase) / denominator
    )
    fitted_phase = slope * centered_frequency + np.average(
        phase, weights=weights
    )
    residual_rms_rad = float(
        np.sqrt(np.average((phase - fitted_phase) ** 2, weights=weights))
    )
    return -slope / (2.0 * np.pi), residual_rms_rad


def run_simulation_monte_carlo(
    output_dir: Path,
    realization_count: int = 200,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260821)
    calibration_rng = np.random.default_rng(20260822)
    ble_frequencies = bluetooth_cs_frequencies_hz()
    uwb_frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    pipeline = RangingPipeline()
    rows: list[dict[str, Any]] = []
    for index in range(realization_count):
        distance = float(rng.uniform(1.0, 20.0))
        direct_delay = distance / C0
        paths = [PropagationPath(direct_delay, 1.0 + 0.0j, "LOS")]
        reflection_count = int(rng.integers(1, 5))
        for reflection_index in range(reflection_count):
            excess_delay = float(rng.uniform(3.0e-9, 30.0e-9))
            magnitude = float(rng.uniform(0.1, 0.75))
            phase = float(rng.uniform(-np.pi, np.pi))
            paths.append(
                PropagationPath(
                    direct_delay + excess_delay,
                    magnitude * np.exp(1j * phase),
                    f"reflection_{reflection_index}",
                )
            )
        snr_db = float(rng.uniform(15.0, 40.0))
        injected_group_delay_s = float(rng.uniform(0.0, 0.5e-9))
        calibration_reference_plane_offset_s = float(
            rng.normal(0.0, 0.02e-9)
        )
        config = RFICImpairmentConfig(
                base_group_delay_s=injected_group_delay_s,
                group_delay_ripple_s=float(rng.uniform(0.0, 0.12e-9)),
                amplitude_ripple_db=float(rng.uniform(0.0, 0.8)),
                phase_noise_std_deg=float(rng.uniform(0.2, 2.0)),
                cfo_hz=float(rng.uniform(-50.0, 50.0)),
                tone_interval_s=8.0e-6,
                snr_db=snr_db,
        )
        (
            calibration_group_delay_s,
            calibration_reference_residual_rms_rad,
        ) = _measure_scalar_group_delay_from_reference_sweep(
            ble_frequencies,
            config,
            calibration_rng,
            reference_plane_offset_s=calibration_reference_plane_offset_s,
        )
        calibration_measurement_error_s = (
            calibration_group_delay_s - injected_group_delay_s
        )
        failure = ""
        try:
            result = pipeline.run(
                paths,
                ble_frequencies,
                uwb_frequencies,
                rfic_config=config,
                seed=index,
                calibrate_rfic_group_delay=True,
                rfic_group_delay_calibration_s=calibration_group_delay_s,
            )
            ble_raw_distance_m = result.ble_raw_distance_m
            ble_calibrated_distance_m = result.ble_calibrated_distance_m
            uwb_raw_distance_m = result.uwb_raw_distance_m
            uwb_calibrated_distance_m = result.uwb_distance_m
            nlos_probability = result.nlos_probability
            ble_robust_available = bool(
                result.ble_robust is not None
                and result.ble_robust.availability
            )
            ble_robust_status = (
                result.ble_robust.status
                if result.ble_robust is not None
                else "missing"
            )
            ble_robust_reason = (
                result.ble_robust.rejection_reason
                if result.ble_robust is not None
                else "missing_quality_decision"
            )
            ble_robust_confidence = (
                result.ble_robust.confidence
                if result.ble_robust is not None
                else 0.0
            )
            ble_robust_estimator_mode = (
                result.ble_robust.estimator_mode
                if result.ble_robust is not None
                else "missing"
            )
            ble_robust_model_order = (
                result.ble_robust.model_order
                if result.ble_robust is not None
                else 0
            )
            ble_robust_reconstruction_residual = (
                result.ble_robust.reconstruction_residual
                if result.ble_robust is not None
                else None
            )
            ble_robust_candidate_distance_m = (
                result.ble_robust_calibrated_distance_m
                if result.ble_robust is not None
                else None
            )
            ble_accepted_distance_m = (
                float(ble_robust_candidate_distance_m)
                if ble_robust_available
                and ble_robust_candidate_distance_m is not None
                else float("nan")
            )
            uwb_status = result.uwb_first_path.status.value
        except Exception as exc:  # availability is part of the regression gate
            ble_raw_distance_m = float("nan")
            ble_calibrated_distance_m = float("nan")
            uwb_raw_distance_m = float("nan")
            uwb_calibrated_distance_m = float("nan")
            nlos_probability = float("nan")
            ble_robust_available = False
            ble_robust_status = "exception"
            ble_robust_reason = "pipeline_exception"
            ble_robust_confidence = 0.0
            ble_robust_estimator_mode = "exception"
            ble_robust_model_order = 0
            ble_robust_reconstruction_residual = None
            ble_robust_candidate_distance_m = None
            ble_accepted_distance_m = float("nan")
            uwb_status = "exception"
            failure = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "index": index,
                "true_distance_m": distance,
                "path_count": len(paths),
                "snr_db": snr_db,
                "injected_group_delay_s": injected_group_delay_s,
                "calibration_group_delay_s": calibration_group_delay_s,
                "calibration_measurement_error_s": (
                    calibration_measurement_error_s
                ),
                "calibration_reference_plane_offset_s": (
                    calibration_reference_plane_offset_s
                ),
                "calibration_reference_residual_rms_rad": (
                    calibration_reference_residual_rms_rad
                ),
                "calibration_residual_s": (
                    injected_group_delay_s - calibration_group_delay_s
                ),
                "ble_raw_distance_m": ble_raw_distance_m,
                "ble_raw_error_m": ble_raw_distance_m - distance,
                "ble_calibrated_distance_m": ble_calibrated_distance_m,
                "ble_calibrated_error_m": (
                    ble_calibrated_distance_m - distance
                ),
                "ble_robust_available": ble_robust_available,
                "ble_robust_status": ble_robust_status,
                "ble_robust_reason": ble_robust_reason,
                "ble_robust_confidence": ble_robust_confidence,
                "ble_robust_estimator_mode": ble_robust_estimator_mode,
                "ble_robust_model_order": ble_robust_model_order,
                "ble_robust_reconstruction_residual": (
                    ble_robust_reconstruction_residual
                ),
                "ble_robust_candidate_distance_m": (
                    ble_robust_candidate_distance_m
                ),
                "ble_accepted_distance_m": ble_accepted_distance_m,
                "ble_accepted_error_m": (
                    ble_accepted_distance_m - distance
                ),
                "uwb_raw_distance_m": uwb_raw_distance_m,
                "uwb_raw_error_m": uwb_raw_distance_m - distance,
                "uwb_calibrated_distance_m": uwb_calibrated_distance_m,
                "uwb_calibrated_error_m": (
                    uwb_calibrated_distance_m - distance
                ),
                "uwb_status": uwb_status,
                "nlos_probability": nlos_probability,
                "failure": failure,
            }
        )

    csv_path = output_dir / "simulation_monte_carlo.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ble_raw_errors = np.asarray(
        [row["ble_raw_error_m"] for row in rows], dtype=float
    )
    ble_calibrated_errors = np.asarray(
        [row["ble_calibrated_error_m"] for row in rows], dtype=float
    )
    uwb_raw_errors = np.asarray(
        [row["uwb_raw_error_m"] for row in rows], dtype=float
    )
    uwb_calibrated_errors = np.asarray(
        [row["uwb_calibrated_error_m"] for row in rows], dtype=float
    )
    ble_accepted_errors = np.asarray(
        [row["ble_accepted_error_m"] for row in rows], dtype=float
    )
    ble_calibrated_metrics = _error_metrics(ble_calibrated_errors)
    uwb_calibrated_metrics = _error_metrics(uwb_calibrated_errors)
    robust_reason_counts: dict[str, int] = {}
    robust_status_counts: dict[str, int] = {}
    robust_estimator_mode_counts: dict[str, int] = {}
    robust_model_order_counts: dict[str, int] = {}
    uwb_status_counts: dict[str, int] = {}
    for row in rows:
        robust_status = str(row["ble_robust_status"])
        robust_status_counts[robust_status] = (
            robust_status_counts.get(robust_status, 0) + 1
        )
        estimator_mode = str(row["ble_robust_estimator_mode"])
        robust_estimator_mode_counts[estimator_mode] = (
            robust_estimator_mode_counts.get(estimator_mode, 0) + 1
        )
        model_order = str(row["ble_robust_model_order"])
        robust_model_order_counts[model_order] = (
            robust_model_order_counts.get(model_order, 0) + 1
        )
        if not row["ble_robust_available"]:
            reason = str(row["ble_robust_reason"] or "unspecified")
            robust_reason_counts[reason] = robust_reason_counts.get(reason, 0) + 1
        uwb_status = str(row["uwb_status"])
        uwb_status_counts[uwb_status] = uwb_status_counts.get(uwb_status, 0) + 1
    ble_robust_metrics = _error_metrics(ble_accepted_errors)
    finite_robust_errors = np.abs(
        ble_accepted_errors[np.isfinite(ble_accepted_errors)]
    )
    ble_robust_metrics["accepted_error_gt_1m_count"] = int(
        np.sum(finite_robust_errors > 1.0)
    )
    ble_robust_metrics["accepted_error_gt_1m_rate"] = (
        float(np.mean(finite_robust_errors > 1.0))
        if finite_robust_errors.size
        else None
    )
    ble_robust_metrics["status_counts"] = robust_status_counts
    ble_robust_metrics["rejection_reasons"] = robust_reason_counts
    ble_robust_metrics["estimator_mode_counts"] = robust_estimator_mode_counts
    ble_robust_metrics["model_order_counts"] = robust_model_order_counts
    calibration_errors_s = np.asarray(
        [row["calibration_measurement_error_s"] for row in rows], dtype=float
    )
    calibration_reference_residuals = np.asarray(
        [
            row["calibration_reference_residual_rms_rad"]
            for row in rows
        ],
        dtype=float,
    )
    return {
        "dataset": "internal simulation only; not field measurements",
        "seed": 20260821,
        "rfic_calibration": (
            "scalar delay is estimated from a separate 40 dB known-through "
            "reference sweep with independent noise and reference-plane "
            "offset; the pipeline has no automatic access to the injected "
            "impairment configuration"
        ),
        "rfic_calibration_evidence": {
            "method": "independent_known_through_phase_slope",
            "reference_sweep_snr_db": 40.0,
            "reference_sweep_phase_noise_std_deg": 0.2,
            "reference_plane_offset_std_s": 0.02e-9,
            "pipeline_received_injected_truth": False,
            "measurement_error_bias_s": float(np.mean(calibration_errors_s)),
            "measurement_error_p95_abs_s": float(
                np.percentile(np.abs(calibration_errors_s), 95)
            ),
            "reference_fit_residual_p95_rad": float(
                np.percentile(calibration_reference_residuals, 95)
            ),
        },
        "ble": ble_calibrated_metrics,
        "uwb": uwb_calibrated_metrics,
        "ble_raw": _error_metrics(ble_raw_errors),
        "ble_calibrated": ble_calibrated_metrics,
        "ble_robust_accepted": ble_robust_metrics,
        "uwb_raw": _error_metrics(uwb_raw_errors),
        "uwb_calibrated": uwb_calibrated_metrics,
        "uwb_status_counts": uwb_status_counts,
        "csv": str(csv_path),
    }


def check_ble_multipath_monte_carlo(
    simulation_metrics: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    metrics = dict(simulation_metrics["ble_robust_accepted"])
    calibration_evidence = dict(
        simulation_metrics["rfic_calibration_evidence"]
    )
    availability = float(metrics.get("availability", 0.0))
    rmse = metrics.get("rmse_m")
    p95 = metrics.get("p95_abs_error_m")
    catastrophic_rate = metrics.get("accepted_error_gt_1m_rate")
    passed = bool(
        availability >= 0.70
        and rmse is not None
        and float(rmse) <= 0.40
        and p95 is not None
        and float(p95) <= 0.75
        and catastrophic_rate is not None
        and float(catastrophic_rate) <= 0.01
        and calibration_evidence.get("pipeline_received_injected_truth") is False
    )
    metrics["calibration_evidence"] = calibration_evidence
    metrics["gates"] = {
        "minimum_availability": 0.70,
        "maximum_rmse_m": 0.40,
        "maximum_p95_abs_error_m": 0.75,
        "maximum_accepted_error_gt_1m_rate": 0.01,
    }
    return (
        passed,
        metrics,
        "固定 seed、200 次内部复多径回归；门槛只防止软件退化，不代表现场覆盖率或协议一致性。",
    )


def check_uwb_monte_carlo(
    simulation_metrics: dict[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    metrics = dict(simulation_metrics["uwb_calibrated"])
    resolution_m = C0 / 1.0e9
    p95 = metrics.get("p95_abs_error_m")
    availability = float(metrics.get("availability", 0.0))
    passed = (
        availability == 1.0
        and p95 is not None
        and float(p95) < resolution_m
    )
    metrics["c_over_bandwidth_gate_m"] = resolution_m
    return (
        passed,
        metrics,
        "固定 seed 的内部多径/噪声回归；是软件退化门槛，不是实测或 IEEE 场地认证。",
    )


def manual_coverage_results(root: Path) -> list[CheckResult]:
    matlab_summary_path = (
        root / "matlab_validation" / "output" / "matlab_validation_summary.json"
    )
    matlab_evidence: dict[str, Any] = {"summary_file": str(matlab_summary_path)}
    matlab_status = STATUS_NOT_TESTED
    matlab_note = "本轮未发现 MATLAB 交叉验证结果。"
    if matlab_summary_path.exists():
        matlab_summary = json.loads(matlab_summary_path.read_text(encoding="utf-8"))
        matlab_evidence.update(matlab_summary)
        source_files = list((root / "matlab_validation").rglob("*.m"))
        latest_source_mtime = max(
            (path.stat().st_mtime for path in source_files), default=0.0
        )
        summary_mtime = matlab_summary_path.stat().st_mtime
        fresh = summary_mtime >= latest_source_mtime
        matlab_evidence.update(
            {
                "summary_mtime": datetime.fromtimestamp(summary_mtime)
                .astimezone()
                .isoformat(timespec="seconds"),
                "latest_source_mtime": datetime.fromtimestamp(
                    latest_source_mtime
                )
                .astimezone()
                .isoformat(timespec="seconds"),
                "fresh_relative_to_matlab_sources": fresh,
            }
        )
        if not fresh:
            matlab_status = STATUS_NOT_TESTED
            matlab_note = "结果早于 MATLAB 源码，按 STALE / NOT TESTED 处理。"
        else:
            matlab_status = (
                STATUS_PASS if matlab_summary.get("all_pass") else STATUS_FAIL
            )
            matlab_note = "结果文件新鲜度已相对 MATLAB 源码时间戳核验。"

    return [
        CheckResult(
            "STD-MATLAB",
            "Python/MATLAB 解析模型交叉验证",
            matlab_status,
            "自由空间和 two-ray 的 MATLAB 独立实现均通过",
            matlab_evidence,
            matlab_note,
        ),
        CheckResult(
            "STD-BT-CS",
            "Bluetooth Core 6.3 Channel Sounding 协议一致性",
            STATUS_PARTIAL,
            "官方频点、Mode 0–3、PBR/RTT、FFO 校准及 CS Test Suite",
            {
                "official_frequency_math": "covered by NUM-BLE-LOS and NUM-BLE-RTT",
                "pbr_round_trip_phase_factor": True,
                "ambiguity_aware_quality_and_rejection": "covered by REG-BLE-ROBUST",
                "mode_1_rtt_timestamp_engine": False,
                "mode_0_to_3_engine": False,
                "ffo_cfo_compensation_inside_estimator": False,
                "cs_packet_and_step_timing": False,
                "official_test_vectors_executed": False,
            },
            "数值解析和拒绝机制可用，但 CFO/FFO 仍须外部补偿，且尚不是 Bluetooth CS 协议实现。",
        ),
        CheckResult(
            "STD-IEEE-UWB-PHY",
            "IEEE 802.15.4-2024 UWB PHY/测距一致性",
            STATUS_NOT_TESTED,
            "HRP UWB 波形、前导码、时间戳和标准测距过程",
            {
                "generic_cfr_to_cir": True,
                "generic_cfar_and_no_path": "covered by REG-UWB-CFAR",
                "hrp_waveform": False,
                "standard_preamble": False,
                "official_test_vectors_executed": False,
            },
            "当前有通用 CFR/CIR、CFAR 与显式 NO_PATH；没有 HRP PHY、前导码或标准时间戳过程。",
        ),
        CheckResult(
            "STD-IEEE-UWB-CHANNEL",
            "IEEE 802.15.4a UWB 信道模型",
            STATUS_NOT_TESTED,
            "按官方场景参数复现路径损耗、簇/射线和 delay spread 统计",
            {
                "saleh_valenzuela_style_generator": True,
                "official_scenario_parameter_tables": False,
                "official_reference_realizations_compared": False,
            },
            "内部随机机制已验收，但参数没有标准追溯链。",
        ),
        CheckResult(
            "STD-ITU-INDOOR",
            "ITU-R P.1238-13 室内传播",
            STATUS_NOT_TESTED,
            "路径损耗、墙/楼层损耗、阴影和 RMS delay spread",
            {
                "itu_path_loss_model": False,
                "itu_material_or_wall_tables": False,
                "itu_statistical_comparison": False,
            },
        ),
        CheckResult(
            "STD-3GPP-CHANNEL",
            "3GPP TR 38.901 通用簇信道",
            STATUS_NOT_TESTED,
            "场景参数、K-factor、簇/角度/时延统计",
            {
                "3gpp_scenario_parameters": False,
                "reference_statistics_compared": False,
            },
        ),
        CheckResult(
            "STD-RFIC",
            "RFIC S 参数与群时延测量闭环",
            STATUS_PARTIAL,
            "真实 Cadence/VNA S2P 的幅相、群时延和参考面一致性",
            {
                "synthetic_touchstone_v1": "covered by IO-S2P",
                "frequency_dependent_profile": "covered by NUM-RFIC-FREQ-CAL",
                "real_s2p_files_found": len(list(root.rglob("*.s2p"))),
                "touchstone_2_1_full_parser": False,
                "vna_reference_plane_documented": False,
            },
            "计算接口存在，但没有真实 RFIC 数据可供验收。",
        ),
        CheckResult(
            "STD-ISO18305",
            "ISO/IEC 18305 系统级场地验收",
            STATUS_NOT_TESTED,
            "测绘真值、建筑/运动场景、覆盖率、延迟和误差分位数",
            {
                "surveyed_ground_truth_dataset": False,
                "multi_building_field_test": False,
                "simulation_error_percentiles": "generated for regression only",
            },
            "模拟 Monte Carlo 指标不能替代标准场地试验。",
        ),
        CheckResult(
            "STD-AI",
            "AI/SNN NLOS 与校正效果",
            STATUS_NOT_TESTED,
            "独立房间/材料/人员留出测试和真实标签",
            {
                "inference_interface": True,
                "simulation_oracle_features_present": True,
                "trained_model": False,
                "labeled_dataset": False,
                "leave_one_site_out": False,
            },
            "现有特征含仿真路径/SNR/CFO/IQ 注入真值，只能验收接口，不能验收模型效果或独立泛化。",
        ),
    ]


def _status_counts(results: list[CheckResult]) -> dict[str, int]:
    return {
        status: sum(result.status == status for result in results)
        for status in (STATUS_PASS, STATUS_PARTIAL, STATUS_FAIL, STATUS_NOT_TESTED)
    }


def determine_gate_statuses(results: list[CheckResult]) -> tuple[str, str, str]:
    software = [
        result
        for result in results
        if result.check_id.startswith(("NUM-", "STAT-", "IO-", "REG-"))
    ]
    standard = [result for result in results if result.check_id.startswith("STD-")]
    software_gate = (
        STATUS_PASS
        if software and all(result.status == STATUS_PASS for result in software)
        else STATUS_FAIL
    )
    if any(result.status == STATUS_FAIL for result in standard):
        standards_status = STATUS_FAIL
    elif any(
        result.status in {STATUS_PARTIAL, STATUS_NOT_TESTED}
        for result in standard
    ):
        standards_status = STATUS_PARTIAL
    else:
        standards_status = STATUS_PASS
    if software_gate == STATUS_FAIL or standards_status == STATUS_FAIL:
        overall_status = STATUS_FAIL
    elif standards_status == STATUS_PARTIAL:
        overall_status = STATUS_PARTIAL
    else:
        overall_status = STATUS_PASS
    return software_gate, standards_status, overall_status


def write_report(
    output_dir: Path,
    results: list[CheckResult],
    simulation_metrics: dict[str, Any],
    overall_status: str,
    software_gate_status: str,
    standards_status: str,
) -> Path:
    counts = _status_counts(results)
    software_blockers = [
        result
        for result in results
        if result.check_id.startswith(("NUM-", "STAT-", "IO-", "REG-"))
        and result.status != STATUS_PASS
    ]
    standards_gaps = [
        result
        for result in results
        if result.check_id.startswith("STD-") and result.status != STATUS_PASS
    ]
    if overall_status == STATUS_FAIL:
        overall_text = "拒收：至少一个必选软件门槛或已执行交叉验证失败"
    elif overall_status == STATUS_PARTIAL:
        overall_text = "软件门槛通过；标准一致性与实测闭环尚未完成"
    else:
        overall_text = "全部列出的软件与标准覆盖门槛通过"
    lines = [
        "# RF Ranging 平台独立验收报告",
        "",
        f"- 验收时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 总体结论：**{overall_status}（{overall_text}）**",
        f"- 必选软件门槛：**{software_gate_status}**",
        f"- 标准/实测覆盖：**{standards_status}**",
        "- 原则：软件计算通过不等于 IEEE/Bluetooth 合规，也不等于实测精度通过。",
        "",
        "## 结论摘要",
        "",
    ]
    lines.extend(
        [
            f"- PASS：{counts[STATUS_PASS]}",
            f"- PARTIAL：{counts[STATUS_PARTIAL]}",
            f"- FAIL：{counts[STATUS_FAIL]}",
            f"- NOT TESTED：{counts[STATUS_NOT_TESTED]}",
            "",
            "平台可以作为跨层物理仿真的研究基线继续开发，但不能声明已经通过 "
            "IEEE 802.15.4、Bluetooth Channel Sounding、ITU 或 ISO 产品级认证。",
            "",
            "## 逐项结果",
            "",
            "| ID | 验收项 | 状态 | 判据/范围 |",
            "|---|---|---:|---|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.check_id} | {result.title} | **{result.status}** | "
            f"{result.criterion.replace('|', '/')} |"
        )
    lines.extend(["", "## 关键证据", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.check_id} — {result.title}",
                "",
                f"- 状态：{result.status}",
                f"- 证据：{json.dumps(result.evidence, ensure_ascii=False)}",
            ]
        )
        if result.note:
            lines.append(f"- 说明：{result.note}")
        lines.append("")

    lines.extend(
        [
            "## 仿真回归指标（非实测）",
            "",
            f"- BLE 原始：{json.dumps(simulation_metrics['ble_raw'], ensure_ascii=False)}",
            f"- BLE 标量校准候选（不含质量拒绝）：{json.dumps(simulation_metrics['ble_calibrated'], ensure_ascii=False)}",
            f"- BLE 鲁棒门控后的接受子集：{json.dumps(simulation_metrics['ble_robust_accepted'], ensure_ascii=False)}",
            f"- UWB 原始：{json.dumps(simulation_metrics['uwb_raw'], ensure_ascii=False)}",
            f"- UWB 标量校准后：{json.dumps(simulation_metrics['uwb_calibrated'], ensure_ascii=False)}",
            f"- UWB 状态分布：{json.dumps(simulation_metrics['uwb_status_counts'], ensure_ascii=False)}",
            "- 这些分位数仅用于以后代码变更的回归比较，不能代替 ISO/IEC 18305 场地测试。",
            "",
            "## 当前阻断项与覆盖缺口",
            "",
        ]
    )
    if software_blockers:
        lines.append("必选软件阻断项：")
        lines.append("")
        for index, result in enumerate(software_blockers, start=1):
            lines.append(
                f"{index}. {result.check_id} — {result.title}（{result.status}）"
            )
    else:
        lines.append("必选软件阻断项：无。")
    lines.extend(["", "标准/实测覆盖缺口：", ""])
    if standards_gaps:
        for index, result in enumerate(standards_gaps, start=1):
            lines.append(
                f"{index}. {result.check_id} — {result.title}（{result.status}）："
                f"{result.note or result.criterion}"
            )
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 已知算法边界",
            "",
            "- BLE 鲁棒门控在洁净、缺频和 5 dB LOS/RTT 压力向量通过，但不能识别所有看似干净的 NLOS；一般随机复杂多径的选择性指标单独报告，不外推为厘米级。CFO/FFO 时序补偿仍是估计器外部前置条件。",
            "- UWB 单帧在 5 dB、0.1:1 弱首径/强晚径下仍可能给出晚径数值；定向测试显式启用保守早径增强后不会错误升级为 PASS。生产 pipeline 和图形平台默认采用四帧相位对齐积累，帧间 coherence 低于 0.75 时会降为低置信度；单帧增强默认关闭。该结论不外推到一般复多径、运动、帧间频率斜率或未补偿 CFO。",
            "- 频率相关 RFIC Profile 已通过独立异频网格的合成校准与安全守卫；真实系统仍必须替换为带参考面记录的 VNA/环回曲线。标量模式不能消除群时延波纹。",
            "- BLE 无模糊范围由实际频道图的频率格最大公因数决定；估计器会拒绝跨越该范围的搜索先验。",
            "- AI/NLOS 卡片仍使用仿真真值路径和注入 SNR/CFO/IQ 特征，明确排除在独立软件精度门槛之外。",
            "",
            "## 标准依据",
            "",
        ]
    )
    for name, url in STANDARD_REFERENCES.items():
        lines.append(f"- [{name}]({url})")
    if overall_status == STATUS_FAIL:
        final_text = (
            "本次判定为 **FAIL**：上列软件阻断项必须修复后才能接收该研究基线。"
        )
    elif overall_status == STATUS_PARTIAL:
        final_text = (
            "本次判定为 **PARTIAL**：全部必选软件门槛通过，但协议级标准、真实 RFIC "
            "标定、实测数据与场地闭环尚未完成；不得宣称 IEEE 或 Bluetooth 认证。"
        )
    else:
        final_text = "本次判定为 **PASS**：报告中列出的全部门槛均已通过。"
    lines.extend(["", "## 最终判定", "", final_text, ""])
    report_path = output_dir / "ACCEPTANCE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent platform acceptance.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "acceptance",
    )
    args = parser.parse_args(argv)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    operations = [
        (
            "REG-BLE-ROBUST",
            "BLE 模糊度感知解析、质量门控与拒绝机制",
            "洁净/缺频/CFO补偿向量通过；5 dB RTT100m 200次可用率>=90%、条件P95<0.15m、接受结果无>1m；纯噪声零接受",
            check_ble_robust_acceptance,
        ),
        (
            "REG-UWB-CFAR",
            "UWB CFAR、LOW_CONFIDENCE 与 NO_PATH",
            "理想误差<0.04m；纯噪声512次零误报；-12dB可用率>=95%；10dB弱首径全部降级且P95<0.04m",
            check_uwb_cfar_acceptance,
        ),
        (
            "REG-UWB-WEAK-LATE-5DB",
            "UWB 5 dB 弱首径/强晚径多帧恢复",
            "默认pipeline与独立4帧相位对齐：0.10m内正确率>=95%、P95<=0.10m、晚锁<=1%、错误晚径PASS=0、纯噪声零误报",
            check_uwb_weak_late_multiframe,
        ),
        (
            "NUM-RFIC-FREQ-CAL",
            "频率相关 RFIC 复数 Profile 显式校准",
            "独立异频网格、path_factor=1/2：raw偏差>0.2m、改善>=20倍、BLE<1e-8m、UWB<一距离格及六类安全拒绝",
            check_frequency_dependent_calibration,
        ),
        (
            "IO-WEB-UI",
            "图形平台 HTTP、状态、严格 JSON 与验收新鲜度",
            "HTTP 200/400/404/413/422；BLE动态拒绝；raw/candidate/calibrated；NO_PATH=null；schema/fresh/stale与DOM契约",
            check_web_ui_contract,
        ),
        (
            "NUM-BLE-LOS",
            "Bluetooth CS 官方频点理想 LOS 距离扫描",
            "72 个允许频点，0.5–149 m 最大绝对误差 < 1e-8 m",
            check_ble_los_sweep,
        ),
        (
            "NUM-BLE-RTT",
            "BLE PBR 往返相位路径系数（非 Mode-1 RTT）",
            "0.5–100 m 最大绝对误差 < 1e-8 m",
            check_ble_round_trip_factor,
        ),
        (
            "NUM-UWB-LOS",
            "UWB 理想 LOS 首径分辨率",
            "0.5–149 m 最大误差小于一个 8 倍零填充距离格",
            check_uwb_los_resolution,
        ),
        (
            "NUM-RFIC-GD",
            "RFIC 固定群时延到距离偏差映射",
            "-0.8–3 ns 最大映射误差 < 1e-8 m",
            check_group_delay_mapping,
        ),
        (
            "NUM-RFIC-CAL",
            "RFIC 固定群时延显式校准",
            "path_factor=1/2：BLE <1e-8 m，UWB <一个零填充距离格；过大错误校准必须拒绝",
            check_group_delay_calibration,
        ),
        (
            "NUM-PHASE-OFFSET",
            "常数相位偏置不改变相位斜率距离",
            "-2.5–2.9 rad 最大距离误差 < 1e-8 m",
            check_phase_offset_invariance,
        ),
        (
            "STAT-SV-ARRIVAL",
            "Saleh–Valenzuela 到达率统计",
            "1000 realizations 的簇/射线均值相对误差均 < 5%",
            check_stochastic_arrival_rates,
        ),
        (
            "STAT-SNR",
            "AWGN 下距离 RMSE 随 SNR 单调改善",
            "10/20/30/40 dB 的 RMSE 严格递减",
            check_noise_scaling,
        ),
        (
            "IO-S2P",
            "Touchstone v1 S21 读取",
            "合成 DB 格式端点幅度误差 < 1e-12",
            check_touchstone_v1_parser,
        ),
    ]
    results = [
        _run_check(check_id, title, criterion, operation)
        for check_id, title, criterion, operation in operations
    ]
    simulation_metrics = run_simulation_monte_carlo(output_dir)
    results.append(
        _run_check(
            "REG-BLE-COMPLEX-MP",
            "BLE 一般复多径子空间解析回归门槛",
            "独立参考扫频校准、固定200次：可用率>=70%、RMSE<=0.40m、P95<=0.75m、接受结果>1m比例<=1%",
            lambda: check_ble_multipath_monte_carlo(simulation_metrics),
        )
    )
    results.append(
        _run_check(
            "REG-UWB-MC",
            "UWB 固定 Monte Carlo 回归门槛",
            "200 次 availability=1 且 P95 绝对误差 < c/B",
            lambda: check_uwb_monte_carlo(simulation_metrics),
        )
    )
    results.extend(manual_coverage_results(root))
    (
        software_gate_status,
        standards_status,
        overall_status,
    ) = determine_gate_statuses(results)

    payload = {
        "schema_version": webapp_module.ACCEPTANCE_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "software_gate_status": software_gate_status,
        "standards_status": standards_status,
        "status_counts": _status_counts(results),
        "checks": [asdict(result) for result in results],
        "simulation_metrics": simulation_metrics,
        "standard_references": STANDARD_REFERENCES,
    }
    json_path = output_dir / "acceptance_results.json"
    json_path.write_text(
        json.dumps(
            payload, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    report_path = write_report(
        output_dir,
        results,
        simulation_metrics,
        overall_status,
        software_gate_status,
        standards_status,
    )
    print(json.dumps(payload["status_counts"], ensure_ascii=False))
    print(f"OVERALL_STATUS={overall_status}")
    print(f"SOFTWARE_GATE_STATUS={software_gate_status}")
    print(f"STANDARDS_STATUS={standards_status}")
    print(f"JSON={json_path}")
    print(f"REPORT={report_path}")
    return 1 if overall_status == STATUS_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
