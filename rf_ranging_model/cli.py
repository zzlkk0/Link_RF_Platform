"""Command-line reference experiment and artifact generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ble.pbr import estimate_distance
from .channel.multipath import frequency_response
from .constants import C0
from .pipeline import RangingPipeline
from .rfic.impairments import RFICImpairmentConfig
from .types import PropagationPath


def _default_paths(distance_m: float) -> list[PropagationPath]:
    direct_delay = distance_m / C0
    return [
        PropagationPath(direct_delay, 1.0 + 0.0j, "LOS"),
        PropagationPath(
            direct_delay + 6.0e-9,
            0.55 * np.exp(1j * 2.55),
            "wall reflection",
        ),
        PropagationPath(
            direct_delay + 15.0e-9,
            0.22 * np.exp(-1j * 0.7),
            "diffuse reflection",
        ),
    ]


def _save_plots(
    output_dir: Path,
    ble_frequencies: np.ndarray,
    result: object,
    true_delay_s: float,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(
        ble_frequencies / 1.0e9,
        20.0 * np.log10(np.maximum(np.abs(result.ble_measured_response), 1.0e-12)),
    )
    axes[0].set(xlabel="Frequency (GHz)", ylabel="Normalized magnitude (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        ble_frequencies / 1.0e9,
        result.ble_pbr.unwrapped_phase_rad,
        label="measured",
    )
    axes[1].plot(
        ble_frequencies / 1.0e9,
        result.ble_pbr.fitted_phase_rad,
        "--",
        label="linear fit",
    )
    axes[1].set(xlabel="Frequency (GHz)", ylabel="Unwrapped phase (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("BLE Channel Sounding: multipath + RFIC")
    figure.savefig(output_dir / "ble_cfr_phase.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    delay_ns = result.cir_delay_s * 1.0e9
    magnitude = np.abs(result.cir)
    display_limit = min(float(delay_ns[-1]), (true_delay_s + 50.0e-9) * 1.0e9)
    mask = delay_ns <= display_limit
    axis.plot(delay_ns[mask], magnitude[mask] / np.max(magnitude), label="CIR")
    axis.axvline(true_delay_s * 1.0e9, color="tab:green", linestyle="--", label="true LOS")
    axis.axvline(
        result.uwb_first_path.delay_s * 1.0e9,
        color="tab:red",
        linestyle=":",
        label="detected first path",
    )
    axis.set(xlabel="Delay (ns)", ylabel="Normalized magnitude", ylim=(0.0, 1.08))
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title("UWB CIR and first-path detection")
    figure.savefig(output_dir / "uwb_cir.png", dpi=160)
    plt.close(figure)


def run_reference_experiment(output_dir: Path, seed: int = 7) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_distance_m = 8.0
    ble_frequencies = np.arange(2.402e9, 2.481e9, 2.0e6)
    uwb_frequencies = np.linspace(3.5e9, 4.5e9, 1001)
    los_path = PropagationPath(true_distance_m / C0, 1.0 + 0.0j, "LOS")
    los_response = frequency_response(ble_frequencies, [los_path])
    los_result = estimate_distance(ble_frequencies, los_response)

    synthetic_calibration_delay_s = 0.18e-9
    impairments = RFICImpairmentConfig(
        phase_offset_rad=np.deg2rad(12.0),
        base_group_delay_s=synthetic_calibration_delay_s,
        group_delay_ripple_s=0.08e-9,
        group_delay_ripple_period_hz=24.0e6,
        amplitude_ripple_db=0.5,
        iq_gain_mismatch_db=0.4,
        iq_phase_mismatch_deg=2.0,
        phase_noise_std_deg=0.5,
        cfo_hz=20.0,
        tone_interval_s=8.0e-6,
        snr_db=28.0,
    )
    paths = _default_paths(true_distance_m)
    result = RangingPipeline().run(
        paths,
        ble_frequencies,
        uwb_frequencies,
        rfic_config=impairments,
        seed=seed,
        calibrate_rfic_group_delay=True,
        rfic_group_delay_calibration_s=synthetic_calibration_delay_s,
    )

    summary: dict[str, object] = {
        "seed": seed,
        "true_distance_m": true_distance_m,
        "path_count": len(paths),
        "ble_los_distance_m": los_result.distance_m,
        "ble_los_error_mm": 1.0e3 * (los_result.distance_m - true_distance_m),
        "ble_multipath_rfic_distance_m": result.ble_pbr.distance_m,
        "ble_multipath_rfic_error_m": result.ble_pbr.distance_m - true_distance_m,
        "ble_robust_status": (
            result.ble_robust.status if result.ble_robust is not None else None
        ),
        "ble_robust_available": (
            result.ble_robust.availability
            if result.ble_robust is not None
            else False
        ),
        "ble_robust_rejection_reason": (
            result.ble_robust.rejection_reason
            if result.ble_robust is not None
            else "missing_quality_decision"
        ),
        "ble_robust_confidence": (
            result.ble_robust.confidence
            if result.ble_robust is not None
            else 0.0
        ),
        "ble_group_delay_calibrated_distance_m": (
            result.ble_calibrated_distance_m
        ),
        "ble_group_delay_calibrated_error_m": (
            result.ble_calibrated_distance_m - true_distance_m
        ),
        "ble_phase_residual_rms_deg": float(
            np.rad2deg(result.ble_pbr.residual_rms_rad)
        ),
        "uwb_first_path_raw_distance_m": result.uwb_raw_distance_m,
        "uwb_first_path_raw_error_m": (
            result.uwb_raw_distance_m - true_distance_m
        ),
        "uwb_first_path_distance_m": result.uwb_distance_m,
        "uwb_first_path_error_m": result.uwb_distance_m - true_distance_m,
        "uwb_first_path_status": result.uwb_first_path.status.value,
        "uwb_first_path_available": result.uwb_first_path.available,
        "uwb_first_path_rejection_reason": (
            result.uwb_first_path.rejection_reason
        ),
        "uwb_first_path_confidence": result.uwb_first_path.confidence,
        "rfic_group_delay_calibration_s": (
            result.rfic_group_delay_calibration_s
        ),
        "rfic_group_delay_calibration_source": (
            "explicit synthetic loopback-equivalent value for demo only"
        ),
        "rfic_calibration_mode": result.rfic_calibration_mode,
        "nlos_probability": result.nlos_probability,
        "ai_features": result.features.as_dict(),
        "note": (
            "AI/SNN outputs use simulation-oracle features in this demo and "
            "are excluded from independent accuracy acceptance."
        ),
    }
    (output_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "paths.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["label", "delay_ns", "magnitude", "phase_rad"],
        )
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "label": path.label,
                    "delay_ns": path.delay_s * 1.0e9,
                    "magnitude": abs(path.complex_gain),
                    "phase_rad": np.angle(path.complex_gain),
                }
            )
    _save_plots(output_dir, ble_frequencies, result, true_distance_m / C0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RF ranging cross-layer reference experiment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "demo",
        help="directory for JSON, CSV, and PNG outputs",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_reference_experiment(args.output, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts: {args.output.resolve()}")
    return 0
