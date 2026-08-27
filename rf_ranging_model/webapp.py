"""Local, dependency-free graphical front end for the ranging platform."""

from __future__ import annotations

import argparse
import json
import math
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .ble.channel_sounding import allowed_cs_frequencies_hz
from .ble.robust import RobustPBRConfig
from .constants import C0
from .pipeline import RangingPipeline
from .rfic.calibration import build_calibration_profile
from .rfic.impairments import (
    RFICImpairmentConfig,
    group_delay_phase_response,
)
from .types import PropagationPath


STATIC_ROOT = Path(__file__).with_name("web_static")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_JSON = PROJECT_ROOT / "outputs" / "acceptance" / "acceptance_results.json"
MAX_REQUEST_BYTES = 1_000_000
ACCEPTANCE_SCHEMA_VERSION = "2.0"


class ScenarioInputError(ValueError):
    """Raised when a graphical-platform request contains invalid input."""


def _number(
    payload: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ScenarioInputError(f"{key} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioInputError(f"{key} must be a number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ScenarioInputError(
            f"{key} must lie in [{minimum:g}, {maximum:g}]"
        )
    return result


def _integer(
    payload: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ScenarioInputError(f"{key} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioInputError(f"{key} must be an integer") from exc
    if result != float(value) or not minimum <= result <= maximum:
        raise ScenarioInputError(
            f"{key} must be an integer in [{minimum}, {maximum}]"
        )
    return result


def _boolean(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ScenarioInputError(f"{key} must be true or false")
    return value


def _choice(
    payload: dict[str, Any],
    key: str,
    default: str,
    choices: set[str],
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ScenarioInputError(f"{key} must be one of: {allowed}")
    return value


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _synthetic_calibration_transfer(
    frequencies_hz: np.ndarray,
    *,
    base_group_delay_s: float,
    group_delay_ripple_s: float,
    amplitude_ripple_db: float,
    ripple_period_hz: float,
) -> np.ndarray:
    """Create an explicit UI demo sweep, separate from DUT injection state."""

    centered = frequencies_hz - float(np.mean(frequencies_hz))
    amplitude = 10.0 ** (
        amplitude_ripple_db
        * np.cos(2.0 * np.pi * centered / ripple_period_hz)
        / 20.0
    )
    phase = group_delay_phase_response(
        frequencies_hz,
        base_group_delay_s=base_group_delay_s,
        ripple_s=group_delay_ripple_s,
        ripple_period_hz=ripple_period_hz,
    )
    return np.asarray(amplitude * phase, dtype=np.complex128)


def _build_ui_profile(
    frequencies_hz: np.ndarray,
    *,
    base_group_delay_s: float,
    group_delay_ripple_s: float,
    amplitude_ripple_db: float,
    radio: str,
):
    transfer = _synthetic_calibration_transfer(
        frequencies_hz,
        base_group_delay_s=base_group_delay_s,
        group_delay_ripple_s=group_delay_ripple_s,
        amplitude_ripple_db=amplitude_ripple_db,
        ripple_period_hz=24.0e6,
    )
    maximum_expected_delay_s = max(
        0.5e-9,
        abs(base_group_delay_s) + abs(group_delay_ripple_s) + 0.5e-9,
    )
    return build_calibration_profile(
        frequencies_hz,
        transfer,
        reference_plane="channel output to ranging estimator input",
        source=f"UI explicit synthetic {radio} calibration sweep (demo only)",
        metadata={
            "radio": radio,
            "kind": "synthetic_demo_sweep",
            "measured_hardware": False,
        },
        maximum_expected_group_delay_s=maximum_expected_delay_s,
    )


def _sample_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximum_points: int = 700,
) -> tuple[list[float], list[float]]:
    if x.size <= maximum_points:
        indices = np.arange(x.size)
    else:
        indices = np.unique(
            np.linspace(0, x.size - 1, maximum_points, dtype=int)
        )
    return (
        [float(value) for value in x[indices]],
        [float(value) for value in y[indices]],
    )


def _build_paths(
    distance_m: float,
    reflection_count: int,
    reflection_gain: float,
    reflection_delay_ns: float,
    reflection_phase_rad: float,
    direct_path_gain: float = 1.0,
) -> list[PropagationPath]:
    direct_delay_s = distance_m / C0
    paths = [
        PropagationPath(
            direct_delay_s,
            complex(direct_path_gain, 0.0),
            "LOS",
        )
    ]
    for index in range(reflection_count):
        relative_delay_s = reflection_delay_ns * (index + 1) * 1.0e-9
        magnitude = reflection_gain / (1.0 + 0.35 * index)
        phase = reflection_phase_rad + 1.17 * index
        paths.append(
            PropagationPath(
                direct_delay_s + relative_delay_s,
                magnitude * np.exp(1j * phase),
                f"reflection_{index + 1}",
            )
        )
    return paths


def run_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a UI payload, run the model, and return chart-ready JSON."""

    if not isinstance(payload, dict):
        raise ScenarioInputError("request body must be a JSON object")
    distance_m = _number(payload, "distance_m", 8.0, 0.5, 100.0)
    direct_path_gain = _number(
        payload, "direct_path_gain", 1.0, 0.01, 1.5
    )
    reflection_count = _integer(payload, "reflection_count", 2, 0, 5)
    reflection_gain = _number(payload, "reflection_gain", 0.55, 0.0, 1.5)
    reflection_delay_ns = _number(
        payload, "reflection_delay_ns", 6.0, 0.5, 80.0
    )
    reflection_phase_rad = _number(
        payload, "reflection_phase_rad", 2.55, -math.pi, math.pi
    )
    snr_db = _number(payload, "snr_db", 28.0, -10.0, 120.0)
    phase_noise_deg = _number(
        payload, "phase_noise_deg", 0.5, 0.0, 30.0
    )
    cfo_hz = _number(payload, "cfo_hz", 20.0, -5_000.0, 5_000.0)
    base_group_delay_ns = _number(
        payload, "base_group_delay_ns", 0.18, -5.0, 20.0
    )
    group_delay_ripple_ns = _number(
        payload, "group_delay_ripple_ns", 0.08, 0.0, 5.0
    )
    amplitude_ripple_db = _number(
        payload, "amplitude_ripple_db", 0.5, 0.0, 6.0
    )
    uwb_frame_count = _integer(payload, "uwb_frame_count", 4, 1, 16)
    # Keep the first graphical prototype's boolean payload compatible while
    # making the new frequency-profile mode the default for the current UI.
    if "calibration_mode" in payload:
        calibration_mode = _choice(
            payload,
            "calibration_mode",
            "frequency_profile",
            {"frequency_profile", "scalar", "none"},
        )
    elif "calibration_enabled" in payload:
        calibration_mode = (
            "scalar"
            if _boolean(payload, "calibration_enabled", True)
            else "none"
        )
    else:
        calibration_mode = "frequency_profile"
    calibration_delay_ns = _number(
        payload, "calibration_delay_ns", 0.18, -5.0, 20.0
    )
    calibration_ripple_ns = _number(
        payload, "calibration_ripple_ns", 0.08, 0.0, 5.0
    )
    calibration_amplitude_ripple_db = _number(
        payload,
        "calibration_amplitude_ripple_db",
        0.5,
        0.0,
        6.0,
    )
    seed = _integer(payload, "seed", 7, 0, 2_147_483_647)

    paths = _build_paths(
        distance_m,
        reflection_count,
        reflection_gain,
        reflection_delay_ns,
        reflection_phase_rad,
        direct_path_gain,
    )
    ble_frequencies_hz = allowed_cs_frequencies_hz()
    uwb_frequencies_hz = np.linspace(3.5e9, 4.5e9, 1001)
    ble_profile = None
    uwb_profile = None
    if calibration_mode == "frequency_profile":
        profile_arguments = {
            "base_group_delay_s": calibration_delay_ns * 1.0e-9,
            "group_delay_ripple_s": calibration_ripple_ns * 1.0e-9,
            "amplitude_ripple_db": calibration_amplitude_ripple_db,
        }
        ble_profile = _build_ui_profile(
            ble_frequencies_hz,
            radio="BLE",
            **profile_arguments,
        )
        uwb_profile = _build_ui_profile(
            uwb_frequencies_hz,
            radio="UWB",
            **profile_arguments,
        )
    result = RangingPipeline().run(
        paths,
        ble_frequencies_hz,
        uwb_frequencies_hz,
        rfic_config=RFICImpairmentConfig(
            base_group_delay_s=base_group_delay_ns * 1.0e-9,
            group_delay_ripple_s=group_delay_ripple_ns * 1.0e-9,
            group_delay_ripple_period_hz=24.0e6,
            amplitude_ripple_db=amplitude_ripple_db,
            phase_noise_std_deg=phase_noise_deg,
            cfo_hz=cfo_hz,
            tone_interval_s=8.0e-6,
            snr_db=snr_db,
        ),
        seed=seed,
        calibrate_rfic_group_delay=calibration_mode == "scalar",
        rfic_group_delay_calibration_s=(
            calibration_delay_ns * 1.0e-9
            if calibration_mode == "scalar"
            else None
        ),
        ble_rfic_calibration_profile=ble_profile,
        uwb_rfic_calibration_profile=uwb_profile,
        ble_robust_config=RobustPBRConfig(max_distance_m=120.0),
        uwb_frame_count=uwb_frame_count,
    )

    ble_magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(result.ble_measured_response), 1.0e-12)
    )
    ble_final_magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(result.ble_ranging_response), 1.0e-12)
    )
    ble_phase_rad = np.unwrap(np.angle(result.ble_measured_response))
    ble_x, ble_y = _sample_xy(
        ble_frequencies_hz / 1.0e6, ble_magnitude_db
    )
    _, ble_final_y = _sample_xy(
        ble_frequencies_hz / 1.0e6, ble_final_magnitude_db
    )
    _, ble_phase = _sample_xy(
        ble_frequencies_hz / 1.0e6, ble_phase_rad
    )

    cir_delay_ns = result.cir_delay_s * 1.0e9
    display_end_ns = min(
        float(cir_delay_ns[-1]), distance_m / C0 * 1.0e9 + 90.0
    )
    cir_mask = cir_delay_ns <= display_end_ns
    cir_magnitude = np.abs(result.cir[cir_mask])
    cir_peak = max(float(np.max(cir_magnitude)), np.finfo(float).tiny)
    cir_x, cir_y = _sample_xy(
        cir_delay_ns[cir_mask], cir_magnitude / cir_peak
    )

    robust = result.ble_robust
    if robust is None:  # Defensive guard for third-party PipelineResult objects.
        raise ValueError("pipeline did not return a BLE robust decision")
    ble_legacy_candidate = _finite_or_none(
        result.ble_calibrated_distance_m
    )
    ble_candidate = _finite_or_none(
        result.ble_robust_calibrated_distance_m
    )
    if robust.availability and ble_candidate is None:
        raise ValueError(
            "available BLE robust decision has no finite calibrated candidate"
        )
    ble_usable = ble_candidate if robust.availability else None
    uwb_available = bool(result.uwb_first_path.available)
    uwb_candidate = _finite_or_none(result.uwb_distance_m)
    uwb_usable = uwb_candidate if uwb_available else None
    uwb_status_object = result.uwb_first_path.status
    uwb_status = getattr(uwb_status_object, "value", str(uwb_status_object))
    uwb_raw_status_object = (
        result.uwb_raw_first_path.status
        if result.uwb_raw_first_path is not None
        else None
    )
    uwb_raw_status = (
        getattr(uwb_raw_status_object, "value", str(uwb_raw_status_object))
        if uwb_raw_status_object is not None
        else None
    )
    calibration_evidence = {
        "mode": result.rfic_calibration_mode,
        "demo_only": calibration_mode == "frequency_profile",
        "ble_profile": ble_profile.summary() if ble_profile is not None else None,
        "uwb_profile": uwb_profile.summary() if uwb_profile is not None else None,
        "ble_application": (
            result.ble_calibration_result.evidence()
            if result.ble_calibration_result is not None
            else None
        ),
        "uwb_application": (
            result.uwb_calibration_result.evidence()
            if result.uwb_calibration_result is not None
            else None
        ),
    }
    return {
        "scenario": {
            "distance_m": distance_m,
            "direct_path_gain": direct_path_gain,
            "reflection_count": reflection_count,
            "reflection_gain": reflection_gain,
            "reflection_delay_ns": reflection_delay_ns,
            "path_count": len(paths),
            "snr_db": snr_db,
            "phase_noise_deg": phase_noise_deg,
            "seed": seed,
            "calibration_mode": calibration_mode,
            "calibration_delay_ns": calibration_delay_ns,
            "calibration_ripple_ns": calibration_ripple_ns,
            "uwb_frame_count": result.uwb_frame_count,
            "uwb_frame_coherence": _finite_or_none(
                result.uwb_frame_coherence
            ),
        },
        "summary": {
            "ble": {
                "available": robust.availability,
                "status": robust.status.upper(),
                "rejection_reason": robust.rejection_reason,
                "raw_distance_m": _finite_or_none(result.ble_raw_distance_m),
                "legacy_single_path_distance_m": ble_legacy_candidate,
                "candidate_distance_m": ble_candidate,
                "candidate_source": "ble_robust_calibrated",
                "calibrated_distance_m": ble_usable,
                "error_m": (
                    ble_usable - distance_m if ble_usable is not None else None
                ),
                "confidence": _finite_or_none(robust.confidence),
                "ambiguity_peak_ratio": _finite_or_none(robust.peak_ratio),
                "phase_residual_rms_deg": _finite_or_none(
                    np.rad2deg(result.ble_pbr.residual_rms_rad)
                ),
                "unambiguous_distance_m": _finite_or_none(
                    robust.unambiguous_distance_m
                ),
                "quality_flags": list(robust.quality_flags),
                "estimator_mode": robust.estimator_mode,
                "model_order": robust.model_order,
                "reconstruction_residual": _finite_or_none(
                    robust.reconstruction_residual
                ),
                "legacy_disagreement_m": _finite_or_none(
                    robust.legacy_disagreement_m
                ),
            },
            "uwb": {
                "available": uwb_available,
                "status": uwb_status,
                "rejection_reason": result.uwb_first_path.rejection_reason,
                "raw_status": uwb_raw_status,
                "raw_distance_m": _finite_or_none(result.uwb_raw_distance_m),
                "candidate_distance_m": uwb_candidate,
                "calibrated_distance_m": uwb_usable,
                "error_m": (
                    uwb_usable - distance_m if uwb_usable is not None else None
                ),
                "confidence": _finite_or_none(
                    result.uwb_first_path.confidence
                ),
                "ambiguity_ratio": _finite_or_none(
                    result.uwb_first_path.ambiguity_ratio
                ),
            },
            "nlos_probability": result.nlos_probability,
            "nlos_evidence_scope": "simulation_oracle_features_only",
            "rfic_calibration": calibration_evidence,
        },
        "series": {
            "ble_frequency_mhz": ble_x,
            "ble_raw_magnitude_db": ble_y,
            "ble_calibrated_magnitude_db": ble_final_y,
            "ble_magnitude_db": ble_final_y,
            "ble_phase_rad": ble_phase,
            "cir_delay_ns": cir_x,
            "cir_normalized_magnitude": cir_y,
            "true_delay_ns": distance_m / C0 * 1.0e9,
            "detected_delay_ns": _finite_or_none(
                result.uwb_first_path.delay_s * 1.0e9
            ),
        },
        "paths": [
            {
                "label": path.label,
                "delay_ns": path.delay_s * 1.0e9,
                "magnitude": abs(path.complex_gain),
                "phase_rad": float(np.angle(path.complex_gain)),
            }
            for path in paths
        ],
        "ble_estimated_paths": [
            {
                "label": f"BLE estimate {index + 1}",
                "delay_ns": (
                    path_delay_s - result.rfic_group_delay_calibration_s
                )
                * 1.0e9,
                "magnitude": float(path_amplitude),
                "selected": bool(
                    robust.delay_s is not None
                    and np.isclose(
                        path_delay_s,
                        robust.delay_s,
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                ),
            }
            for index, (path_delay_s, path_amplitude) in enumerate(
                zip(robust.path_delays_s, robust.path_amplitudes)
            )
        ],
    }


def _latest_acceptance_input_mtime() -> tuple[float, Path | None]:
    candidates: list[Path] = []
    for directory in (
        PROJECT_ROOT / "rf_ranging_model",
        PROJECT_ROOT / "validation",
        PROJECT_ROOT / "tests",
    ):
        if directory.is_dir():
            candidates.extend(directory.rglob("*.py"))
    candidates.extend(
        path
        for path in (
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "README.md",
            STATIC_ROOT / "index.html",
            STATIC_ROOT / "app.js",
            STATIC_ROOT / "styles.css",
        )
        if path.is_file()
    )
    if not candidates:
        return 0.0, None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.stat().st_mtime, latest


def load_acceptance_summary() -> dict[str, Any]:
    if not ACCEPTANCE_JSON.exists():
        return {"available": False, "message": "尚未运行独立验收。"}
    try:
        payload = json.loads(ACCEPTANCE_JSON.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "message": f"验收结果不可读：{exc}",
        }
    if not isinstance(payload, dict) or not isinstance(
        payload.get("checks"), list
    ) or not all(isinstance(check, dict) for check in payload.get("checks", [])):
        return {
            "available": False,
            "message": "验收结果 schema 无效：缺少 checks 列表。",
        }
    checks = payload.get("checks", [])
    schema_shape_valid = bool(
        all(
            isinstance(payload.get(field), str)
            for field in (
                "generated_at",
                "overall_status",
                "software_gate_status",
                "standards_status",
            )
        )
        and isinstance(payload.get("status_counts"), dict)
        and all(
            all(key in check for key in ("check_id", "title", "status", "criterion"))
            for check in checks
        )
    )
    schema_valid = bool(
        payload.get("schema_version") == ACCEPTANCE_SCHEMA_VERSION
        and schema_shape_valid
    )
    latest_input_mtime, latest_input = _latest_acceptance_input_mtime()
    result_mtime = ACCEPTANCE_JSON.stat().st_mtime
    fresh = schema_valid and result_mtime >= latest_input_mtime
    stale_reasons: list[str] = []
    if not schema_valid:
        stale_reasons.append(
            "验收 schema 已过期，必须重新运行当前验收套件"
        )
    if result_mtime < latest_input_mtime:
        stale_reasons.append(
            "源码晚于验收结果，当前 PASS 不得沿用"
        )
    recorded_overall = payload.get("overall_status")
    recorded_software = payload.get("software_gate_status")
    recorded_standards = payload.get("standards_status")
    return {
        "available": True,
        "schema_version": payload.get("schema_version"),
        "expected_schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "schema_valid": schema_valid,
        "fresh": fresh,
        "message": "；".join(stale_reasons) if stale_reasons else "验收结果为当前版本。",
        "generated_at": payload.get("generated_at"),
        "result_file_mtime": datetime.fromtimestamp(
            result_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "latest_input_mtime": datetime.fromtimestamp(
            latest_input_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "latest_input": (
            str(latest_input.relative_to(PROJECT_ROOT))
            if latest_input is not None
            else None
        ),
        "overall_status": recorded_overall if fresh else "STALE",
        "software_gate_status": recorded_software if fresh else "STALE",
        "standards_status": recorded_standards if fresh else "STALE",
        "recorded_overall_status": recorded_overall,
        "recorded_software_gate_status": recorded_software,
        "recorded_standards_status": recorded_standards,
        "status_counts": payload.get("status_counts", {}),
        "checks": [
            {
                "check_id": check.get("check_id"),
                "title": check.get("title"),
                "status": check.get("status"),
                "criterion": check.get("criterion"),
            }
            for check in checks
        ],
    }


class RangingRequestHandler(BaseHTTPRequestHandler):
    server_version = "RFRangingLocal/1.0"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, relative_name: str) -> None:
        allowed = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        content_type = allowed.get(relative_name)
        if content_type is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = STATIC_ROOT / relative_name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_static("index.html")
        elif path == "/app.js":
            self._send_static("app.js")
        elif path == "/styles.css":
            self._send_static("styles.css")
        elif path == "/api/health":
            self._send_json({"status": "ok", "service": "rf-ranging-local"})
        elif path == "/api/acceptance":
            self._send_json(load_acceptance_summary())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - standard-library handler API
        if urlparse(self.path).path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "invalid Content-Length"}, 400)
            return
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self._send_json({"error": "request body size is invalid"}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            response = run_scenario(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json({"error": f"invalid JSON: {exc}"}, 400)
            return
        except ScenarioInputError as exc:
            self._send_json({"error": str(exc)}, 422)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 422)
            return
        self._send_json(response)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local RF ranging graphical platform."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the graphical platform in the default browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must lie in [1, 65535]")
    server = ThreadingHTTPServer(
        (args.host, args.port), RangingRequestHandler
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"RF Ranging graphical platform: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping graphical platform.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
