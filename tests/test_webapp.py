from __future__ import annotations

import json
import math
import re
import tempfile
import threading
import unittest
from collections import Counter
from dataclasses import replace
from html.parser import HTMLParser
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from rf_ranging_model import webapp
from rf_ranging_model.webapp import ScenarioInputError, run_scenario


class _HTMLIDCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def _collect(self, attributes: list[tuple[str, str | None]]) -> None:
        for name, value in attributes:
            if name == "id" and value is not None:
                self.ids.append(value)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self._collect(attrs)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self._collect(attrs)


class WebPlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._log_patch = patch.object(
            webapp.RangingRequestHandler,
            "log_message",
            lambda self, format, *args: None,
        )
        cls._log_patch.start()
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), webapp.RangingRequestHandler
        )
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever,
            name="webapp-contract-test-server",
            daemon=True,
        )
        cls.server_thread.start()
        cls.host, cls.port = cls.server.server_address[:2]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5.0)
        cls._log_patch.stop()
        super().tearDownClass()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        declared_length: int | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection(self.host, self.port, timeout=20.0)
        try:
            connection.putrequest(method, path)
            request_headers = dict(headers or {})
            if declared_length is not None:
                request_headers["Content-Length"] = str(declared_length)
            elif body is not None:
                request_headers["Content-Length"] = str(len(body))
            for name, value in request_headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            if body:
                connection.send(body)
            response = connection.getresponse()
            response_body = response.read()
            return (
                response.status,
                {name.lower(): value for name, value in response.getheaders()},
                response_body,
            )
        finally:
            connection.close()

    def _json_request(
        self, payload: object
    ) -> tuple[int, dict[str, str], bytes]:
        return self._request(
            "POST",
            "/api/run",
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def _strict_json(self, body: bytes) -> Any:
        def reject_non_standard_constant(value: str) -> None:
            raise AssertionError(f"non-standard JSON constant: {value}")

        return json.loads(
            body.decode("utf-8"),
            parse_constant=reject_non_standard_constant,
        )

    def _assert_finite_json_tree(self, value: Any, path: str = "$") -> None:
        if isinstance(value, float):
            self.assertTrue(math.isfinite(value), path)
        elif isinstance(value, dict):
            for key, child in value.items():
                self._assert_finite_json_tree(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._assert_finite_json_tree(child, f"{path}[{index}]")

    def test_default_scenario_returns_chart_ready_finite_results(self) -> None:
        first = run_scenario({})
        second = run_scenario({})
        self.assertEqual(first["scenario"]["path_count"], 3)
        self.assertEqual(first["scenario"]["direct_path_gain"], 1.0)
        self.assertEqual(first["scenario"]["uwb_frame_count"], 4)
        self.assertGreater(first["scenario"]["uwb_frame_coherence"], 0.0)
        self.assertLessEqual(first["scenario"]["uwb_frame_coherence"], 1.0)
        self.assertEqual(first["summary"], second["summary"])
        self.assertGreater(len(first["series"]["cir_delay_ns"]), 100)
        self.assertLessEqual(len(first["series"]["cir_delay_ns"]), 700)
        self.assertEqual(
            len(first["series"]["ble_frequency_mhz"]), 72
        )
        self.assertEqual(
            len(first["series"]["ble_frequency_mhz"]),
            len(first["series"]["ble_raw_magnitude_db"]),
        )
        self.assertEqual(
            len(first["series"]["ble_frequency_mhz"]),
            len(first["series"]["ble_calibrated_magnitude_db"]),
        )
        self.assertEqual(
            len(first["series"]["ble_frequency_mhz"]),
            len(first["series"]["ble_phase_rad"]),
        )
        self.assertEqual(
            len(first["series"]["cir_delay_ns"]),
            len(first["series"]["cir_normalized_magnitude"]),
        )
        self.assertEqual(len(first["paths"]), first["scenario"]["path_count"])
        self.assertEqual(
            len(first["ble_estimated_paths"]),
            first["summary"]["ble"]["model_order"],
        )

    def test_calibration_is_explicit_and_separate_from_injection(self) -> None:
        uncalibrated = run_scenario(
            {
                "reflection_count": 0,
                "snr_db": 120,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "group_delay_ripple_ns": 0,
                "base_group_delay_ns": 1.0,
                "calibration_enabled": False,
                "calibration_delay_ns": 0.0,
            }
        )
        calibrated = run_scenario(
            {
                "reflection_count": 0,
                "snr_db": 120,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "group_delay_ripple_ns": 0,
                "base_group_delay_ns": 1.0,
                "calibration_enabled": True,
                "calibration_delay_ns": 1.0,
            }
        )
        raw = uncalibrated["summary"]["ble"]["calibrated_distance_m"]
        corrected = calibrated["summary"]["ble"]["calibrated_distance_m"]
        self.assertGreater(raw - corrected, 0.29)
        self.assertLess(raw - corrected, 0.31)

    def test_guided_visual_examples_run_real_models_with_expected_states(self) -> None:
        no_calibration = {
            "base_group_delay_ns": 0,
            "group_delay_ripple_ns": 0,
            "amplitude_ripple_db": 0,
            "calibration_mode": "none",
            "calibration_delay_ns": 0,
            "calibration_ripple_ns": 0,
            "calibration_amplitude_ripple_db": 0,
        }
        cases = {
            "los": {
                **no_calibration,
                "distance_m": 8,
                "direct_path_gain": 1,
                "reflection_count": 0,
                "reflection_gain": 0,
                "reflection_delay_ns": 6,
                "reflection_phase_rad": 0,
                "snr_db": 60,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "uwb_frame_count": 4,
                "seed": 7,
            },
            "ble_multipath": {
                "distance_m": 12,
                "direct_path_gain": 1,
                "reflection_count": 4,
                "reflection_gain": 1.05,
                "reflection_delay_ns": 3,
                "reflection_phase_rad": 2.8,
                "snr_db": 25,
                "phase_noise_deg": 1.2,
                "cfo_hz": 30,
                "uwb_frame_count": 4,
                "base_group_delay_ns": 0.35,
                "group_delay_ripple_ns": 0.12,
                "amplitude_ripple_db": 0.5,
                "calibration_mode": "frequency_profile",
                "calibration_delay_ns": 0.35,
                "calibration_ripple_ns": 0.12,
                "calibration_amplitude_ripple_db": 0.5,
                "seed": 19,
            },
            "uwb_late_single": {
                **no_calibration,
                "distance_m": 8,
                "direct_path_gain": 0.1,
                "reflection_count": 1,
                "reflection_gain": 1,
                "reflection_delay_ns": 10,
                "reflection_phase_rad": 0,
                "snr_db": 5,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "uwb_frame_count": 1,
                "seed": 0,
            },
            "uwb_late_four": {
                **no_calibration,
                "distance_m": 8,
                "direct_path_gain": 0.1,
                "reflection_count": 1,
                "reflection_gain": 1,
                "reflection_delay_ns": 10,
                "reflection_phase_rad": 0,
                "snr_db": 5,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "uwb_frame_count": 4,
                "seed": 0,
            },
            "low_coherence": {
                **no_calibration,
                "distance_m": 8,
                "direct_path_gain": 1,
                "reflection_count": 0,
                "reflection_gain": 0,
                "reflection_delay_ns": 10,
                "reflection_phase_rad": 0,
                "snr_db": -10,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "uwb_frame_count": 4,
                "seed": 41,
            },
            "bad_calibration": {
                **no_calibration,
                "distance_m": 0.5,
                "direct_path_gain": 1,
                "reflection_count": 0,
                "reflection_gain": 0,
                "reflection_delay_ns": 10,
                "reflection_phase_rad": 0,
                "snr_db": 120,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "uwb_frame_count": 4,
                "calibration_mode": "scalar",
                "calibration_delay_ns": 20,
                "seed": 7,
            },
        }
        results: dict[str, dict[str, Any]] = {}
        for name, example in cases.items():
            with self.subTest(example=name):
                result = run_scenario(example)
                self._assert_finite_json_tree(result)
                json.dumps(result, allow_nan=False)
                self.assertEqual(len(result["paths"]), result["scenario"]["path_count"])
                results[name] = result

        los = results["los"]
        self.assertLess(abs(los["summary"]["ble"]["error_m"]), 0.01)
        self.assertLess(abs(los["summary"]["uwb"]["error_m"]), 0.01)

        multipath = results["ble_multipath"]["summary"]["ble"]
        self.assertEqual(multipath["estimator_mode"], "multipath_matrix_pencil")
        self.assertLess(
            abs(multipath["candidate_distance_m"] - 12.0),
            abs(multipath["legacy_single_path_distance_m"] - 12.0),
        )

        single = results["uwb_late_single"]["summary"]["uwb"]
        four = results["uwb_late_four"]["summary"]["uwb"]
        self.assertEqual(single["status"], "PASS")
        self.assertGreater(abs(single["error_m"]), 2.5)
        self.assertEqual(four["status"], "LOW_CONFIDENCE")
        self.assertLess(abs(four["error_m"]), 0.1)

        low_coherence = results["low_coherence"]
        self.assertLess(low_coherence["scenario"]["uwb_frame_coherence"], 0.75)
        self.assertEqual(
            low_coherence["summary"]["uwb"]["rejection_reason"],
            "LOW_FRAME_COHERENCE",
        )

        bad = results["bad_calibration"]["summary"]
        self.assertEqual(bad["ble"]["status"], "REJECTED")
        self.assertEqual(bad["uwb"]["status"], "NO_PATH")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ScenarioInputError):
            run_scenario({"distance_m": 0.0})
        with self.assertRaises(ScenarioInputError):
            run_scenario({"reflection_count": 1.5})
        with self.assertRaises(ScenarioInputError):
            run_scenario({"calibration_enabled": "yes"})
        with self.assertRaises(ScenarioInputError):
            run_scenario({"uwb_frame_count": 0})
        with self.assertRaises(ScenarioInputError):
            run_scenario({"uwb_frame_count": 2.5})
        for invalid_gain in (0.0, 1.51, True, math.nan, math.inf):
            with self.subTest(direct_path_gain=invalid_gain):
                with self.assertRaises(ScenarioInputError):
                    run_scenario({"direct_path_gain": invalid_gain})
        with self.assertRaises(ScenarioInputError):
            run_scenario({"phase_noise_deg": 30.1})

    def test_get_health_static_assets_and_unknown_path_contract(self) -> None:
        status, headers, body = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        self.assertEqual(
            self._strict_json(body),
            {"status": "ok", "service": "rf-ranging-local"},
        )

        expected_types = {
            "/": "text/html",
            "/index.html": "text/html",
            "/app.js": "text/javascript",
            "/styles.css": "text/css",
        }
        for path, content_type in expected_types.items():
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["content-type"].startswith(content_type))
                self.assertGreater(len(body), 1000)

        status, _, _ = self._request("GET", "/not-a-route")
        self.assertEqual(status, 404)
        status, _, _ = self._request(
            "POST", "/not-a-route", b"", declared_length=0
        )
        self.assertEqual(status, 404)

    def test_post_run_default_frequency_profile_contract_is_strict_json(self) -> None:
        status, headers, body = self._json_request({})
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        payload = self._strict_json(body)
        self._assert_finite_json_tree(payload)

        self.assertEqual(payload["scenario"]["calibration_mode"], "frequency_profile")
        self.assertEqual(payload["scenario"]["uwb_frame_count"], 4)
        self.assertIsInstance(payload["scenario"]["uwb_frame_coherence"], float)
        calibration = payload["summary"]["rfic_calibration"]
        self.assertEqual(calibration["mode"], "frequency_profile")
        self.assertTrue(calibration["demo_only"])
        for profile_key in ("ble_profile", "uwb_profile"):
            profile = calibration[profile_key]
            self.assertIsInstance(profile, dict)
            self.assertIsInstance(profile["profile_id"], str)
            self.assertTrue(profile["profile_id"])

        for radio in ("ble", "uwb"):
            summary = payload["summary"][radio]
            for field in (
                "raw_distance_m",
                "candidate_distance_m",
                "calibrated_distance_m",
            ):
                self.assertIn(field, summary)
            self.assertIsInstance(summary["raw_distance_m"], float)
            self.assertIsInstance(summary["candidate_distance_m"], float)

        ble = payload["summary"]["ble"]
        self.assertEqual(ble["status"], "AVAILABLE")
        self.assertTrue(ble["available"])
        self.assertEqual(ble["candidate_source"], "ble_robust_calibrated")
        self.assertIsInstance(ble["calibrated_distance_m"], float)
        self.assertIsInstance(ble["quality_flags"], list)

    def test_available_ble_without_finite_candidate_is_rejected_as_invalid(self) -> None:
        original_result = webapp.RangingPipeline().run(
            webapp._build_paths(8.0, 0, 0.0, 6.0, 0.0),
            webapp.allowed_cs_frequencies_hz(),
            np.linspace(3.5e9, 4.5e9, 1001),
            seed=7,
        )
        broken_result = replace(
            original_result,
            ble_robust_calibrated_distance_m=None,
        )
        with patch.object(webapp, "RangingPipeline") as pipeline_class:
            pipeline_class.return_value.run.return_value = broken_result
            with self.assertRaisesRegex(
                ValueError, "no finite calibrated candidate"
            ):
                run_scenario({"calibration_mode": "none"})

    def test_post_run_rejected_ble_status_is_dynamic_not_hard_coded(self) -> None:
        status, _, body = self._json_request({"snr_db": -10})
        self.assertEqual(status, 200)
        payload = self._strict_json(body)
        self._assert_finite_json_tree(payload)
        ble = payload["summary"]["ble"]
        self.assertEqual(ble["status"], "REJECTED")
        self.assertFalse(ble["available"])
        self.assertEqual(ble["rejection_reason"], "ambiguous_delay_peaks")
        self.assertIsNone(ble["calibrated_distance_m"])
        self.assertIn("ambiguous_delay_peaks", ble["quality_flags"])
        self.assertIn("phase_residual_too_high", ble["quality_flags"])
        self.assertIn("confidence_below_threshold", ble["quality_flags"])

    def test_gross_scalar_calibration_is_rejected_not_displayed_as_range(self) -> None:
        status, _, body = self._json_request(
            {
                "distance_m": 0.5,
                "reflection_count": 0,
                "snr_db": 120,
                "phase_noise_deg": 0,
                "cfo_hz": 0,
                "base_group_delay_ns": 0,
                "group_delay_ripple_ns": 0,
                "amplitude_ripple_db": 0,
                "calibration_mode": "scalar",
                "calibration_delay_ns": 20,
            }
        )
        self.assertEqual(status, 200)
        payload = self._strict_json(body)
        ble = payload["summary"]["ble"]
        uwb = payload["summary"]["uwb"]
        self.assertEqual(ble["status"], "REJECTED")
        self.assertEqual(
            ble["rejection_reason"], "calibrated_distance_out_of_range"
        )
        self.assertLess(ble["candidate_distance_m"], 0.0)
        self.assertIsNone(ble["calibrated_distance_m"])
        self.assertEqual(uwb["status"], "NO_PATH")
        self.assertEqual(
            uwb["rejection_reason"],
            "CALIBRATION_EXCEEDS_DETECTED_DELAY",
        )
        self.assertIsNone(uwb["calibrated_distance_m"])

    def test_post_run_error_status_contract(self) -> None:
        status, _, body = self._request(
            "POST",
            "/api/run",
            b"{this is not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", self._strict_json(body)["error"])

        status, _, body = self._json_request({"distance_m": 0})
        self.assertEqual(status, 422)
        self.assertIn("distance_m", self._strict_json(body)["error"])

        status, _, body = self._request(
            "POST", "/api/run", b"", declared_length=0
        )
        self.assertEqual(status, 413)
        self.assertIn("body size", self._strict_json(body)["error"])

        status, _, body = self._request(
            "POST",
            "/api/run",
            b"",
            declared_length=webapp.MAX_REQUEST_BYTES + 1,
        )
        self.assertEqual(status, 413)
        self.assertIn("body size", self._strict_json(body)["error"])

    def test_acceptance_summary_freshness_and_schema_contract(self) -> None:
        acceptance_payload = {
            "schema_version": webapp.ACCEPTANCE_SCHEMA_VERSION,
            "generated_at": "2026-08-22T12:00:00+08:00",
            "overall_status": "PARTIAL",
            "software_gate_status": "PASS",
            "standards_status": "PARTIAL",
            "status_counts": {"PASS": 1, "PARTIAL": 1},
            "checks": [
                {
                    "check_id": "WEB-TEST",
                    "title": "Synthetic contract fixture",
                    "status": "PASS",
                    "criterion": "deterministic temporary fixture",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            acceptance_path = Path(temporary_directory) / "acceptance.json"
            acceptance_path.write_text(
                json.dumps(acceptance_payload), encoding="utf-8"
            )
            result_mtime = acceptance_path.stat().st_mtime

            with patch.object(webapp, "ACCEPTANCE_JSON", acceptance_path), patch.object(
                webapp,
                "_latest_acceptance_input_mtime",
                return_value=(result_mtime - 10.0, None),
            ):
                fresh = webapp.load_acceptance_summary()
            self.assertTrue(fresh["available"])
            self.assertTrue(fresh["schema_valid"])
            self.assertTrue(fresh["fresh"])
            self.assertEqual(fresh["software_gate_status"], "PASS")
            self.assertEqual(fresh["overall_status"], "PARTIAL")

            newer_source = webapp.PROJECT_ROOT / "rf_ranging_model" / "webapp.py"
            with patch.object(webapp, "ACCEPTANCE_JSON", acceptance_path), patch.object(
                webapp,
                "_latest_acceptance_input_mtime",
                return_value=(result_mtime + 10.0, newer_source),
            ):
                stale = webapp.load_acceptance_summary()
            self.assertTrue(stale["available"])
            self.assertTrue(stale["schema_valid"])
            self.assertFalse(stale["fresh"])
            self.assertEqual(stale["overall_status"], "STALE")
            self.assertEqual(stale["software_gate_status"], "STALE")
            self.assertEqual(stale["standards_status"], "STALE")
            self.assertEqual(stale["recorded_software_gate_status"], "PASS")
            self.assertIn("源码晚于验收结果", stale["message"])

    def test_static_html_ids_are_unique_and_cover_javascript_references(self) -> None:
        html_text = (webapp.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (webapp.STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        collector = _HTMLIDCollector()
        collector.feed(html_text)
        counts = Counter(collector.ids)
        duplicate_ids = sorted(
            identifier for identifier, count in counts.items() if count > 1
        )
        self.assertEqual(duplicate_ids, [])

        referenced_ids = set(
            re.findall(
                r"getElementById\(\s*[\"']([^\"']+)[\"']\s*\)",
                javascript,
            )
        )
        missing_ids = sorted(referenced_ids - set(collector.ids))
        self.assertTrue(referenced_ids)
        self.assertEqual(missing_ids, [])
        self.assertTrue(
            {
                "exampleGrid",
                "exampleName",
                "exampleExpected",
                "exampleObserved",
                "distanceChart",
                "pathChart",
                "bleConfidence",
            }
            <= set(collector.ids)
        )
        example_names = set(
            re.findall(r'data-example="([^"]+)"', html_text)
        )
        self.assertEqual(
            example_names,
            {
                "los",
                "ble_multipath",
                "uwb_late_single",
                "uwb_late_four",
                "low_coherence",
                "bad_calibration",
            },
        )

    def test_static_assets_exist_and_no_path_values_encode_as_null(self) -> None:
        for name in ("index.html", "styles.css", "app.js"):
            path = webapp.STATIC_ROOT / name
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 1000)

        self.assertIsNone(webapp._finite_or_none(math.nan))
        self.assertIsNone(webapp._finite_or_none(math.inf))
        encoded = json.dumps(
            {"detected_delay_ns": webapp._finite_or_none(math.nan)},
            allow_nan=False,
        )
        self.assertIsNone(json.loads(encoded)["detected_delay_ns"])


if __name__ == "__main__":
    unittest.main()
