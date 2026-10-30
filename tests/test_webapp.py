"""Web control panel API tests. Pipeline mocked; fully offline."""

import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from frame_extract.config import Settings
from frame_extract.webapp.server import (
    ExtractRequest, _apply_request, create_app,
)

CONFIG = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture
def client():
    return TestClient(create_app(CONFIG))


class TestStatusEndpoint:
    def test_initial_state_is_idle(self, client):
        body = client.get("/api/status").json()
        assert body["state"] == "idle"
        assert body["progress"] == 0.0

    def test_snapshot_carries_network_and_zones(self, client):
        body = client.get("/api/status").json()
        assert "network" in body and "zones" in body

    def test_index_serves_flow_screens(self, client):
        text = client.get("/").text
        for marker in ("Cloud Optimised DashCam", "Run Local Demo",
                       "Select Video File", "Processing Complete"):
            assert marker in text


class TestNetworksEndpoint:
    def test_networks_lists_default(self, client):
        body = client.get("/api/networks").json()
        assert body["default"] in body["networks"]


class TestExtractEndpoint:
    def test_launch_and_complete(self, client):
        with patch(
            "frame_extract.webapp.server.ExtractionPipeline"
        ) as pipeline_cls, patch(
            "frame_extract.webapp.server._read_zone_counts", return_value={},
        ):
            pipeline_cls.return_value.run.return_value = []
            res = client.post("/api/extract", json={"gates": {}})
            assert res.status_code == 200
            state = "running"
            deadline = time.time() + 5.0
            while time.time() < deadline:
                state = client.get("/api/status").json()["state"]
                if state in ("done", "failed"):
                    break
                time.sleep(0.05)
            assert state == "done"

    def test_concurrent_job_rejected(self, client):
        with patch(
            "frame_extract.webapp.server.ExtractionPipeline"
        ) as pipeline_cls, patch(
            "frame_extract.webapp.server._read_zone_counts", return_value={},
        ):
            pipeline_cls.return_value.run.side_effect = (
                lambda progress=None: time.sleep(0.5) or []
            )
            first = client.post("/api/extract", json={"gates": {}})
            second = client.post("/api/extract", json={"gates": {}})
        assert first.status_code == 200
        assert second.status_code == 409

    def test_bad_config_reports_failed(self, tmp_path):
        client = TestClient(create_app(tmp_path / "missing.yaml"))
        res = client.post("/api/extract", json={"gates": {}})
        assert res.status_code == 400
        assert client.get("/api/status").json()["state"] == "failed"

    def test_unknown_network_is_400(self, client):
        res = client.post(
            "/api/extract", json={"network": "carrier_pigeon", "gates": {}},
        )
        assert res.status_code == 400


class TestApplyRequest:
    def test_toggles_neutralise_thresholds(self):
        settings = Settings.load(CONFIG)
        req = ExtractRequest(gates={
            "blur": False, "exposure": False, "duplicate": False,
            "illumination": False, "weiss": False, "coverage": False,
        })
        out = _apply_request(settings, req)
        assert out.quality.blur_threshold == 0.0
        assert out.quality.max_mean_intensity == 256.0
        assert out.quality.duplicate_threshold == 0.0
        assert not out.illumination.enabled
        assert not out.intrinsic.enabled
        assert not out.coverage.enabled

    def test_toggle_cannot_enable_disabled_subsystem(self):
        settings = Settings.load(CONFIG)
        settings = replace(
            settings, intrinsic=replace(settings.intrinsic, enabled=False),
        )
        out = _apply_request(settings, ExtractRequest(gates={"weiss": True}))
        assert not out.intrinsic.enabled

    def test_gpx_source_maps_to_telemetry(self):
        settings = Settings.load(CONFIG)
        settings = replace(
            settings, telemetry=replace(
                settings.telemetry,
                gpx_path="local.gpx", gcs_bucket="b", gcs_blob="t.gpx",
            ),
        )
        out = _apply_request(
            settings, ExtractRequest(gpx_source="mock", gates={}),
        )
        assert out.telemetry.use_mock_gcs
        assert out.telemetry.gpx_path is None

    def test_network_selection_applied(self):
        settings = Settings.load(CONFIG)
        out = _apply_request(
            settings, ExtractRequest(network="o2", gates={}),
        )
        assert out.coverage.network == "o2"

    def test_sampling_choice_applied(self):
        settings = Settings.load(CONFIG)
        out = _apply_request(
            settings, ExtractRequest(sampling="frames_30", gates={}),
        )
        assert out.sampling.mode == "frames"
        assert out.sampling.every_n_frames == 30


class TestCompareAndExportEndpoints:
    def test_compare_returns_runs_videos_and_carriers(self, client):
        res = client.get("/api/compare")
        assert res.status_code == 200
        body = res.json()
        assert "runs" in body
        assert "videos" in body
        assert "carriers" in body
        assert len(body["runs"]) >= 1
        assert len(body["carriers"]) >= 1

    def test_export_json_and_csv_and_yaml(self, client):
        res_json = client.get("/api/export?format=json")
        assert res_json.status_code == 200
        payload = res_json.json()
        assert "video_summaries" in payload
        assert "frame_manifest" in payload
        assert "runs" in payload

        res_sum = client.get("/api/export?format=csv_summaries")
        assert res_sum.status_code == 200
        assert "text/csv" in res_sum.headers["content-type"]
        assert "video" in res_sum.text

        res_man = client.get("/api/export?format=csv_manifest")
        assert res_man.status_code == 200
        assert "text/csv" in res_man.headers["content-type"]
        assert "blur_score" in res_man.text

        res_yaml = client.get("/api/export?format=yaml_coverage")
        assert res_yaml.status_code == 200
        assert "zones" in res_yaml.text

    def test_index_contains_comparison_zoom_and_export_markers(self, client):
        text = client.get("/").text
        for marker in ("Comparison Mode", "Timeline Zoom", "Export Stats"):
            assert marker in text
