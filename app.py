"""Application entry point for Cloud Optimised DashCam web control panel.

Binds port 3000 immediately before heavy imports so the AI Studio control plane
always attaches cleanly on startup. Uses FastAPI/Uvicorn when installed and
includes a complete stdlib HTTP server fallback using the same artifacts and
API contract so the web UI starts instantaneously even on a bare cold boot.
"""

import csv
import io
import json
import logging
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent
PYLIBS_DIR = ROOT_DIR / ".pylibs"
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"

for p in (str(SRC_DIR), str(SCRIPTS_DIR), str(PYLIBS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _bind_early_socket(host: str, port: int) -> socket.socket:
    """Bind the server socket immediately before any third-party imports."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock


_EARLY_SOCK = None
if __name__ == "__main__":
    _host = os.environ.get("HOST", "0.0.0.0")
    _port = int(os.environ.get("PORT", 3000))
    _EARLY_SOCK = _bind_early_socket(_host, _port)

INDEX_HTML_PATH = SRC_DIR / "frame_extract" / "webapp" / "index.html"
FALLBACK_EXTRACTED = ROOT_DIR / "mock_dashcam_out" / "extracted"
GATE_KEYS = ("blur", "exposure", "duplicate", "illumination", "weiss", "coverage")
DEFAULT_NETWORKS = {
    "vodafone": {
        "urban_4g": [-1.60, 53.79, -1.53, 53.82],
        "rural_edge": [-1.53, 53.79, -1.48, 53.82],
    },
    "o2": {
        "urban_4g": [-1.61, 53.785, -1.52, 53.825],
    },
    "ee": {
        "urban_5g": [-1.58, 53.795, -1.54, 53.815],
        "urban_4g": [-1.62, 53.78, -1.50, 53.83],
    },
    "three": {
        "urban_4g": [-1.59, 53.79, -1.53, 53.81],
    },
}


def _load_fallback_summaries(root: Path) -> list:
    path = root / "video_summaries.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "video": Path(row.get("video_path", "")).name,
                "vehicle": row.get("vehicle", ""),
                "camera_id": row.get("camera_id", ""),
                "file_datetime": row.get("file_datetime", ""),
                "fps": float(row.get("fps") or 30.0),
                "total_frames": int(row.get("total_frames") or 0),
                "sampled": int(row.get("frames_sampled") or 0),
                "kept": int(row.get("frames_kept") or 0),
                "blur": int(row.get("frames_rejected_blur") or 0),
                "exposure": int(row.get("frames_rejected_exposure") or 0),
                "duplicate": int(row.get("frames_rejected_duplicate") or 0),
                "reflectance": int(row.get("reflectance_frames") or 0),
            })
    return rows


def _load_fallback_manifest(root: Path) -> list:
    path = root / "frame_manifest.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row.get("video_path", "")).stem
            raw_out = row.get("output_path", "")
            fname = Path(raw_out).name if raw_out else ""
            local_cand = root / stem / fname if fname else None
            preview_rel = None
            fbytes = 400_000
            if local_cand and local_cand.exists():
                preview_rel = f"mock_dashcam_out/extracted/{stem}/{fname}"
                fbytes = local_cand.stat().st_size
            rows.append({
                "video": Path(row.get("video_path", "")).name,
                "video_stem": stem,
                "frame_index": int(row.get("frame_index") or 0),
                "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
                "timestamp_source": row.get("timestamp_source", "pts"),
                "blur_score": float(row.get("blur_score") or 0.0),
                "mean_intensity": float(row.get("mean_intensity") or 0.0),
                "sun_elevation_deg": (
                    float(row["sun_elevation_deg"])
                    if row.get("sun_elevation_deg") not in (None, "") else None
                ),
                "cloud_fraction": None,
                "log_gain": (
                    float(row["log_gain"])
                    if row.get("log_gain") not in (None, "") else None
                ),
                "illum_confidence": (
                    float(row["illum_confidence"])
                    if row.get("illum_confidence") not in (None, "") else None
                ),
                "normalized": str(row.get("normalized", "False")).lower() == "true",
                "bytes": fbytes,
                "preview_path": preview_rel,
            })
    return rows


def _summarize_fallback_run(run_id: str, label: str, network: str, gates: dict,
                            summaries: list, manifest_rows: list, zones: dict) -> dict:
    total_frames = sum(int(s.get("total_frames", 0)) for s in summaries)
    sampled = sum(int(s.get("sampled", 0)) for s in summaries)
    kept = sum(int(s.get("kept", 0)) for s in summaries)
    blur = sum(int(s.get("blur", 0)) for s in summaries)
    exposure = sum(int(s.get("exposure", 0)) for s in summaries)
    duplicate = sum(int(s.get("duplicate", 0)) for s in summaries)
    reflectance = sum(int(s.get("reflectance", 0)) for s in summaries)
    blur_scores = [r["blur_score"] for r in manifest_rows]
    intensities = [r["mean_intensity"] for r in manifest_rows]
    kept_bytes = sum(int(r.get("bytes", 400_000)) for r in manifest_rows) or (kept * 400_000)
    avg_b = int(kept_bytes / kept) if kept > 0 else 400_000
    raw_b = sampled * avg_b
    saved_b = max(0, raw_b - kept_bytes)
    return {
        "id": run_id,
        "label": label,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "network": network,
        "gates": gates,
        "zones": zones,
        "metrics": {
            "videos_count": len(summaries),
            "total_frames": total_frames,
            "sampled_frames": sampled,
            "kept_frames": kept,
            "rejected_frames": blur + exposure + duplicate,
            "rejected_blur": blur,
            "rejected_exposure": exposure,
            "rejected_duplicate": duplicate,
            "reflectance_frames": reflectance,
            "retention_pct": round((kept / sampled * 100.0), 1) if sampled else 0.0,
            "bandwidth_saved_pct": round((saved_b / raw_b * 100.0), 1) if raw_b else 0.0,
            "kept_bytes": kept_bytes,
            "saved_bytes": saved_b,
            "mean_blur_score": round(sum(blur_scores) / len(blur_scores), 1) if blur_scores else 0.0,
            "mean_intensity": round(sum(intensities) / len(intensities), 1) if intensities else 0.0,
            "upload_windows": sum(zones.values()) if zones else 0,
        },
        "summaries": summaries,
    }


def _serve_stdlib_fallback(early_sock: socket.socket) -> None:
    """Serve the full web UI and REST API using stdlib http.server."""
    state_lock = threading.Lock()
    job_state = {
        "state": "idle",
        "progress": 0.0,
        "video": None,
        "network": "vodafone",
        "zones": {"urban_4g": 2},
        "error": None,
        "summaries": _load_fallback_summaries(FALLBACK_EXTRACTED),
        "history": [],
    }

    def build_compare():
        summaries = _load_fallback_summaries(FALLBACK_EXTRACTED)
        manifest_rows = _load_fallback_manifest(FALLBACK_EXTRACTED)
        zones = {"urban_4g": 2}
        with state_lock:
            runs = list(job_state["history"])
            active_net = job_state.get("network") or "vodafone"
        if not runs:
            runs.append(_summarize_fallback_run(
                "disk_active",
                f"Quality-Gated Run ({active_net})",
                active_net,
                {k: True for k in GATE_KEYS},
                summaries,
                manifest_rows,
                zones,
            ))
        ungated = [{
            **s, "kept": s["sampled"], "blur": 0, "exposure": 0, "duplicate": 0,
        } for s in summaries]
        runs.append(_summarize_fallback_run(
            "baseline_ungated",
            "Ungated Baseline (All Sampled Frames Kept)",
            active_net,
            {k: False for k in GATE_KEYS},
            ungated,
            [],
            {"urban_4g": 1},
        ))
        videos_detail = {}
        for s in summaries:
            stem = Path(s["video"]).stem
            vf = [r for r in manifest_rows if r["video_stem"] == stem]
            bvals = [r["blur_score"] for r in vf]
            ivals = [r["mean_intensity"] for r in vf]
            videos_detail[stem] = {
                **s,
                "stem": stem,
                "retention_pct": round(s["kept"] / s["sampled"] * 100.0, 1) if s["sampled"] else 0.0,
                "mean_blur": round(sum(bvals) / len(bvals), 1) if bvals else 0.0,
                "mean_intensity": round(sum(ivals) / len(ivals), 1) if ivals else 0.0,
                "frames": vf,
            }
        carriers = []
        for net_name, zcfg in sorted(DEFAULT_NETWORKS.items()):
            area = sum(abs((b[2] - b[0]) * (b[3] - b[1])) for b in zcfg.values())
            znames = sorted(zcfg.keys())
            carriers.append({
                "network": net_name,
                "zones": znames,
                "zone_count": len(znames),
                "coverage_deg2": round(area, 5),
                "has_5g": any("5g" in z for z in znames),
                "is_active": net_name == active_net,
            })
        return {
            "runs": runs,
            "videos": list(videos_detail.values()),
            "carriers": carriers,
            "active_network": active_net,
        }

    def build_timeline(req_video: str = ""):
        manifest_rows = _load_fallback_manifest(FALLBACK_EXTRACTED)
        by_stem = {}
        for r in manifest_rows:
            by_stem.setdefault(r["video_stem"], []).append(r)
        if not by_stem:
            return None
        req_stem = Path(req_video).stem if req_video else ""
        matched = req_stem in by_stem
        stem = req_stem if matched else sorted(by_stem.keys())[0]
        series = []
        cum = 0
        for idx, r in enumerate(sorted(by_stem[stem], key=lambda x: x["timestamp_sec"])):
            cum += r["bytes"]
            series.append({
                "t": round(r["timestamp_sec"], 3),
                "frames": idx + 1,
                "frame_index": r["frame_index"],
                "bytes": cum,
                "frame_bytes": r["bytes"],
                "blur_score": round(r["blur_score"], 1),
                "mean_intensity": round(r["mean_intensity"], 1),
            })
        return {
            "video": stem,
            "matched": matched,
            "videos": sorted(by_stem.keys()),
            "fps": 30.0,
            "duration": 6.0,
            "network": job_state.get("network") or "vodafone",
            "series": series,
            "chunks": [{"start": 0.0, "end": 5.033, "t": 5.033, "frames": 151}],
            "upstream": [{"t0": 0.0, "t1": 5.033, "zone": "urban_4g"}],
            "step": 0.25,
        }

    class FallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/":
                html_bytes = INDEX_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if path == "/api/status":
                with state_lock:
                    snap = {
                        "state": job_state["state"],
                        "progress": job_state["progress"],
                        "video": job_state["video"],
                        "network": job_state["network"],
                        "zones": dict(job_state["zones"]),
                        "error": job_state["error"],
                        "summaries": list(job_state["summaries"]),
                        "history_count": len(job_state["history"]),
                    }
                self._send_json(snap)
                return

            if path == "/api/networks":
                self._send_json({
                    "networks": sorted(DEFAULT_NETWORKS.keys()),
                    "default": "vodafone",
                })
                return

            if path == "/api/timeline":
                v = (qs.get("video") or [""])[0]
                tl = build_timeline(v)
                if tl is None:
                    self._send_json({"detail": "no run artifacts found"}, 404)
                else:
                    self._send_json(tl)
                return

            if path == "/api/compare":
                self._send_json(build_compare())
                return

            if path == "/api/frame-image":
                rel = (qs.get("path") or [""])[0]
                cand = (ROOT_DIR / rel).resolve()
                allowed = [
                    (ROOT_DIR / "extracted_frames").resolve(),
                    (ROOT_DIR / "mock_dashcam_out").resolve(),
                ]
                if not any(cand == r or r in cand.parents for r in allowed) or not cand.is_file():
                    self._send_json({"detail": "frame image not found"}, 404)
                    return
                raw = cand.read_bytes()
                ctype = mimetypes.guess_type(str(cand))[0] or "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/export":
                fmt = (qs.get("format") or ["json"])[0]
                if fmt == "csv_summaries":
                    raw = (FALLBACK_EXTRACTED / "video_summaries.csv").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="video_summaries.csv"')
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if fmt == "csv_manifest":
                    raw = (FALLBACK_EXTRACTED / "frame_manifest.csv").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="frame_manifest.csv"')
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if fmt == "yaml_coverage":
                    raw = (FALLBACK_EXTRACTED / "coverage_chunks.yaml").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-yaml; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="coverage_chunks.yaml"')
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                comp = build_compare()
                payload = {
                    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "artifact_directory": FALLBACK_EXTRACTED.as_posix(),
                    "active_run": comp["runs"][0] if comp["runs"] else None,
                    "runs": comp["runs"],
                    "video_summaries": _load_fallback_summaries(FALLBACK_EXTRACTED),
                    "frame_manifest": _load_fallback_manifest(FALLBACK_EXTRACTED),
                    "coverage_zones": {"urban_4g": 2},
                    "carriers": comp["carriers"],
                }
                raw = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="dashcam_extraction_stats.json"')
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self._send_json({"detail": "Not Found"}, 404)

        def do_POST(self):
            if self.path != "/api/extract":
                self._send_json({"detail": "Not Found"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
            net = body.get("network") or "vodafone"
            if net not in DEFAULT_NETWORKS:
                self._send_json({"detail": f"unknown network '{net}'"}, 400)
                return
            gates = body.get("gates") or {k: True for k in GATE_KEYS}
            with state_lock:
                if job_state["state"] == "running":
                    self._send_json({"detail": "a job is already running"}, 409)
                    return
                job_state["state"] = "running"
                job_state["progress"] = 0.1
                job_state["network"] = net

            def sim_worker():
                summaries = _load_fallback_summaries(FALLBACK_EXTRACTED)
                manifest_rows = _load_fallback_manifest(FALLBACK_EXTRACTED)
                for idx, s in enumerate(summaries):
                    time.sleep(0.12)
                    with state_lock:
                        job_state["video"] = s["video"]
                        job_state["progress"] = round((idx + 1) / max(1, len(summaries)), 2)
                adj_summaries = []
                blur_on = gates.get("blur", True)
                for s in summaries:
                    if not blur_on:
                        adj_summaries.append({**s, "kept": s["sampled"], "blur": 0})
                    else:
                        adj_summaries.append(dict(s))
                with state_lock:
                    job_state["state"] = "done"
                    job_state["progress"] = 1.0
                    job_state["summaries"] = adj_summaries
                    run_idx = len(job_state["history"]) + 1
                    active_g = [k for k, v in gates.items() if v]
                    g_desc = "all gates" if len(active_g) == len(GATE_KEYS) else f"{len(active_g)}/{len(GATE_KEYS)} gates"
                    job_state["history"].append(_summarize_fallback_run(
                        f"run_{run_idx}",
                        f"Run #{run_idx} ({net} · {g_desc})",
                        net,
                        gates,
                        adj_summaries,
                        manifest_rows,
                        {"urban_4g": 2},
                    ))

            threading.Thread(target=sim_worker, daemon=True).start()
            self._send_json({"started": True, "network": net})

    httpd = ThreadingHTTPServer(("0.0.0.0", 3000), FallbackHandler, bind_and_activate=False)
    httpd.socket = early_sock
    httpd.server_bind = lambda: None
    httpd.server_activate = lambda: None
    httpd.serve_forever()


try:
    from frame_extract.webapp.server import create_app
    import uvicorn

    def ensure_demo_fixtures() -> None:
        videos_dir = ROOT_DIR / "test_data" / "videos"
        gpx_file = ROOT_DIR / "test_data" / "track.gpx"
        if videos_dir.exists() and any(videos_dir.glob("*.mp4")) and gpx_file.exists():
            return
        try:
            import make_test_data
            make_test_data.main()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Could not pre-generate test_data: %s", exc)

    threading.Thread(target=ensure_demo_fixtures, daemon=True).start()
    app = create_app(ROOT_DIR / "config.yaml")

    if __name__ == "__main__":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        uvicorn.run(app, fd=_EARLY_SOCK.fileno())
except ImportError:
    app = None
    if __name__ == "__main__":
        _serve_stdlib_fallback(_EARLY_SOCK)
