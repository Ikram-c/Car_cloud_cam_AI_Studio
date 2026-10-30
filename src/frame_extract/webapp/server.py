"""FastAPI web control panel: launch runs, watch progress, review results.

Serves a dark five-screen flow (welcome -> demo -> configure ->
processing -> complete) plus a footage-review player that draws a
semi-transparent processing-timeline overlay (chunk creation, data
upstream windows) on top of a posted video, driven by the artifacts
of the last pipeline run.

Gate toggles neutralise thresholds via ``dataclasses.replace`` (a
disabled gate can never fire) and AND with the YAML: the browser can
disable what config enables, never the reverse. One job runs at a
time; concurrent launches receive 409.
"""

import csv
import io
import json
import logging
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from ..config import Settings
from ..models import ProgressEvent, VideoSummary
from ..pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent / "index.html"

GATE_KEYS = ("blur", "exposure", "duplicate", "illumination", "weiss", "coverage")

# Neutral thresholds: values at which a gate can never fire.
NEUTRAL_BLUR_THRESHOLD = 0.0
NEUTRAL_MIN_INTENSITY = -1.0
NEUTRAL_MAX_INTENSITY = 256.0
NEUTRAL_DUPLICATE_THRESHOLD = 0.0

FALLBACK_FRAME_BYTES = 400_000
TIMELINE_STEP_SECONDS = 0.25


class ExtractRequest(BaseModel):
    """One extraction launch request from the browser."""

    videos: Optional[str] = None
    output: Optional[str] = None
    gpx_source: Optional[str] = None      # config | local | mock | none
    gpx_path: Optional[str] = None
    network: Optional[str] = None
    sampling: Optional[str] = None        # e.g. "seconds_1", "frames_30"
    gates: Dict[str, bool] = {}


def _apply_request(settings: Settings, req: ExtractRequest) -> Settings:
    """Fold a browser request into an immutable settings copy.

    Args:
        settings (Settings): Config-loaded baseline.
        req (ExtractRequest): Browser overrides and gate toggles.

    Returns:
        Settings: New settings with the request applied.

    Raises:
        ValueError: For an unknown network or malformed sampling.
    """
    out = settings.with_overrides(
        video_directory=Path(req.videos) if req.videos else None,
        output_directory=Path(req.output) if req.output else None,
    )
    if req.gpx_source == "mock":
        out = replace(out, telemetry=replace(
            out.telemetry,
            gpx_path=None,
            gcs_bucket=out.telemetry.gcs_bucket or "mock-bucket",
            gcs_blob=out.telemetry.gcs_blob or "telemetry/track.gpx",
            use_mock_gcs=True,
        ))
    elif req.gpx_source == "local" and req.gpx_path:
        out = replace(out, telemetry=replace(
            out.telemetry, gpx_path=req.gpx_path,
        ))
    elif req.gpx_source == "none":
        out = replace(out, telemetry=replace(
            out.telemetry, gpx_path=None, gcs_bucket=None, gcs_blob=None,
        ))
    if req.network is not None:
        if req.network not in out.coverage.networks:
            raise ValueError(
                f"unknown network '{req.network}'; "
                f"expected one of {sorted(out.coverage.networks)}"
            )
        out = replace(out, coverage=replace(out.coverage, network=req.network))
    if req.sampling is not None:
        out = replace(out, sampling=_parse_sampling(out, req.sampling))

    def enabled(key: str) -> bool:
        return req.gates.get(key, True)

    quality = out.quality
    if not enabled("blur"):
        quality = replace(quality, blur_threshold=NEUTRAL_BLUR_THRESHOLD)
    if not enabled("exposure"):
        quality = replace(
            quality,
            min_mean_intensity=NEUTRAL_MIN_INTENSITY,
            max_mean_intensity=NEUTRAL_MAX_INTENSITY,
        )
    if not enabled("duplicate"):
        quality = replace(quality, duplicate_threshold=NEUTRAL_DUPLICATE_THRESHOLD)
    return replace(
        out,
        quality=quality,
        illumination=replace(
            out.illumination,
            enabled=out.illumination.enabled and enabled("illumination"),
        ),
        intrinsic=replace(
            out.intrinsic,
            enabled=out.intrinsic.enabled and enabled("weiss"),
        ),
        coverage=replace(
            out.coverage,
            enabled=out.coverage.enabled and enabled("coverage"),
        ),
    )


def _parse_sampling(settings: Settings, choice: str):
    """Translate a browser sampling choice into a SamplingConfig.

    Args:
        settings (Settings): Baseline (for the untouched fields).
        choice (str): "<mode>_<value>", e.g. "frames_30", "seconds_2".

    Returns:
        SamplingConfig: The updated sampling section.

    Raises:
        ValueError: If the choice is malformed.
    """
    mode, _, value = choice.partition("_")
    if mode == "frames" and value.isdigit():
        return replace(
            settings.sampling, mode="frames", every_n_frames=int(value),
        )
    if mode == "seconds":
        try:
            return replace(
                settings.sampling, mode="seconds", every_n_seconds=float(value),
            )
        except ValueError as e:
            raise ValueError(f"bad sampling choice '{choice}'") from e
    raise ValueError(f"bad sampling choice '{choice}'")


def _read_zone_counts(output_root: Path, manifest_name: str) -> dict:
    """Read per-zone chunk counts from the coverage manifest.

    Args:
        output_root (Path): Pipeline output directory.
        manifest_name (str): Coverage manifest filename.

    Returns:
        dict: zone -> number of upload windows; {} when absent.
    """
    path = Path(output_root) / manifest_name
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}
    zones = payload.get("zones") or {}
    return {zone: len(ranges or []) for zone, ranges in zones.items()}


def _resolve_artifact_root(settings: Settings, last_output: Optional[Path] = None) -> Optional[Path]:
    """Locate the active or fallback extraction output directory."""
    candidates = []
    if last_output:
        candidates.append(Path(last_output))
    candidates.append(Path(settings.runtime.output_directory))
    candidates.append(Path("mock_dashcam_out/extracted"))
    for root in candidates:
        if (root / settings.output.manifest_name).exists() or (root / "video_summaries.csv").exists():
            return root
    return None


def _load_artifact_summaries(output_root: Path) -> List[dict]:
    """Read video_summaries.csv into the standard summary dict shape."""
    summaries_path = output_root / "video_summaries.csv"
    if not summaries_path.exists():
        return []
    rows = []
    try:
        with summaries_path.open(newline="", encoding="utf-8") as handle:
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
    except (OSError, ValueError) as e:
        logger.warning("Could not parse %s: %s", summaries_path, e)
    return rows


def _load_manifest_rows(output_root: Path, manifest_name: str) -> List[dict]:
    """Read frame_manifest.csv rows with resolved local preview paths."""
    manifest_path = output_root / manifest_name
    if not manifest_path.exists():
        return []
    rows = []
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video_stem = Path(row.get("video_path", "")).stem
                raw_out = row.get("output_path", "")
                filename = Path(raw_out).name if raw_out else ""
                local_candidate = output_root / video_stem / filename if filename else None
                preview_rel = None
                file_bytes = FALLBACK_FRAME_BYTES
                if local_candidate and local_candidate.exists():
                    preview_rel = f"{output_root.as_posix()}/{video_stem}/{filename}"
                    file_bytes = local_candidate.stat().st_size
                elif raw_out and Path(raw_out).exists():
                    preview_rel = Path(raw_out).as_posix()
                    file_bytes = Path(raw_out).stat().st_size
                rows.append({
                    "video": Path(row.get("video_path", "")).name,
                    "video_stem": video_stem,
                    "frame_index": int(row.get("frame_index") or 0),
                    "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
                    "timestamp_source": row.get("timestamp_source", "pts"),
                    "blur_score": float(row.get("blur_score") or 0.0),
                    "mean_intensity": float(row.get("mean_intensity") or 0.0),
                    "sun_elevation_deg": (
                        float(row["sun_elevation_deg"])
                        if row.get("sun_elevation_deg") not in (None, "") else None
                    ),
                    "cloud_fraction": (
                        float(row["cloud_fraction"])
                        if row.get("cloud_fraction") not in (None, "") else None
                    ),
                    "log_gain": (
                        float(row["log_gain"])
                        if row.get("log_gain") not in (None, "") else None
                    ),
                    "illum_confidence": (
                        float(row["illum_confidence"])
                        if row.get("illum_confidence") not in (None, "") else None
                    ),
                    "normalized": str(row.get("normalized", "False")).lower() == "true",
                    "bytes": file_bytes,
                    "preview_path": preview_rel,
                })
    except (OSError, ValueError) as e:
        logger.warning("Could not parse %s: %s", manifest_path, e)
    return rows


def _summarize_run(run_id: str, label: str, network: str, gates: dict,
                   summaries: List[dict], manifest_rows: List[dict],
                   zones: dict, timestamp: Optional[str] = None) -> dict:
    """Aggregate KPIs for a single run snapshot for Comparison Mode & Export."""
    total_frames = sum(int(s.get("total_frames", 0)) for s in summaries)
    sampled = sum(int(s.get("sampled", 0)) for s in summaries)
    kept = sum(int(s.get("kept", 0)) for s in summaries)
    blur = sum(int(s.get("blur", 0)) for s in summaries)
    exposure = sum(int(s.get("exposure", 0)) for s in summaries)
    duplicate = sum(int(s.get("duplicate", 0)) for s in summaries)
    reflectance = sum(int(s.get("reflectance", 0)) for s in summaries)

    blur_scores = [r["blur_score"] for r in manifest_rows if r.get("blur_score") is not None]
    intensities = [r["mean_intensity"] for r in manifest_rows if r.get("mean_intensity") is not None]
    kept_bytes = sum(int(r.get("bytes", FALLBACK_FRAME_BYTES)) for r in manifest_rows)
    if not kept_bytes and kept > 0:
        kept_bytes = kept * FALLBACK_FRAME_BYTES
    avg_frame_bytes = int(kept_bytes / kept) if kept > 0 else FALLBACK_FRAME_BYTES
    raw_sampled_bytes = sampled * avg_frame_bytes
    saved_bytes = max(0, raw_sampled_bytes - kept_bytes)

    return {
        "id": run_id,
        "label": label,
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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
            "bandwidth_saved_pct": round((saved_bytes / raw_sampled_bytes * 100.0), 1) if raw_sampled_bytes else 0.0,
            "kept_bytes": kept_bytes,
            "saved_bytes": saved_bytes,
            "mean_blur_score": round(sum(blur_scores) / len(blur_scores), 1) if blur_scores else 0.0,
            "mean_intensity": round(sum(intensities) / len(intensities), 1) if intensities else 0.0,
            "upload_windows": sum(zones.values()) if zones else 0,
        },
        "summaries": summaries,
    }


class JobState:
    """Single-job state machine guarded by one lock, with run history."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"
        self._progress = 0.0
        self._video: Optional[str] = None
        self._network: Optional[str] = None
        self._gates: Dict[str, bool] = {k: True for k in GATE_KEYS}
        self._zones: dict = {}
        self._error: Optional[str] = None
        self._summaries: List[dict] = []
        self._history: List[dict] = []

    def try_start(self, network: Optional[str], gates: Optional[Dict[str, bool]] = None) -> bool:
        """Atomically claim the single job slot."""
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._progress = 0.0
            self._video = None
            self._network = network
            self._gates = {k: (gates or {}).get(k, True) for k in GATE_KEYS}
            self._zones = {}
            self._error = None
            self._summaries = []
            return True

    def update(self, event: ProgressEvent):
        """Fold one pipeline progress observation into the snapshot."""
        with self._lock:
            videos = max(1, event.video_count)
            frames = max(1, event.frames_total)
            fraction = (event.video_index + event.frames_done / frames) / videos
            self._progress = min(1.0, max(self._progress, fraction))
            self._video = event.video_path.name

    def finish(self, summaries: List[VideoSummary], zones: dict,
               network: Optional[str], manifest_rows: Optional[List[dict]] = None):
        """Record completion and append to run comparison history."""
        with self._lock:
            self._state = "done"
            self._progress = 1.0
            self._network = network
            self._zones = dict(zones)
            self._summaries = [{
                "video": Path(s.video_path).name,
                "fps": s.fps,
                "total_frames": s.total_frames,
                "sampled": s.frames_sampled,
                "kept": s.frames_kept,
                "blur": s.frames_rejected_blur,
                "exposure": s.frames_rejected_exposure,
                "duplicate": s.frames_rejected_duplicate,
                "reflectance": s.reflectance_frames,
            } for s in summaries]
            run_idx = len(self._history) + 1
            active_gates = [k for k, v in self._gates.items() if v]
            gate_desc = "all gates" if len(active_gates) == len(GATE_KEYS) else (
                f"{len(active_gates)}/{len(GATE_KEYS)} gates"
            )
            run_entry = _summarize_run(
                run_id=f"run_{run_idx}",
                label=f"Run #{run_idx} ({network or 'default'} · {gate_desc})",
                network=network or "vodafone",
                gates=dict(self._gates),
                summaries=list(self._summaries),
                manifest_rows=manifest_rows or [],
                zones=dict(self._zones),
            )
            self._history.append(run_entry)

    def fail(self, message: str):
        """Record a failure."""
        with self._lock:
            self._state = "failed"
            self._error = message

    def snapshot(self) -> dict:
        """Return an atomic copy of the visible state."""
        with self._lock:
            return {
                "state": self._state,
                "progress": self._progress,
                "video": self._video,
                "network": self._network,
                "zones": dict(self._zones),
                "error": self._error,
                "summaries": list(self._summaries),
                "history_count": len(self._history),
            }

    def history(self) -> List[dict]:
        """Return a copy of completed run snapshots."""
        with self._lock:
            return list(self._history)


def _run_job(settings: Settings, job: JobState):
    """Worker-thread body: run the pipeline and settle the job state."""
    try:
        summaries = ExtractionPipeline(settings).run(progress=job.update)
        out_dir = Path(settings.runtime.output_directory)
        zones = _read_zone_counts(
            out_dir,
            settings.coverage.chunk_manifest_name,
        )
        manifest_rows = _load_manifest_rows(out_dir, settings.output.manifest_name)
        job.finish(summaries, zones, settings.coverage.network, manifest_rows)
    except Exception as e:  # noqa: BLE001 - thread boundary
        logger.exception("Pipeline run failed")
        job.fail(str(e))


def _build_timeline(settings: Settings, video: Optional[str]) -> Optional[dict]:
    """Derive overlay timeline events from the last run's artifacts."""
    output_root = Path(settings.runtime.output_directory)
    manifest = output_root / settings.output.manifest_name
    if not manifest.exists():
        fallback_root = Path("mock_dashcam_out/extracted")
        if (fallback_root / settings.output.manifest_name).exists():
            output_root = fallback_root
            manifest = output_root / settings.output.manifest_name
        else:
            return None
    by_stem: Dict[str, list] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["video_path"]).stem
            try:
                t = float(row["timestamp_sec"])
            except (KeyError, ValueError):
                continue
            blur_val = float(row.get("blur_score") or 0.0)
            intensity_val = float(row.get("mean_intensity") or 0.0)
            frame_idx = int(row.get("frame_index") or 0)
            by_stem.setdefault(stem, []).append((
                t, row.get("output_path", ""), blur_val, intensity_val, frame_idx,
            ))
    if not by_stem:
        return None
    requested = Path(video).stem if video else None
    matched = requested in by_stem
    stem = requested if matched else max(by_stem, key=lambda s: len(by_stem[s]))

    fps, duration = _video_stats(output_root, stem, settings)

    series = []
    cumulative_bytes = 0
    sizes_seen = []
    for t, output_path, blur_val, intensity_val, frame_idx in sorted(by_stem[stem]):
        size = None
        if output_path:
            p = Path(output_path)
            if p.exists():
                size = p.stat().st_size
            else:
                local_p = output_root / stem / Path(output_path).name
                if local_p.exists():
                    size = local_p.stat().st_size
        if size is None:
            size = int(sum(sizes_seen) / len(sizes_seen)) if sizes_seen \
                else FALLBACK_FRAME_BYTES
        else:
            sizes_seen.append(size)
        cumulative_bytes += size
        series.append({
            "t": round(t, 3),
            "frames": len(series) + 1,
            "frame_index": frame_idx,
            "bytes": cumulative_bytes,
            "frame_bytes": size,
            "blur_score": round(blur_val, 1),
            "mean_intensity": round(intensity_val, 1),
        })

    chunks = []
    slices_path = output_root / stem / "kept_slices.yaml"
    if slices_path.exists():
        try:
            payload = yaml.safe_load(slices_path.read_text(encoding="utf-8")) or {}
            for item in payload.get("slices", []):
                chunks.append({
                    "start": item["start"] / fps,
                    "end": item["end"] / fps,
                    "t": item["end"] / fps,
                    "frames": item["end"] - item["start"],
                })
        except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning("Could not read %s: %s", slices_path, e)

    upstream = []
    network = settings.coverage.network
    coverage_path = output_root / settings.coverage.chunk_manifest_name
    if coverage_path.exists():
        try:
            payload = yaml.safe_load(coverage_path.read_text(encoding="utf-8")) or {}
            network = payload.get("network", network)
            for zone, ranges in (payload.get("zones") or {}).items():
                for item in ranges or []:
                    if Path(str(item.get("video", ""))).stem != stem:
                        continue
                    upstream.append({
                        "t0": item["start"] / fps,
                        "t1": item["end"] / fps,
                        "zone": zone,
                    })
        except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning("Could not read %s: %s", coverage_path, e)
    upstream.sort(key=lambda w: w["t0"])

    return {
        "video": stem,
        "matched": matched,
        "videos": sorted(by_stem),
        "fps": fps,
        "duration": duration,
        "network": network,
        "series": series,
        "chunks": chunks,
        "upstream": upstream,
        "step": TIMELINE_STEP_SECONDS,
    }


def _video_stats(output_root: Path, stem: str, settings: Settings):
    """Look up fps and duration for a video from the summaries CSV."""
    fps = settings.runtime.default_fps
    duration = 0.0
    summaries = output_root / "video_summaries.csv"
    if summaries.exists():
        with summaries.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if Path(row.get("video_path", "")).stem != stem:
                    continue
                try:
                    fps = float(row["fps"]) or fps
                    duration = int(row["total_frames"]) / fps
                except (KeyError, ValueError, ZeroDivisionError):
                    pass
                break
    return fps, duration


def _build_comparison_payload(settings: Settings, job: JobState,
                              last_output: Optional[Path]) -> dict:
    """Build comparison data across runs, videos, and carrier networks."""
    active_root = _resolve_artifact_root(settings, last_output)
    fallback_root = Path("mock_dashcam_out/extracted")

    runs = list(job.history())

    # Include on-disk active artifacts if no in-memory runs yet
    if active_root and not runs:
        summaries = _load_artifact_summaries(active_root)
        manifest_rows = _load_manifest_rows(active_root, settings.output.manifest_name)
        zones = _read_zone_counts(active_root, settings.coverage.chunk_manifest_name)
        runs.append(_summarize_run(
            run_id="disk_active",
            label=f"Quality-Gated Run ({settings.coverage.network})",
            network=settings.coverage.network,
            gates={k: True for k in GATE_KEYS},
            summaries=summaries,
            manifest_rows=manifest_rows,
            zones=zones,
        ))

    # Always provide a "Raw Unfiltered (No Quality Gates)" reference baseline
    # so users can immediately compare gated extraction vs ungated baseline
    base_source_summaries = (
        runs[-1]["summaries"] if runs else _load_artifact_summaries(fallback_root)
    )
    if base_source_summaries:
        ungated_summaries = []
        for s in base_source_summaries:
            sampled = int(s.get("sampled", 0))
            ungated_summaries.append({
                **s,
                "kept": sampled,
                "blur": 0,
                "exposure": 0,
                "duplicate": 0,
                "reflectance": 0,
            })
        runs.append(_summarize_run(
            run_id="baseline_ungated",
            label="Ungated Baseline (All Sampled Frames Kept)",
            network=settings.coverage.network,
            gates={k: False for k in GATE_KEYS},
            summaries=ungated_summaries,
            manifest_rows=[],
            zones={z: 1 for z in settings.coverage.networks.get(settings.coverage.network, {})},
        ))

    # Also include Mock Reference if active_root differs from fallback_root
    if fallback_root.exists() and active_root and active_root.resolve() != fallback_root.resolve():
        ref_summaries = _load_artifact_summaries(fallback_root)
        ref_manifest = _load_manifest_rows(fallback_root, settings.output.manifest_name)
        ref_zones = _read_zone_counts(fallback_root, settings.coverage.chunk_manifest_name)
        if ref_summaries:
            runs.append(_summarize_run(
                run_id="reference_fixture",
                label="Bundled Reference Fixture (CAR01 Survey)",
                network="vodafone",
                gates={k: True for k in GATE_KEYS},
                summaries=ref_summaries,
                manifest_rows=ref_manifest,
                zones=ref_zones,
            ))

    manifest_rows = (
        _load_manifest_rows(active_root, settings.output.manifest_name)
        if active_root else []
    )
    summaries = (
        _load_artifact_summaries(active_root)
        if active_root else []
    )

    # Group frames by video stem for side-by-side video/frame inspection
    videos_detail = {}
    for s in summaries:
        stem = Path(s["video"]).stem
        v_frames = [r for r in manifest_rows if r["video_stem"] == stem]
        blur_vals = [r["blur_score"] for r in v_frames]
        int_vals = [r["mean_intensity"] for r in v_frames]
        videos_detail[stem] = {
            **s,
            "stem": stem,
            "retention_pct": round(s["kept"] / s["sampled"] * 100.0, 1) if s["sampled"] else 0.0,
            "mean_blur": round(sum(blur_vals) / len(blur_vals), 1) if blur_vals else 0.0,
            "mean_intensity": round(sum(int_vals) / len(int_vals), 1) if int_vals else 0.0,
            "frames": v_frames,
        }

    # Carrier network comparison
    carriers = []
    for net_name, zones_cfg in sorted(settings.coverage.networks.items()):
        total_area = 0.0
        zone_names = sorted(zones_cfg.keys())
        for bbox in zones_cfg.values():
            if len(bbox) == 4:
                total_area += abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        has_5g = any("5g" in z.lower() for z in zone_names)
        carriers.append({
            "network": net_name,
            "zones": zone_names,
            "zone_count": len(zone_names),
            "coverage_deg2": round(total_area, 5),
            "has_5g": has_5g,
            "is_active": net_name == (job.snapshot().get("network") or settings.coverage.network),
        })

    return {
        "runs": runs,
        "videos": list(videos_detail.values()),
        "carriers": carriers,
        "active_network": job.snapshot().get("network") or settings.coverage.network,
    }


def create_app(config_path: Path) -> FastAPI:
    """Build the FastAPI app around one config file and one job slot."""
    app = FastAPI(title="Cloud Optimised DashCam")
    job = JobState()
    cache: dict = {"settings": None}

    def load_settings() -> Settings:
        if cache["settings"] is None:
            cache["settings"] = Settings.load(Path(config_path))
        return cache["settings"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_PATH.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        return job.snapshot()

    @app.get("/api/networks")
    def networks() -> dict:
        settings = load_settings()
        return {
            "networks": sorted(settings.coverage.networks),
            "default": settings.coverage.network,
        }

    @app.post("/api/extract")
    def extract(req: ExtractRequest) -> dict:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            job.fail(f"config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        try:
            applied = _apply_request(settings, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not job.try_start(applied.coverage.network, req.gates):
            raise HTTPException(status_code=409, detail="a job is already running")
        cache["last_output"] = applied.runtime.output_directory
        threading.Thread(
            target=_run_job, args=(applied, job), daemon=True,
        ).start()
        return {"started": True, "network": applied.coverage.network}

    @app.get("/api/timeline")
    def timeline(video: Optional[str] = None) -> dict:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        if cache.get("last_output"):
            settings = replace(settings, runtime=replace(
                settings.runtime, output_directory=cache["last_output"],
            ))
        payload = _build_timeline(settings, video)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail="no run artifacts found; run an extraction first",
            )
        return payload

    @app.get("/api/compare")
    def compare() -> dict:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _build_comparison_payload(settings, job, cache.get("last_output"))

    @app.get("/api/frame-image")
    def frame_image(path: str = Query(..., description="Relative frame image path")):
        candidate = Path(path).resolve()
        allowed_roots = [
            Path("extracted_frames").resolve(),
            Path("mock_dashcam_out").resolve(),
        ]
        if cache.get("last_output"):
            allowed_roots.append(Path(cache["last_output"]).resolve())
        if not any(
            candidate == root or root in candidate.parents
            for root in allowed_roots
        ):
            raise HTTPException(status_code=403, detail="access denied")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="frame image not found")
        return FileResponse(candidate, media_type="image/png")

    @app.get("/api/export")
    def export_stats(
        format: str = Query("json", pattern="^(json|csv_summaries|csv_manifest|yaml_coverage)$"),
    ) -> Response:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        active_root = _resolve_artifact_root(settings, cache.get("last_output"))
        if active_root is None:
            raise HTTPException(status_code=404, detail="no run artifacts available to export")

        summaries = _load_artifact_summaries(active_root)
        manifest_rows = _load_manifest_rows(active_root, settings.output.manifest_name)
        zones = _read_zone_counts(active_root, settings.coverage.chunk_manifest_name)
        comparison = _build_comparison_payload(settings, job, cache.get("last_output"))

        if format == "csv_summaries":
            summaries_file = active_root / "video_summaries.csv"
            if summaries_file.exists():
                content = summaries_file.read_text(encoding="utf-8")
            else:
                buf = io.StringIO()
                writer = csv.DictWriter(
                    buf,
                    fieldnames=[
                        "video", "fps", "total_frames", "sampled", "kept",
                        "blur", "exposure", "duplicate", "reflectance",
                    ],
                )
                writer.writeheader()
                writer.writerows(summaries)
                content = buf.getvalue()
            return Response(
                content=content,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="video_summaries.csv"'},
            )

        if format == "csv_manifest":
            manifest_file = active_root / settings.output.manifest_name
            if manifest_file.exists():
                content = manifest_file.read_text(encoding="utf-8")
            else:
                buf = io.StringIO()
                fieldnames = list(manifest_rows[0].keys()) if manifest_rows else [
                    "video", "frame_index", "timestamp_sec", "blur_score", "mean_intensity",
                ]
                writer = csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)
                content = buf.getvalue()
            return Response(
                content=content,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="frame_manifest.csv"'},
            )

        if format == "yaml_coverage":
            cov_file = active_root / settings.coverage.chunk_manifest_name
            if cov_file.exists():
                content = cov_file.read_text(encoding="utf-8")
            else:
                content = yaml.safe_dump({"network": settings.coverage.network, "zones": zones})
            return Response(
                content=content,
                media_type="application/x-yaml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="coverage_chunks.yaml"'},
            )

        payload = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifact_directory": active_root.as_posix(),
            "status": job.snapshot(),
            "active_run": comparison["runs"][0] if comparison["runs"] else None,
            "runs": comparison["runs"],
            "video_summaries": summaries,
            "frame_manifest": manifest_rows,
            "coverage_zones": zones,
            "carriers": comparison["carriers"],
        }
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="dashcam_extraction_stats.json"'},
        )

    return app


def main():
    """Entry point: serve the panel on the configured host/port."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Frame extraction web panel")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--host", default=None,
                        help="Override ui.host (e.g. 0.0.0.0 for containers)")
    parser.add_argument("--port", type=int, default=None,
                        help="Override ui.port (e.g. $PORT on Cloud Run)")
    args = parser.parse_args()
    host, port = "127.0.0.1", 8321
    try:
        ui = Settings.load(args.config).ui
        host, port = ui.host, ui.port
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning("Could not read ui config (%s); using %s:%d", e, host, port)
    if args.host is not None:
        host = args.host
    if args.port is not None:
        port = args.port
    uvicorn.run(create_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
