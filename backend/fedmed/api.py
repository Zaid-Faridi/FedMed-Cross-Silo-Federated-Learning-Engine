"""FedMed REST + WebSocket API.

Exposes everything the React dashboard needs:

* ``GET  /api/status``            -- run status, DP budget, progress
* ``GET  /api/nodes``             -- per-hospital node telemetry
* ``GET  /api/metrics``           -- per-round metrics history
* ``POST /api/train/baseline``    -- launch the centralized baseline
* ``POST /api/train/federated``   -- launch the federated run
* ``GET  /api/mri/analyze``       -- run segmentation + render the masks
* ``GET  /api/privacy``           -- HE/DP posture and audit logs
* ``GET  /api/settings``          -- the effective configuration
* ``POST /api/settings``          -- persist configuration overrides
* ``WS   /ws/metrics``            -- live metric stream

The heavy jobs (baseline training, the federated loop) run in background
threads so the HTTP layer stays responsive; their results are published on
the shared :data:`fedmed.state.STATE` and streamed to the dashboards.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import CONFIG, DATA_ROOT, MODEL_ROOT
from .model import count_parameters, build_model
from .privacy import dp as dp_mod
from .privacy import he as he_mod
from .segmentation import available_models, dataset_overview, run_inference
from .state import STATE

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FedMed Federated Learning Engine",
    description=(
        "Cross-silo federated learning engine for privacy-preserving brain "
        "tumour segmentation across hospital nodes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background job management
# ---------------------------------------------------------------------------


class JobRegistry:
    """Tracks background training jobs so the API can report on them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def start(self, kind: str, thread: threading.Thread) -> None:
        with self._lock:
            self.jobs[kind] = {
                "kind": kind,
                "state": "running",
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
                "thread": thread,
            }

    def finish(self, kind: str, error: str | None = None) -> None:
        with self._lock:
            job = self.jobs.get(kind)
            if job is None:
                return
            job["state"] = "failed" if error else "completed"
            job["finished_at"] = time.time()
            job["error"] = error

    def snapshot(self) -> list:
        with self._lock:
            return [
                {
                    "kind": j["kind"],
                    "state": j["state"],
                    "started_at": j["started_at"],
                    "finished_at": j["finished_at"],
                    "error": j["error"],
                    "duration_seconds": round(
                        (j["finished_at"] or time.time()) - j["started_at"], 1
                    ),
                }
                for j in self.jobs.values()
            ]


JOBS = JobRegistry()


def _restore_persisted_run() -> None:
    """Re-hydrate the dashboard from the last completed run on disk.

    The API process is usually restarted between sessions, so the persisted
    ``federated_summary.json`` is loaded to populate the round history and
    the summary panel. A live run overwrites this as soon as it starts.
    """
    summary_path = Path(MODEL_ROOT) / "federated_summary.json"
    if not summary_path.exists():
        return

    try:
        with open(summary_path) as fh:
            summary = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not restore the persisted run: %s", exc)
        return

    STATE.federated_summary = summary
    STATE.restore_rounds(summary.get("history", []))
    STATE.total_rounds = summary.get("rounds_completed") or CONFIG.federated.num_rounds

    # Restore the privacy audit trail and the DP accountant.
    audit_path = Path(MODEL_ROOT) / "audit_log.json"
    if audit_path.exists():
        try:
            with open(audit_path) as fh:
                audit = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not restore the audit trail: %s", exc)
        else:
            STATE.restore_events(
                audit.get("privacy_events"), audit.get("security_events")
            )
            accountant = audit.get("dp_accountant") or {}
            STATE.dp_epsilon = float(accountant.get("epsilon", 0.0))
            STATE.dp_delta = float(
                accountant.get("delta", CONFIG.privacy.dp_delta)
            )
            STATE.dp_rounds = int(accountant.get("rounds", 0))
            STATE.dp_per_node = dict(accountant.get("per_node") or {})

    logger.info(
        "Restored persisted run: %s rounds, best dice %.2f%%",
        summary.get("rounds_completed", 0),
        summary.get("federated_best_dice", 0.0),
    )


_restore_persisted_run()


def _run_baseline_job(epochs: int) -> None:
    from .baseline import train_baseline

    try:
        summary = train_baseline(epochs=epochs)
        STATE.baseline = summary
        STATE.bus.publish({"type": "baseline", "data": summary})
        JOBS.finish("baseline")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Baseline training failed")
        JOBS.finish("baseline", error=str(exc))
        STATE.bus.publish({"type": "error", "data": {"job": "baseline", "error": str(exc)}})


def _run_federated_job(rounds: int, dropout_round: int) -> None:
    """Run the full federated cluster (aggregator + hospital nodes).

    This runs in a worker thread, so Flower's process-wide signal
    handlers are skipped -- Python only permits ``signal.signal`` on the
    main thread.
    """
    from .orchestrator import run_cluster

    try:
        summary = run_cluster(
            rounds=rounds,
            dropout_round=dropout_round,
            install_signal_handlers=False,
        )
        STATE.federated_summary = summary
        JOBS.finish("federated")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Federated training failed")
        JOBS.finish("federated", error=str(exc))
        STATE.bus.publish(
            {"type": "error", "data": {"job": "federated", "error": str(exc)}}
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BaselineRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=200)


class FederatedRequest(BaseModel):
    rounds: int = Field(default=CONFIG.federated.num_rounds, ge=1, le=100)
    dropout_round: int = Field(default=0, ge=0, le=100)


class SettingsUpdate(BaseModel):
    num_rounds: int | None = Field(default=None, ge=1, le=100)
    local_epochs: int | None = Field(default=None, ge=1, le=50)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    learning_rate: float | None = Field(default=None, gt=0, le=1.0)
    dp_noise_multiplier: float | None = Field(default=None, gt=0, le=100)
    dp_clip_norm: float | None = Field(default=None, gt=0, le=1000)
    dp_delta: float | None = Field(default=None, gt=0, lt=1)
    enable_he: bool | None = None
    enable_dp: bool | None = None


#: Persisted configuration overrides (applied on the next run).
OVERRIDES: dict[str, Any] = {}
SETTINGS_FILE = Path(MODEL_ROOT) / "settings.json"


def _effective_config() -> dict:
    """Return the effective configuration (defaults + persisted overrides)."""
    model = build_model()
    return {
        "federated": {
            "num_rounds": OVERRIDES.get("num_rounds", CONFIG.federated.num_rounds),
            "local_epochs": OVERRIDES.get("local_epochs", CONFIG.federated.local_epochs),
            "batch_size": OVERRIDES.get("batch_size", CONFIG.federated.batch_size),
            "learning_rate": OVERRIDES.get("learning_rate", CONFIG.federated.learning_rate),
            "weight_decay": CONFIG.federated.weight_decay,
            "strategy": CONFIG.federated.strategy,
            "dropout_round": OVERRIDES.get(
                "dropout_round", CONFIG.federated.dropout_round
            ),
        },
        "model": {
            "architecture": "3D U-Net (MONAI)",
            "channels": list(CONFIG.model.channels),
            "strides": list(CONFIG.model.strides),
            "num_res_units": CONFIG.model.num_res_units,
            "num_classes": CONFIG.model.num_classes,
            "patch_size": list(CONFIG.model.spatial_size),
            "parameters": count_parameters(model),
        },
        "data": {
            "num_train": CONFIG.data.num_samples,
            "num_val": CONFIG.data.num_val_samples,
            "hospital_split": list(CONFIG.data.hospital_split),
            "patch_size": list(CONFIG.data.patch_size),
        },
        "privacy": {
            **dp_mod.dp_status(),
            **he_mod.he_status(),
            "he_coeff_modulus": list(CONFIG.privacy.he_coeff_modulus),
            "he_poly_modulus": CONFIG.privacy.he_poly_modulus,
        },
        "transport": {
            "protocol": "gRPC + Mutual TLS",
            "fl_server": CONFIG.server.fl_address,
            "api_server": f"{CONFIG.server.host}:{CONFIG.server.port}",
            "certificate_authority": str(CONFIG.server.ca_cert),
            "nodes": [
                {
                    "hospital_id": h.hospital_id,
                    "node_id": h.node_id,
                    "grpc_port": h.port,
                }
                for h in CONFIG.hospitals
            ],
        },
        "overrides": OVERRIDES,
    }


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": app.version,
        "tenseal_available": he_mod.TENSEAL_AVAILABLE,
    }


@app.get("/api/status")
def get_status() -> dict:
    """Overall run status."""
    status = STATE.status_snapshot()
    status["jobs"] = JOBS.snapshot()
    status["baseline_ready"] = (Path(MODEL_ROOT) / "baseline_metrics.json").exists()
    status["federated_ready"] = (Path(MODEL_ROOT) / "federated_summary.json").exists()
    status["dataset_ready"] = (Path(DATA_ROOT) / "manifest.json").exists()
    return status


@app.get("/api/nodes")
def get_nodes() -> dict:
    """Per-hospital node telemetry."""
    nodes = STATE.node_snapshot()
    online = sum(1 for n in nodes if n["status"] == "online")
    return {
        "nodes": nodes,
        "total": len(nodes),
        "online": online,
        "offline": len(nodes) - online,
        "training": sum(1 for n in nodes if n["status"] == "training"),
    }


@app.get("/api/metrics")
def get_metrics() -> dict:
    """Per-round metrics history plus the baseline comparison."""
    rounds = STATE.rounds_snapshot()
    baseline = STATE.baseline or {}
    if not baseline:
        path = Path(MODEL_ROOT) / "baseline_metrics.json"
        if path.exists():
            with open(path) as fh:
                baseline = json.load(fh)
            STATE.baseline = baseline

    final = rounds[-1] if rounds else {}
    baseline_dice = float(baseline.get("best_dice", 0.0))
    fed_dice = float(final.get("dice", 0.0))

    return {
        "rounds": rounds,
        "baseline": {
            "best_dice": baseline_dice,
            "final_dice": baseline.get("final_dice", 0.0),
            "final_loss": baseline.get("final_loss", 0.0),
            "epochs": baseline.get("epochs", 0),
            "history": baseline.get("history", []),
        },
        "summary": STATE.federated_summary,
        "comparison": {
            "federated_dice": fed_dice,
            "baseline_dice": baseline_dice,
            "gap": round(baseline_dice - fed_dice, 3) if baseline_dice else None,
            "ratio_pct": round(fed_dice / baseline_dice * 100, 2) if baseline_dice else None,
        },
    }


@app.post("/api/train/baseline")
def start_baseline(req: BaselineRequest) -> dict:
    """Launch the centralized baseline trainer in the background."""
    if not (Path(DATA_ROOT) / "manifest.json").exists():
        raise HTTPException(
            status_code=409, detail="Dataset not prepared. Run `python -m fedmed.prepare_data`."
        )

    running = [j for j in JOBS.snapshot() if j["kind"] == "baseline" and j["state"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail="Baseline training is already running")

    thread = threading.Thread(
        target=_run_baseline_job, args=(req.epochs,), daemon=True, name="fedmed-baseline"
    )
    JOBS.start("baseline", thread)
    thread.start()

    return {
        "started": True,
        "job": "baseline",
        "epochs": req.epochs,
        "message": "Centralized baseline training started",
    }


@app.post("/api/train/federated")
def start_federated(req: FederatedRequest) -> dict:
    """Launch the federated training loop in the background."""
    if not (Path(DATA_ROOT) / "manifest.json").exists():
        raise HTTPException(
            status_code=409, detail="Dataset not prepared. Run `python -m fedmed.prepare_data`."
        )

    running = [j for j in JOBS.snapshot() if j["kind"] == "federated" and j["state"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail="Federated training is already running")

    thread = threading.Thread(
        target=_run_federated_job,
        args=(req.rounds, req.dropout_round),
        daemon=True,
        name="fedmed-federated",
    )
    JOBS.start("federated", thread)
    thread.start()

    return {
        "started": True,
        "job": "federated",
        "rounds": req.rounds,
        "dropout_round": req.dropout_round,
        "message": "Federated training started",
    }


@app.get("/api/mri/analyze")
def mri_analyze(model: str = "federated", sample: int | None = None) -> dict:
    """Run segmentation inference and render the mask overlay."""
    try:
        result = run_inference(which=model, sample_index=sample)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    result["dataset"] = dataset_overview()
    result["models"] = available_models()
    return result


@app.get("/api/privacy")
def get_privacy() -> dict:
    """HE/DP posture plus the privacy and security audit logs."""
    privacy_events, security_events = STATE.events_snapshot()
    return {
        "he": he_mod.he_status(),
        "dp": dp_mod.dp_status(),
        "accountant": {
            "epsilon": STATE.dp_epsilon,
            "delta": STATE.dp_delta,
            "rounds": STATE.dp_rounds,
            "projected_epsilon": CONFIG.privacy.dp_epsilon_bound,
            "per_node": STATE.dp_per_node,
        },
        "privacy_events": privacy_events,
        "security_events": security_events,
        "guarantees": [
            {
                "title": "No raw data leaves the hospital",
                "detail": (
                    "MRI volumes and patient identifiers never traverse the "
                    "network. Only model parameters are exchanged."
                ),
            },
            {
                "title": "Homomorphic aggregation",
                "detail": (
                    "Model updates are encrypted with TenSEAL (CKKS, 16384-bit "
                    "polynomial modulus). The server adds ciphertexts and "
                    "decrypts only the aggregate."
                ),
            },
            {
                "title": "Differential privacy (DP-SGD)",
                "detail": (
                    "Per-patient gradients are clipped to an L2 bound and "
                    "noised with the Gaussian mechanism; the cumulative "
                    "(epsilon, delta) budget is tracked with an RDP accountant."
                ),
            },
            {
                "title": "Mutual TLS transport",
                "detail": (
                    "gRPC channels between the aggregator and every hospital "
                    "node are protected by mutual TLS with client certificate "
                    "verification."
                ),
            },
        ],
    }


@app.get("/api/settings")
def get_settings() -> dict:
    """The effective configuration."""
    return _effective_config()


@app.post("/api/settings")
def update_settings(req: SettingsUpdate) -> dict:
    """Persist configuration overrides applied to the next run."""
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No settings supplied")

    OVERRIDES.update(updates)

    with open(SETTINGS_FILE, "w") as fh:
        json.dump(OVERRIDES, fh, indent=2)

    return {"updated": True, "overrides": OVERRIDES, "config": _effective_config()}


@app.get("/api/logs")
def get_logs(limit: int = 100) -> dict:
    """Recent federated log lines."""
    path = Path(CONFIG.server.ca_cert).parent.parent / "logs" / "federated.log"
    if not path.exists():
        return {"lines": [], "path": str(path)}
    lines = path.read_text(errors="replace").splitlines()
    return {"lines": lines[-limit:], "path": str(path)}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/metrics")
async def metrics_socket(websocket: WebSocket) -> None:
    """Stream live training metrics to the dashboard."""
    await websocket.accept()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def on_event(event: dict) -> None:
        """Called from the training threads (via ``call_soon_threadsafe``)."""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except asyncio.QueueFull:  # pragma: no cover - slow consumer
            pass

    STATE.bus.subscribe(on_event)

    try:
        # Initial snapshot so the dashboard paints immediately.
        await websocket.send_json({"type": "snapshot", "data": STATE.full_snapshot()})

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Keep-alive ping so proxies do not drop the socket.
                await websocket.send_json(
                    {"type": "ping", "data": {"t": time.time()}}
                )
                continue

            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("Dashboard disconnected")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("WebSocket error: %s", exc)
    finally:
        STATE.bus.unsubscribe(on_event)


@app.get("/")
def root() -> JSONResponse:
    """Service banner."""
    return JSONResponse(
        {
            "service": "FedMed Federated Learning Engine",
            "version": app.version,
            "docs": "/docs",
            "websocket": "/ws/metrics",
        }
    )
