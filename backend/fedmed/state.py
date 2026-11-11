"""Shared runtime state for the FedMed engine.

A single in-process store holds everything the API and WebSocket layers
need to observe: per-round metrics, per-node status, privacy accounting
and the latest segmentation preview. The federated server writes to it,
the API reads from it.

A small pub/sub mechanism (:class:`EventBus`) is attached to the state so
that the WebSocket layer can push a frame to every connected dashboard as
soon as a round completes, rather than polling.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Callable, Deque

from .config import CONFIG


class NodeStatus:
    """Lifecycle states for a hospital node."""

    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    FAILED = "failed"


class EventBus:
    """Thread-safe publish/subscribe bridge between threads and asyncio.

    The federated training loop runs in worker threads while the WebSocket
    handlers run on the asyncio event loop. Subscribers register a
    ``threading.Event``-style callback that is scheduled onto the loop with
    ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Associate the bus with the running asyncio loop."""
        self._loop = loop

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, event: dict) -> None:
        """Publish an event to every subscriber (safe from any thread)."""
        with self._lock:
            subscribers = list(self._subscribers)
            loop = self._loop

        for callback in subscribers:
            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(callback, event)
                except RuntimeError:  # pragma: no cover - loop shutting down
                    continue
            else:
                callback(event)


class RuntimeState:
    """Thread-safe store for live federated training state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.current_round: int = 0
        self.total_rounds: int = CONFIG.federated.num_rounds
        self.is_running: bool = False

        self.rounds: Deque = deque(maxlen=200)
        self.nodes: dict = {}
        self.privacy_events: Deque = deque(maxlen=100)
        self.security_events: Deque = deque(maxlen=100)

        self.baseline: dict | None = None
        self.federated_summary: dict | None = None
        self.latest_segmentation: dict | None = None

        self.dp_epsilon: float = 0.0
        self.dp_delta: float = CONFIG.privacy.dp_delta
        self.dp_rounds: int = 0
        self.dp_per_node: dict[str, dict] = {}

        #: Published events are consumed by the WebSocket layer.
        self.bus = EventBus()

        self._init_nodes()

    # -- nodes ---------------------------------------------------------
    def _init_nodes(self) -> None:
        for hospital in CONFIG.hospitals:
            self.nodes[hospital.hospital_id] = {
                "hospital_id": hospital.hospital_id,
                "node_id": hospital.node_id,
                "port": hospital.port,
                "status": NodeStatus.OFFLINE,
                "samples": 0,
                "last_seen": None,
                "last_sync": None,
                "rounds_participated": 0,
                "local_loss": None,
                "local_dice": None,
                "encrypted": False,
                "payload_bytes": 0,
                "progress": 0.0,
                "message": "Awaiting connection",
            }

    def update_node(self, hospital_id: str, **fields: Any) -> None:
        """Update a node's record under the lock."""
        with self._lock:
            node = self.nodes.get(hospital_id)
            if node is None:
                return
            node.update(fields)
            node["last_seen"] = time.time()

    def set_node_status(
        self, hospital_id: str, status: str, message: str | None = None
    ) -> None:
        self.update_node(hospital_id, status=status, message=message or status)
        self.bus.publish(
            {
                "type": "node_status",
                "hospital_id": hospital_id,
                "status": status,
                "message": message or status,
            }
        )

    def mark_node_online(self, hospital_id: str) -> None:
        self.update_node(
            hospital_id,
            status=NodeStatus.ONLINE,
            last_sync=time.time(),
            message="Connected and ready",
        )
        self.bus.publish(
            {"type": "node_status", "hospital_id": hospital_id, "status": NodeStatus.ONLINE}
        )

    def node_snapshot(self) -> list:
        with self._lock:
            return [dict(v) for v in self.nodes.values()]

    # -- rounds --------------------------------------------------------
    def add_round(self, round_data: dict) -> None:
        with self._lock:
            self.rounds.append(round_data)
            self.current_round = round_data.get("round", self.current_round)
        self.bus.publish({"type": "round", "data": round_data})

    def rounds_snapshot(self) -> list:
        with self._lock:
            return list(self.rounds)

    def restore_rounds(self, rounds: list) -> None:
        """Re-hydrate the round history from a persisted run.

        The API is restarted between sessions, so the last completed run
        is loaded from disk to keep the dashboard populated. Live runs
        still take precedence: this only appends rounds that are not
        already present.
        """
        with self._lock:
            existing = {r.get("round") for r in self.rounds}
            for record in rounds or []:
                round_no = record.get("round")
                if round_no in existing:
                    continue
                self.rounds.append(record)
                existing.add(round_no)
            if self.rounds:
                self.current_round = max(
                    r.get("round", 0) for r in self.rounds
                )

    # -- events --------------------------------------------------------
    def log_privacy_event(self, event: str, detail: dict | None = None) -> None:
        with self._lock:
            payload = {
                "event": event,
                "timestamp": time.time(),
                "detail": detail or {},
            }
            self.privacy_events.append(payload)
        self.bus.publish({"type": "privacy_event", "data": payload})

    def log_security_event(self, event: str, detail: dict | None = None) -> None:
        with self._lock:
            payload = {
                "event": event,
                "timestamp": time.time(),
                "detail": detail or {},
            }
            self.security_events.append(payload)
        self.bus.publish({"type": "security_event", "data": payload})

    def events_snapshot(self) -> tuple:
        with self._lock:
            return list(self.privacy_events), list(self.security_events)

    def restore_events(
        self,
        privacy_events: list | None = None,
        security_events: list | None = None,
    ) -> None:
        """Re-hydrate the audit trail from a persisted run."""
        with self._lock:
            for event in privacy_events or []:
                self.privacy_events.append(event)
            for event in security_events or []:
                self.security_events.append(event)

    def dump_events(self) -> dict:
        """Serialise the audit trail for persistence."""
        with self._lock:
            return {
                "privacy_events": list(self.privacy_events),
                "security_events": list(self.security_events),
            }

    # -- lifecycle -----------------------------------------------------
    def mark_started(self) -> None:
        with self._lock:
            self.is_running = True
            self.started_at = time.time()
            self.finished_at = None
        self.bus.publish({"type": "run_started", "data": self.status_snapshot()})

    def mark_finished(self, summary: dict | None = None) -> None:
        with self._lock:
            self.is_running = False
            self.finished_at = time.time()
            if summary:
                self.federated_summary = summary
        self.bus.publish({"type": "run_finished", "data": summary or {}})

    def set_segmentation(self, payload: dict) -> None:
        """Store the latest segmentation preview."""
        with self._lock:
            self.latest_segmentation = payload
        self.bus.publish({"type": "segmentation", "data": payload})

    def status_snapshot(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.started_at if self.started_at else 0.0
            return {
                "is_running": self.is_running,
                "current_round": self.current_round,
                "total_rounds": self.total_rounds,
                "elapsed_seconds": round(elapsed, 1),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "dp_epsilon": self.dp_epsilon,
                "dp_delta": self.dp_delta,
                "dp_rounds": self.dp_rounds,
                "dp_accountant": {
                    "epsilon": self.dp_epsilon,
                    "delta": self.dp_delta,
                    "rounds": self.dp_rounds,
                    "per_node": dict(self.dp_per_node),
                },
            }

    def full_snapshot(self) -> dict:
        """A single self-contained payload used for the initial WS frame."""
        privacy_events, security_events = self.events_snapshot()
        return {
            "status": self.status_snapshot(),
            "nodes": self.node_snapshot(),
            "rounds": self.rounds_snapshot(),
            "privacy_events": privacy_events,
            "security_events": security_events,
            "baseline": self.baseline,
            "summary": self.federated_summary,
            "segmentation": self.latest_segmentation,
        }


#: Process-wide singleton used by the server, API and client processes.
STATE = RuntimeState()
