
"""FedMed cluster orchestrator.

Runs the complete federated topology in one process:

* the central **aggregator** (a Flower server with the homomorphic
  aggregation strategy), and
* one **Flower client** per hospital node (each with its own private
  dataset slice, DP-SGD and CKKS encryption).

Each hospital node additionally exposes a gRPC/TLS coordination service on
its own port, so the control plane required by the project brief is
present and verifiable.

The aggregator's gRPC server is started *first* so the nodes can connect
immediately; Flower's ``start_server`` blocks for the duration of the run,
so it is executed on a background thread and the nodes are launched once
the server's port is accepting connections.

Usage:
    python -m fedmed.orchestrator --rounds 10
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from pathlib import Path

from .config import CONFIG, DATA_ROOT
from .privacy import he as he_mod
from .state import STATE, NodeStatus

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Return True when a TCP connection to ``host:port`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _assert_ports_free() -> None:
    """Fail fast when a previous run still holds the cluster's ports.

    A hospital node that survives its aggregator will happily reconnect to
    the *next* aggregator, silently joining the round as a second client
    for the same hospital. Detecting the stale listener up front keeps the
    federated run honest.
    """
    busy: list[str] = []

    aggregator_host, _, aggregator_port = CONFIG.server.fl_address.rpartition(":")
    if _port_open(aggregator_host, int(aggregator_port)):
        busy.append(f"aggregator {CONFIG.server.fl_address}")

    for hospital in CONFIG.hospitals:
        if _port_open("127.0.0.1", hospital.port):
            busy.append(f"{hospital.node_id} gRPC :{hospital.port}")

    if busy:
        raise RuntimeError(
            "The following FedMed ports are already in use, which usually "
            "means a previous run is still active: "
            + ", ".join(busy)
            + ". Stop the old process before starting a new run."
        )


def _node_thread(
    hospital_id: str,
    node_id: str,
    address: str,
    use_tls: bool,
    ready: threading.Event,
    seed: int | None = None,
) -> None:
    """Run one hospital node (gRPC service + Flower client)."""
    from .client import HospitalClient
    from .grpc_service import serve

    hospital = next(h for h in CONFIG.hospitals if h.hospital_id == hospital_id)

    # 1. Start the node's gRPC/TLS coordination service.
    grpc_server = None
    try:
        grpc_server = serve(hospital_id, node_id, hospital.port)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[%s] gRPC server failed to start: %s", node_id, exc)

    # 2. Wait for the aggregator to accept connections.
    host, _, port_str = address.rpartition(":")
    deadline = time.time() + 90
    while time.time() < deadline and not _port_open(host, int(port_str)):
        time.sleep(0.25)
    if not _port_open(host, int(port_str)):
        logger.error("[%s] aggregator never became reachable", hospital.node_id)
        ready.set()
        return

    # 3. Start the Flower client.
    try:
        from flwr.compat.client.app import start_client

        client = HospitalClient(hospital_id, node_id, seed=seed)
        STATE.set_node_status(hospital_id, NodeStatus.ONLINE, "Client started")
        ready.set()

        if use_tls:
            ca = Path(CONFIG.server.ca_cert)
            if not ca.exists():
                raise FileNotFoundError(
                    f"CA certificate not found at {ca}. Run python -m fedmed.tls or pass --insecure."
                )
            root = ca.read_bytes()
            start_client(
                server_address=address,
                client=client,
                root_certificates=root,
                insecure=False,
            )
        else:
            start_client(server_address=address, client=client, insecure=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[%s] client stopped: %s", node_id, exc)
        ready.set()
    finally:
        if grpc_server is not None:
            grpc_server.stop(grace=1)
        STATE.set_node_status(hospital_id, NodeStatus.OFFLINE, "Client disconnected")


def run_cluster(
    rounds: int | None = None,
    address: str | None = None,
    dropout_round: int | None = None,
    enable_he: bool = True,
    enable_dp: bool = True,
    use_tls: bool = True,
    install_signal_handlers: bool = True,
    seed: int | None = None,
) -> dict:
    """Start every hospital node and run the federated loop to completion.

    The Flower aggregator runs on the calling thread. When the caller is
    the main thread the standard ``start_server`` entrypoint is used; when
    it is a worker thread (the API's background job) the signal handlers
    are skipped because Python only allows ``signal.signal`` on the main
    thread. The hospital nodes always run on background threads and wait
    for the aggregator's port to open.
    """
    from .server import run_federated

    if not (Path(DATA_ROOT) / "manifest.json").exists():
        raise FileNotFoundError(
            "Dataset not prepared. Run `python -m fedmed.prepare_data` first."
        )

    if use_tls:
        ca = Path(CONFIG.server.ca_cert)
        cert = Path(CONFIG.server.server_cert)
        key = Path(CONFIG.server.server_key)
        missing = [str(p) for p in (ca, cert, key) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"TLS material incomplete; missing: {', '.join(missing)}. "
                "Run python -m fedmed.tls or pass --insecure."
            )

    server_address = address or CONFIG.server.fl_address
    num_rounds = rounds if rounds is not None else CONFIG.federated.num_rounds

    _assert_ports_free()

    # 1. Start the hospital nodes. They block until the aggregator's port
    #    is accepting connections.
    threads = []
    for hospital in CONFIG.hospitals:
        ready = threading.Event()
        thread = threading.Thread(
            target=_node_thread,
            args=(
                hospital.hospital_id,
                hospital.node_id,
                server_address,
                use_tls,
                ready,
                seed,
            ),
            daemon=True,
            name=f"fedmed-{hospital.hospital_id}",
        )
        threads.append(thread)
        thread.start()

    # 2. Run the aggregator on the main thread.
    summary = run_federated(
        rounds=num_rounds,
        address=server_address,
        dropout_round=dropout_round,
        enable_he=enable_he,
        enable_dp=enable_dp,
        use_tls=use_tls,
        install_signal_handlers=install_signal_handlers,
        seed=seed,
    )

    # 3. Let the nodes finish their shutdown.
    for thread in threads:
        thread.join(timeout=5)

    summary["he_available"] = he_mod.TENSEAL_AVAILABLE
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed cluster orchestrator")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--dropout-round", type=int, default=None)
    parser.add_argument("--no-he", action="store_true")
    parser.add_argument("--no-dp", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to FEDMED_SEED or 20260101)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    summary = run_cluster(
        rounds=args.rounds,
        address=args.address,
        dropout_round=args.dropout_round,
        enable_he=not args.no_he,
        enable_dp=not args.no_dp,
        use_tls=not args.insecure,
        seed=args.seed,
    )

    print()
    print("=" * 68)
    print(f"FEDERATED RUN COMPLETE  ({summary.get('rounds_completed', 0)} rounds)")
    print("=" * 68)
    print(f"  final dice : {summary.get('federated_final_dice', 0):.2f}%")
    print(f"  baseline   : {summary.get('baseline_best_dice', 0):.2f}%")
    print(f"  ratio      : {summary.get('dice_ratio_pct', 0):.1f}%")
    print()


if __name__ == "__main__":
    main()
