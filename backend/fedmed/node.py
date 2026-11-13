"""Hospital node entrypoint.

Runs a single hospital node: the Flower client (which trains the model on
local data) plus the gRPC/TLS coordination server.

Usage:
    python -m fedmed.node --hospital hospital-1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import grpc
from flwr.compat.client.app import start_client

from .config import CONFIG, DATA_ROOT
from .client import HospitalClient
from .grpc_service import channel_credentials, serve
from .state import STATE, NodeStatus

logger = logging.getLogger(__name__)


def probe_server(address: str, timeout: float = 5.0) -> bool:
    """Check that the central aggregator is reachable over TLS."""
    try:
        with grpc.secure_channel(address, channel_credentials()) as channel:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        return True
    except Exception as exc:
        logger.debug("Server probe failed for %s: %s", address, exc)
        return False


def run_node(
    hospital_id: str,
    server_address: str | None = None,
    wait_for_server: bool = True,
    use_tls: bool = True,
    seed: int | None = None,
) -> None:
    """Start a hospital node and block until it disconnects."""
    hospital = next(h for h in CONFIG.hospitals if h.hospital_id == hospital_id)

    if not (hospital.data_dir / "train.pt").exists():
        raise FileNotFoundError(
            f"Node data not found at {hospital.data_dir}. "
            "Run `python -m fedmed.prepare_data` first."
        )

    logger.info(
        "Starting %s (%s) on gRPC port %d", hospital.hospital_id, hospital.node_id,
        hospital.port,
    )

    # 1. gRPC/TLS coordination server.
    grpc_server = serve(
        hospital.hospital_id, hospital.node_id, hospital.port, cert_root=None
    )

    # 2. Wait for the aggregator.
    address = server_address or CONFIG.server.fl_address
    if wait_for_server:
        for attempt in range(60):
            if probe_server(address):
                logger.info("[%s] aggregator reachable at %s", hospital.node_id, address)
                break
            if attempt % 10 == 0 and attempt:
                logger.info("[%s] waiting for aggregator...", hospital.node_id)
            time.sleep(1)
        else:
            logger.warning(
                "[%s] aggregator not reachable; starting anyway", hospital.node_id
            )

    # 3. Flower client.
    client = HospitalClient(hospital.hospital_id, hospital.node_id, seed=seed)
    STATE.set_node_status(
        hospital.hospital_id, NodeStatus.ONLINE, "Client started"
    )

    try:
        if use_tls:
            start_client(
                server_address=address,
                client=client,
                root_certificates=_root_certificates(),
                insecure=False,
            )
        else:
            start_client(server_address=address, client=client, insecure=True)
    except KeyboardInterrupt:
        logger.info("[%s] shutting down", hospital.node_id)
    finally:
        grpc_server.stop(grace=2)
        STATE.set_node_status(
            hospital.hospital_id, NodeStatus.OFFLINE, "Client disconnected"
        )


def _root_certificates() -> bytes | None:
    """Return the CA certificate used to verify the aggregator."""
    ca = Path(CONFIG.server.ca_cert)
    if ca.exists():
        return ca.read_bytes()
    logger.warning("CA certificate missing at %s; TLS verification unavailable", ca)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed hospital node")
    parser.add_argument(
        "--hospital",
        type=str,
        required=True,
        choices=[h.hospital_id for h in CONFIG.hospitals],
    )
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to FEDMED_SEED or 20260101)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    run_node(
        args.hospital,
        server_address=args.server,
        wait_for_server=not args.no_wait,
        use_tls=not args.insecure,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
