"""FedMed central aggregator entrypoint.

Runs the Flower federated server (FedAvg with homomorphic aggregation) on
top of the gRPC/TLS coordination plane.

Usage:
    python -m fedmed.server --rounds 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from flwr.compat.server.app import ServerConfig, start_server

from .config import CONFIG, DATA_ROOT, MODEL_ROOT
from .seed import set_seed
from .state import STATE
from .strategy import FederatedStrategy

logger = logging.getLogger(__name__)


def _load_baseline() -> dict:
    """Load the centralized baseline metrics if the baseline has been run."""
    path = Path(MODEL_ROOT) / "baseline_metrics.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    STATE.baseline = data
    return data


def _server_certificates(use_tls: bool = True) -> tuple[bytes, bytes, bytes] | None:
    """Load the (CA, server certificate, server key) triple for TLS.

    Flower's gRPC transport expects exactly this ordering. When TLS is enabled,
    missing certificates raise FileNotFoundError (fail-closed).
    """
    if not use_tls:
        return None

    ca = Path(CONFIG.server.ca_cert)
    cert = Path(CONFIG.server.server_cert)
    key = Path(CONFIG.server.server_key)

    missing = [str(p) for p in (ca, cert, key) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"TLS material incomplete; missing: {', '.join(missing)}. "
            "Run python -m fedmed.tls or pass --insecure."
        )

    return (ca.read_bytes(), cert.read_bytes(), key.read_bytes())


def _write_summary(summary: dict) -> None:
    """Persist the federated run summary for the API and the dashboard."""
    path = Path(MODEL_ROOT) / "federated_summary.json"
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)


def _build_summary(history: list, test_metrics: dict | None = None, best_val_dice: float | None = None, best_round: int | None = None) -> dict:
    """Compare the federated run against the centralized baseline."""
    baseline = STATE.baseline or {}
    final = history[-1] if history else {}

    fed_dice = float(final.get("dice", 0.0))
    base_dice = float(baseline.get("best_dice", 0.0))
    base_test_dice = float(baseline.get("test_dice", 0.0))

    val_best = float(best_val_dice) if best_val_dice is not None and best_val_dice >= 0 else max((r.get("dice", 0.0) for r in history), default=0.0)
    fed_test_dice = float((test_metrics or {}).get("dice", 0.0))

    gap = 0.0
    if base_dice > 0:
        gap = round(base_dice - val_best, 4)

    test_gap = 0.0
    if base_test_dice > 0:
        test_gap = round(base_test_dice - fed_test_dice, 4)

    ratio = 0.0
    if base_dice > 0:
        ratio = round((val_best / base_dice) * 100, 2)

    return {
        "federated_final_dice": fed_dice,
        "federated_final_loss": final.get("loss", 0.0),
        "federated_final_accuracy": final.get("accuracy", 0.0),
        "federated_best_dice": val_best,
        "federated_best_round": best_round or 0,
        "federated_test_dice": fed_test_dice,
        "test_dice": fed_test_dice,
        "val_dice": val_best,
        "test_metrics": test_metrics or {},
        "baseline_best_dice": base_dice,
        "baseline_test_dice": base_test_dice,
        "baseline_final_loss": baseline.get("final_loss", 0.0),
        "dice_gap": gap,
        "test_dice_gap": test_gap,
        "dice_ratio_pct": ratio,
        "rounds_completed": len(history),
        "converged": bool(base_dice == 0.0 or gap <= 5.0),
    }


def _run_aggregator_inline(
    server_address: str,
    config: ServerConfig,
    strategy: FederatedStrategy,
    certificates: tuple[bytes, bytes, bytes] | None,
) -> None:
    """Start the Flower gRPC server and training loop without signal handlers.

    This is the thread-safe variant of :func:`flwr.compat.server.app.start_server`.
    It performs exactly the same work minus the ``SIGINT``/``SIGTERM``
    registration, which Python forbids outside the main thread.
    """
    from flwr.server.server import init_defaults, run_fl
    from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server

    initialized_server, initialized_config = init_defaults(
        server=None,
        config=config,
        strategy=strategy,
        client_manager=None,
    )

    grpc_server = start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=server_address,
        certificates=certificates,
    )
    logger.info("Flower aggregator listening on %s (TLS=%s)", server_address,
                "on" if certificates else "off")

    try:
        run_fl(server=initialized_server, config=initialized_config)
    finally:
        grpc_server.stop(grace=1)


def run_federated(
    rounds: int | None = None,
    address: str | None = None,
    dropout_round: int | None = None,
    enable_he: bool = True,
    enable_dp: bool = True,
    use_tls: bool = True,
    install_signal_handlers: bool = True,
    seed: int | None = None,
) -> dict:
    """Run the federated training loop and return the summary.

    Args:
        rounds: Number of federated rounds to run.
        address: Address of the Flower aggregator.
        dropout_round: Round at which a hospital node is dropped offline.
        enable_he: Encrypt the model updates with CKKS.
        enable_dp: Apply DP-SGD on the hospital nodes.
        use_tls: Serve the aggregator over mutual TLS.
        install_signal_handlers: Set to ``False`` when running off the
            main thread (the API's background job) -- Flower's signal
            handlers can only be installed on the main thread.
        seed: Random seed (defaults to FEDMED_SEED or 20260101).
    """
    effective_seed = set_seed(seed)
    if not (DATA_ROOT / "manifest.json").exists():
        raise FileNotFoundError(
            "Dataset not prepared. Run `python -m fedmed.prepare_data` first."
        )

    _load_baseline()

    num_rounds = rounds if rounds is not None else CONFIG.federated.num_rounds
    server_address = address or CONFIG.server.fl_address

    strategy = FederatedStrategy(
        num_rounds=num_rounds,
        dropout_round=(
            dropout_round
            if dropout_round is not None
            else CONFIG.federated.dropout_round
        ),
        enable_he=enable_he,
        enable_dp=enable_dp,
    )

    STATE.mark_started()
    STATE.total_rounds = num_rounds

    config = ServerConfig(num_rounds=num_rounds)

    # Secure transport: the aggregator presents the server certificate and
    # the hospital nodes verify it against the local CA. Missing certificates
    # raise FileNotFoundError unless use_tls is False.
    certificates = _server_certificates(use_tls=use_tls)

    logger.info(
        "Starting federated server on %s for %d rounds (TLS=%s)",
        server_address,
        num_rounds,
        "on" if certificates else "off",
    )

    try:
        if install_signal_handlers:
            start_server(
                server_address=server_address,
                config=config,
                strategy=strategy,
                certificates=certificates,
            )
        else:
            # ``start_server`` registers process-wide signal handlers, which
            # Python only permits on the main thread. The API runs the
            # aggregator inside a worker thread, so the gRPC server and the
            # training loop are started directly instead.
            _run_aggregator_inline(
                server_address=server_address,
                config=config,
                strategy=strategy,
                certificates=certificates,
            )
    except Exception as exc:
        logger.error("Federated server failed: %s", exc)
        STATE.mark_finished()
        raise

    rounds_data = STATE.rounds_snapshot()
    test_metrics = strategy.evaluate_test()
    summary = _build_summary(
        rounds_data,
        test_metrics=test_metrics,
        best_val_dice=strategy.best_val_dice,
        best_round=strategy.best_round,
    )
    summary["history"] = rounds_data

    # Persist the trained global model so the MRI page can segment with it.
    try:
        import torch

        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in strategy.best_state.items()
                },
                "config": {
                    "rounds": num_rounds,
                    "enable_he": enable_he,
                    "enable_dp": enable_dp,
                },
                "metrics": {
                    "final_dice": summary["federated_final_dice"],
                    "best_val_dice": summary["federated_best_dice"],
                    "best_round": summary["federated_best_round"],
                    "test_dice": summary["federated_test_dice"],
                    "test_loss": test_metrics.get("loss", 0.0),
                    "test_accuracy": test_metrics.get("accuracy", 0.0),
                    "final_loss": summary["federated_final_loss"],
                    "final_accuracy": summary["federated_final_accuracy"],
                },
            },
            Path(MODEL_ROOT) / "federated_model.pt",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not persist the federated model: %s", exc)

    STATE.mark_finished(summary)
    _write_summary(summary)

    logger.info(
        "Federated run complete: best val dice=%.2f%% (round %d), test dice=%.2f%% (baseline val %.2f%%, test %.2f%%)",
        summary["federated_best_dice"],
        summary["federated_best_round"],
        summary["federated_test_dice"],
        summary["baseline_best_dice"],
        summary["baseline_test_dice"],
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed federated server")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--dropout-round", type=int, default=None)
    parser.add_argument(
        "--no-he", action="store_true", help="Disable homomorphic encryption"
    )
    parser.add_argument(
        "--no-dp", action="store_true", help="Disable differential privacy"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to FEDMED_SEED or 20260101)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    run_federated(
        rounds=args.rounds,
        address=args.address,
        dropout_round=args.dropout_round,
        enable_he=not args.no_he,
        enable_dp=not args.no_dp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
