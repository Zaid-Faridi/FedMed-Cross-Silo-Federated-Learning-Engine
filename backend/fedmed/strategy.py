"""Federated server strategy with homomorphic aggregation.

Implements FedAvg but replaces the plaintext weighted-average with a
**homomorphic** one::

    aggregate = Dec( sum_i ct_i )

where ``ct_i`` is the CKKS ciphertext produced by hospital node *i*. Each
client pre-scales its update by its FedAvg weight before encrypting (the
weight is derived from sample counts, which are public), so the server
performs only ciphertext addition -- exact in CKKS -- and decrypts only the
aggregate. The server therefore never observes an individual plaintext
update.

Identifying the nodes
---------------------
Flower gives each connection an opaque UUID, so the strategy asks every
client for its properties once per round and caches the mapping from the
Flower proxy id to the hospital it represents. That mapping is what lets
the dashboard attribute a round to a specific hospital.

Handling a node that drops offline mid-round
---------------------------------------------
        The surviving clients' weights sum to less than one. The server knows each
        participating client's weight (public metadata), so it re-normalises the
        *decrypted aggregate* by the weight it actually received. This keeps the
        update mathematically identical to FedAvg over the participants without
        ever needing to scale an individual ciphertext (note: the aggregator holds
        the secret key and possesses the capability to decrypt individual updates,
        but follows the protocol to decrypt only the aggregate).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.compat.common import typing as fct
from flwr.compat.server.app import Strategy

from .config import CONFIG, DATA_ROOT, MODEL_ROOT
from .metrics import evaluate_model
from .model import build_model, load_state_dict
from .privacy import dp as dp_mod
from .privacy import he as he_mod
from .state import STATE, NodeStatus

logger = logging.getLogger(__name__)

#: Metric key flagging an encrypted payload coming back from a client.
HE_METRIC = "encrypted"


def _node_sample_counts() -> dict:
    """Read each hospital's sample count from the dataset manifest."""
    manifest_path = DATA_ROOT / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    return {
        hospital_id: info.get("count", 0)
        for hospital_id, info in manifest.get("train", {}).items()
    }


class FederatedStrategy(Strategy):
    """FedAvg variant that aggregates homomorphically and audits each round."""

    def __init__(
        self,
        num_rounds: int | None = None,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        accept_failures: bool = True,
        dropout_round: int = 0,
        dropout_node: int = 3,
        enable_he: bool | None = None,
        enable_dp: bool | None = None,
    ) -> None:
        cfg = CONFIG.federated
        self.num_rounds = num_rounds if num_rounds is not None else cfg.num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.dropout_round = dropout_round
        self.dropout_node = dropout_node

        self.enable_he = (
            enable_he if enable_he is not None else CONFIG.privacy.enable_he
        )
        self.enable_dp = (
            enable_dp if enable_dp is not None else CONFIG.privacy.enable_dp
        )

        self._round = 0
        self._sample_counts = _node_sample_counts()
        self.node_to_hospital = {h.node_id: h.hospital_id for h in CONFIG.hospitals}
        #: Maps a Flower proxy id to the hospital id that owns it.
        self._proxy_to_hospital: dict[str, str] = {}
        self._hospital_identity: dict[str, dict] = {}

        # Server-side HE context: encrypts nothing, but decrypts the
        # homomorphic aggregate. The *same* context is shared with the
        # hospital nodes so their ciphertexts remain addable.
        self._server_he = he_mod.create_session("fedmed-server")
        self._context_blob: bytes | None = None

        self._dp_accountant = dp_mod.DPAccountant(
            noise_multiplier=CONFIG.privacy.dp_noise_multiplier,
            clip_norm=CONFIG.privacy.dp_clip_norm,
            delta=CONFIG.privacy.dp_delta,
            participation_ratio=1.0,
            sample_rate=CONFIG.privacy.dp_sample_rate,
        )

        self.global_model = build_model()
        self.reference_state = {
            k: v.detach().cpu().clone()
            for k, v in self.global_model.state_dict().items()
        }

        logger.info(
            "Strategy ready: rounds=%d HE=%s DP=%s dropout_round=%s",
            self.num_rounds,
            "on" if (self.enable_he and self._server_he.available) else "off",
            "on" if self.enable_dp else "off",
            self.dropout_round or "none",
        )

    # -- lifecycle -----------------------------------------------------
    def initialize_parameters(self, client_manager: Any) -> fct.Parameters:
        """Return the initial global model to broadcast."""
        return self._to_parameters(self.reference_state)

    def configure_fit(
        self, server_round: int, parameters: fct.Parameters, client_manager: Any
    ) -> list:
        """Select clients for this round and broadcast the training config."""
        sample_size, min_num_clients = self._num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Resolve the identity of every connected client.
        self._resolve_identities(clients)

        # Node resilience: exclude the victim node from the round.
        if self.dropout_round and server_round == self.dropout_round:
            victim_hospital = self._hospital_for_index(self.dropout_node)
            kept = []
            for client in clients:
                hospital_id = self._hospital_for_proxy(client)
                if hospital_id == victim_hospital:
                    STATE.set_node_status(
                        hospital_id,
                        NodeStatus.OFFLINE,
                        "Node dropped offline (simulated)",
                    )
                    STATE.log_security_event(
                        "node_offline",
                        {
                            "hospital_id": hospital_id,
                            "node_id": self.node_to_hospital.get(hospital_id),
                            "round": server_round,
                        },
                    )
                    logger.warning(
                        "Round %d: excluding %s (simulated offline)",
                        server_round,
                        hospital_id,
                    )
                    continue
                kept.append(client)
            clients = kept

        # Total samples across the selected nodes -> lets each client compute
        # its FedAvg weight before encrypting.
        total_examples = 0
        for client in clients:
            hospital_id = self._hospital_for_proxy(client)
            if hospital_id:
                total_examples += int(self._sample_counts.get(hospital_id, 0))

        config = {
            "round": server_round,
            "local_epochs": CONFIG.federated.local_epochs,
            "learning_rate": CONFIG.federated.learning_rate,
            "batch_size": CONFIG.federated.batch_size,
            "total_examples": float(total_examples or 0),
            "dropout_round": float(self.dropout_round or 0),
            "dropout_node": self._hospital_for_index(self.dropout_node),
            # The shared CKKS context. Every hospital node must encrypt
            # under the *same* key, otherwise the aggregator cannot add
            # their ciphertexts together.
            "he_context": self._shared_context_blob(),
        }

        fit_ins = fct.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int, results: list, failures: list
    ) -> tuple:
        """Aggregate client updates, homomorphically when encrypted."""
        self._round = server_round
        round_index = self._round

        for _, failure in failures or []:
            logger.warning("Round %d: client failure: %s", round_index, failure)

        if not results:
            logger.warning("Round %d: no client updates received", round_index)
            STATE.log_security_event(
                "round_no_updates", {"round": round_index}
            )
            return self._to_parameters(self.reference_state), {}

        payloads: list[bytes] = []
        weights: list[float] = []
        node_ids: list[str] = []
        client_metrics: dict = {}
        encrypted_count = 0
        total_examples = 0
        dp_steps = 0

        for _, fit in results:
            metrics = dict(fit.metrics or {})
            node_id = str(metrics.get("node_id") or "unknown")
            payloads.append(fit.parameters.tensors[0])
            node_ids.append(node_id)
            client_metrics[node_id] = metrics
            weights.append(float(metrics.get("aggregation_weight") or 0.0))
            total_examples += max(int(fit.num_examples or 0), 0)
            if metrics.get(HE_METRIC):
                encrypted_count += 1
            dp_steps += int(metrics.get("dp_steps") or 0)

        weight_sum = sum(weights)

        # Fail-closed HE check: if homomorphic encryption is expected, reject round
        # if any client submitted an unencrypted update or server HE is unavailable.
        if self.enable_he:
            if encrypted_count < len(payloads) or not self._server_he.available:
                reason = (
                    "unencrypted_update_received"
                    if encrypted_count < len(payloads)
                    else "server_he_unavailable"
                )
                logger.error(
                    "Round %d rejected: HE enabled but %d/%d updates encrypted (server HE: %s)",
                    server_round,
                    encrypted_count,
                    len(payloads),
                    self._server_he.available,
                )
                STATE.log_security_event(
                    "round_rejected",
                    {
                        "round": server_round,
                        "reason": reason,
                        "encrypted_count": encrypted_count,
                        "total_payloads": len(payloads),
                    },
                )
                # Keep global model unchanged; return current parameters and empty metrics
                return self._to_parameters(self.reference_state), {}

        use_he = (
            self.enable_he
            and encrypted_count == len(payloads)
            and self._server_he.available
        )

        aggregation_mode: str
        if use_he:
            try:
                vector = self._homomorphic_aggregate(payloads)
                if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
                    # A node dropped mid-round: re-normalise the aggregate by
                    # the weight actually received.
                    vector = vector / weight_sum
                aggregated_state = he_mod.flat_vector_to_tensor(
                    vector, self.reference_state
                )
                aggregation_mode = "homomorphic_ckks"
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("HE aggregation failed (%s); rejecting round (fail-closed)", exc)
                STATE.log_security_event(
                    "round_rejected",
                    {
                        "round": server_round,
                        "reason": "he_aggregation_failed",
                        "error": str(exc),
                    },
                )
                return self._to_parameters(self.reference_state), {}
        else:
            aggregated_state = self._plaintext_aggregate(payloads, weights)
            aggregation_mode = "plaintext"

        # Apply the aggregated delta to the global model.
        new_state = {}
        for key, ref in self.reference_state.items():
            new_state[key] = (
                self.reference_state[key].to(torch.float64)
                + aggregated_state[key].to(torch.float64)
            ).to(dtype=ref.dtype)

        self.reference_state = new_state
        load_state_dict(self.global_model, new_state)

        if self.enable_dp:
            # Charge the privacy budget for the optimiser steps that ran.
            self._dp_accountant.spend(1)
            self._dp_accountant.spend_steps(dp_steps)
            for node_id, metrics in client_metrics.items():
                hospital_id = self.node_to_hospital.get(node_id)
                if hospital_id:
                    steps_node = int(metrics.get("dp_steps") or 0)
                    self._dp_accountant.spend_node_steps(hospital_id, steps_node)

            STATE.dp_epsilon = self._dp_accountant.epsilon
            STATE.dp_rounds = self._dp_accountant.rounds
            STATE.dp_per_node = self._dp_accountant.get_per_node_dict()

        eval_metrics = self._evaluate_global()

        for node_id, metrics in client_metrics.items():
            hospital_id = self.node_to_hospital.get(node_id)
            if hospital_id is None:
                continue
            node = STATE.nodes[hospital_id]
            STATE.update_node(
                hospital_id,
                rounds_participated=node["rounds_participated"] + 1,
                local_loss=metrics.get("local_loss"),
                local_dice=metrics.get("local_dice"),
                encrypted=bool(metrics.get(HE_METRIC)),
                payload_bytes=int(metrics.get("payload_bytes") or 0),
                status=NodeStatus.ONLINE,
                last_sync=time.time(),
                message=f"Participated in round {round_index}",
            )

        round_record = {
            "round": round_index,
            "loss": eval_metrics["loss"],
            "dice": eval_metrics["dice"],
            "accuracy": eval_metrics["accuracy"],
            "per_class_dice": eval_metrics["per_class_dice"],
            "clients": len(results),
            "failed_clients": len(failures or []),
            "participating_nodes": node_ids,
            "participating_hospitals": [
                self.node_to_hospital.get(n, n) for n in node_ids
            ],
            "aggregation": aggregation_mode,
            "encrypted_updates": encrypted_count,
            "dp_epsilon": self._dp_accountant.epsilon,
            "dp_delta": self._dp_accountant.delta,
            "total_examples": total_examples,
            "timestamp": time.time(),
            "client_metrics": client_metrics,
        }
        STATE.add_round(round_record)

        STATE.log_security_event(
            "aggregation_completed",
            {
                "round": round_index,
                "mode": aggregation_mode,
                "encrypted_updates": encrypted_count,
                "no_raw_data_transferred": True,
            },
        )

        self._persist_audit_trail()

        logger.info(
            "Round %d: dice=%.2f%% loss=%.4f clients=%d mode=%s",
            round_index,
            eval_metrics["dice"],
            eval_metrics["loss"],
            len(results),
            aggregation_mode,
        )

        return self._to_parameters(new_state), round_record

    def evaluate(
        self, server_round: int, parameters: fct.Parameters
    ) -> tuple:
        """Evaluate the broadcast global model on the pooled validation set."""
        metrics = self._evaluate_global()
        return metrics["loss"], {
            "dice": metrics["dice"],
            "accuracy": metrics["accuracy"],
            "per_class_dice": metrics["per_class_dice"],
        }

    def configure_evaluate(
        self, server_round: int, parameters: fct.Parameters, client_manager: Any
    ) -> list:
        """Every available client evaluates the broadcast global model."""
        sample_size, min_num_clients = self._num_evaluate_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        evaluate_ins = fct.EvaluateIns(parameters, {"round": server_round})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self, server_round: int, results: list, failures: list
    ) -> tuple:
        """Average the per-node evaluation metrics."""
        if not results:
            return None, {}

        total = sum(max(int(r.num_examples or 1), 1) for _, r in results)
        loss = sum(
            float((r.metrics or {}).get("loss", 0.0))
            * max(int(r.num_examples or 1), 1)
            for _, r in results
        ) / max(total, 1)
        dice = sum(
            float((r.metrics or {}).get("dice", 0.0))
            * max(int(r.num_examples or 1), 1)
            for _, r in results
        ) / max(total, 1)

        return round(loss, 6), {
            "dice": round(dice, 4),
            "clients": len(results),
            "failed_clients": len(failures or []),
        }

    # -- persistence ---------------------------------------------------
    def _persist_audit_trail(self) -> None:
        """Write the audit trail and DP accounting to disk.

        The privacy audit log is the evidence that the privacy mechanisms
        actually ran, so it has to survive an API restart.
        """
        try:
            path = Path(MODEL_ROOT) / "audit_log.json"
            with open(path, "w") as fh:
                json.dump(
                    {
                        **STATE.dump_events(),
                        "dp_accountant": self._dp_accountant.to_dict(),
                    },
                    fh,
                    indent=2,
                )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist the audit trail: %s", exc)

    # -- homomorphic encryption context -------------------------------
    def _shared_context_blob(self) -> bytes:
        """Serialise the cluster-wide CKKS context for distribution.

        In TenSEAL CKKS, every node has to encrypt under the exact same
        key pair distributed by the aggregator. The blob is cached because
        serialising a 16k-degree context is comparatively expensive.
        """
        if not self._server_he.available:
            return b""
        if self._context_blob is None:
            self._context_blob = he_mod.serialize_context(self._server_he)
            logger.info(
                "Shared CKKS context ready (%d bytes)", len(self._context_blob)
            )
        return self._context_blob

    # -- client identity ----------------------------------------------
    def _resolve_identities(self, clients: list) -> None:
        """Ask each client for its properties and cache the mapping."""
        for client in clients:
            proxy_id = self._proxy_id(client)
            if not proxy_id or proxy_id in self._proxy_to_hospital:
                continue
            try:
                res = client.get_properties(
                    fct.GetPropertiesIns({}), timeout=30.0, group_id=None
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("get_properties failed for %s: %s", proxy_id, exc)
                continue

            properties = dict(getattr(res, "properties", {}) or {})
            node_id = str(properties.get("node_id") or "")
            hospital_id = str(properties.get("hospital_id") or "")
            if not hospital_id and node_id:
                hospital_id = self.node_to_hospital.get(node_id, "")

            if hospital_id:
                self._proxy_to_hospital[proxy_id] = hospital_id
                self._hospital_identity[hospital_id] = properties
                STATE.update_node(
                    hospital_id,
                    node_id=node_id or STATE.nodes[hospital_id]["node_id"],
                    samples=int(properties.get("samples") or 0),
                    encrypted=bool(properties.get("encrypted")),
                )
                logger.info(
                    "Mapped Flower client %s -> %s (%s)",
                    proxy_id[:8],
                    hospital_id,
                    node_id,
                )

    def _proxy_id(self, client: Any) -> str:
        cid = getattr(client, "cid", None)
        if cid:
            return str(cid)
        return str(id(client))

    def _hospital_for_proxy(self, client: Any) -> str:
        """Resolve the hospital id behind a Flower client proxy."""
        proxy_id = self._proxy_id(client)
        hospital_id = self._proxy_to_hospital.get(proxy_id)
        if hospital_id:
            return hospital_id
        # Fall back to the identity reported in the round's metrics.
        return ""

    def _hospital_for_index(self, index: int) -> str:
        hospitals = list(CONFIG.hospitals)
        if 1 <= index <= len(hospitals):
            return hospitals[index - 1].hospital_id
        return hospitals[0].hospital_id

    # -- aggregation helpers ------------------------------------------
    def _num_fit_clients(self, num_available: int) -> tuple:
        num_clients = int(num_available * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def _num_evaluate_clients(self, num_available: int) -> tuple:
        num_clients = int(num_available * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def _homomorphic_aggregate(self, payloads: list) -> np.ndarray:
        """Sum CKKS ciphertexts on the server, then decrypt the aggregate."""
        import tenseal as ts

        ctx = self._server_he.server_context or self._server_he.context
        acc = None
        for payload in payloads:
            vector = ts.CKKSVector.load(ctx, payload)
            acc = vector if acc is None else (acc + vector)

        if acc is None:
            raise ValueError("no ciphertext payloads to aggregate")

        return np.array(acc.decrypt())

    def _plaintext_aggregate(self, payloads: list, weights: list) -> dict:
        """Standard weighted average (fallback path)."""
        total = sum(weights) or 1.0
        buffer: np.ndarray | None = None

        for payload, weight in zip(payloads, weights):
            flat = np.frombuffer(payload, dtype=np.float32).astype(np.float64)
            contribution = flat * (weight / total)
            buffer = contribution if buffer is None else buffer + contribution

        return he_mod.flat_vector_to_tensor(buffer, self.reference_state)

    def _evaluate_global(self) -> dict:
        """Evaluate the global model on the pooled validation set."""
        from .data import load_global_validation, make_loader

        try:
            val_x, val_y = load_global_validation()
        except FileNotFoundError:
            return {"dice": 0.0, "loss": 0.0, "accuracy": 0.0, "per_class_dice": {}}

        loader = make_loader(val_x, val_y, batch_size=2, shuffle=False)
        return evaluate_model(self.global_model, loader, CONFIG.model.num_classes)

    def _to_parameters(self, state: dict) -> fct.Parameters:
        flat = he_mod.tensor_to_flat_vector(state).astype(np.float32)
        return fct.Parameters(tensors=[flat.tobytes()], tensor_type="float32")
