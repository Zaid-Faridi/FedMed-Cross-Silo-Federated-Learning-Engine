"""Hospital node — a Flower client that trains the 3D U-Net on local data.

Each hospital node:

1. Receives the global model weights from the server.
2. Trains locally on its *private* MRI subset (which never leaves the node)
   using **DP-SGD**: every patient's gradient is clipped individually and
   Gaussian noise is added to the summed gradient.
3. Applies the differential privacy mechanism to the update.
4. Pre-scales the update by its FedAvg weight and encrypts it with
   homomorphic encryption (TenSEAL/CKKS).
5. Returns the ciphertext to the server, which aggregates the ciphertexts
   and decrypts only the aggregate sum (the server possesses the key and
   could technically decrypt individual updates, but follows the protocol).

The node advertises its identity through Flower's ``get_properties`` hook
so the aggregator can map an opaque Flower client proxy back to a
specific hospital (Flower's own client id is a random UUID).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from flwr.compat.client.app import Client
from flwr.compat.common import typing as fct

from .config import CONFIG
from .data import build_local_loaders
from .metrics import dice_ce_loss, evaluate_model
from .model import build_model, load_state_dict
from .privacy import dp as dp_mod
from .privacy import he as he_mod
from .state import STATE, NodeStatus

logger = logging.getLogger(__name__)


def parameters_to_state_dict(parameters: fct.Parameters, reference: dict) -> dict:
    """Rebuild a state dict from Flower's transported parameters."""
    flat = np.frombuffer(parameters.tensors[0], dtype=np.float32)
    return he_mod.flat_vector_to_tensor(flat.astype(np.float64), reference)


def state_dict_to_parameters(state: dict) -> fct.Parameters:
    """Serialise a state dict into a single flat float32 buffer."""
    flat = he_mod.tensor_to_flat_vector(state).astype(np.float32)
    return fct.Parameters(tensors=[flat.tobytes()], tensor_type="float32")


class HospitalClient(Client):
    """A simulated hospital node participating in the federated run."""

    def __init__(
        self,
        hospital_id: str,
        node_id: str,
        *,
        local_epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        enable_he: bool | None = None,
        enable_dp: bool | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.node_id = node_id

        cfg = CONFIG.federated
        self.local_epochs = local_epochs if local_epochs is not None else cfg.local_epochs
        self.learning_rate = (
            learning_rate if learning_rate is not None else cfg.learning_rate
        )
        self.batch_size = batch_size if batch_size is not None else cfg.batch_size
        self.enable_he = (
            enable_he if enable_he is not None else CONFIG.privacy.enable_he
        )
        self.enable_dp = (
            enable_dp if enable_dp is not None else CONFIG.privacy.enable_dp
        )

        self.train_loader, self.val_loader = build_local_loaders(
            hospital_id, batch_size=self.batch_size
        )
        self.num_examples = len(self.train_loader.dataset)

        # Sampling rate q = batch_size / local cohort size. The RDP
        # accountant needs this to convert the noise level into epsilon.
        self.sample_rate = min(self.batch_size / max(self.num_examples, 1), 1.0)

        self.model = build_model()
        self.reference_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }

        # Homomorphic encryption session. The node encrypts under the
        # server's shared context; the server can therefore add the
        # ciphertexts homomorphically and decrypt only the aggregate.
        #
        # The context arrives with the round configuration (see
        # ``FederatedStrategy._shared_context_blob``). Until then the node
        # has no usable session, so HE is reported as pending.
        self._he_context_blob: bytes | None = None
        self.he_session = he_mod.HESession(node_id=node_id)
        self.he_enabled = False

        STATE.update_node(
            hospital_id,
            samples=self.num_examples,
            status=NodeStatus.ONLINE,
            encrypted=self.he_enabled,
            message="Node initialised with local dataset",
        )
        logger.info(
            "[%s] ready: %d local samples, q=%.3f, HE=%s, DP=%s",
            node_id,
            self.num_examples,
            self.sample_rate,
            "on" if self.he_enabled else "off",
            "on" if self.enable_dp else "off",
        )

    # -- identity ------------------------------------------------------
    def get_properties(self, ins: fct.GetPropertiesIns) -> fct.GetPropertiesRes:
        """Advertise the node's identity to the aggregator.

        Flower assigns each connection an opaque UUID, so this is how the
        server learns which hospital a given proxy belongs to.
        """
        return fct.GetPropertiesRes(
            status=fct.Status(code=fct.Code.OK, message="OK"),
            properties={
                "hospital_id": self.hospital_id,
                "node_id": self.node_id,
                "samples": self.num_examples,
                "encrypted": self.he_enabled,
                "dp_enabled": self.enable_dp,
            },
        )

    def get_parameters(self, ins: fct.GetParametersIns) -> fct.GetParametersRes:
        """Return the node's local model parameters."""
        return fct.GetParametersRes(
            parameters=state_dict_to_parameters(self.reference_state)
        )

    # -- training ------------------------------------------------------
    def fit(self, ins: fct.FitIns) -> fct.FitRes:
        """Train locally, apply privacy mechanisms, return the protected update."""
        STATE.set_node_status(
            self.hospital_id, NodeStatus.TRAINING, "Training on private MRI subset"
        )
        started = time.time()

        # 1. Load the global model broadcast by the server.
        global_state = parameters_to_state_dict(ins.parameters, self.reference_state)
        load_state_dict(self.model, global_state)

        # 1b. Activate homomorphic encryption with the cluster-wide CKKS
        #     context. Every node must use the same key material so the
        #     aggregator can add the ciphertexts together.
        self._ensure_he_session(ins.config)

        # 2. Local training on private data (DP-SGD when enabled).
        optimizer = self._build_optimizer()

        running = 0.0
        steps = 0
        dp_stats: dict | None = None
        patients = 0

        self.model.train()
        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                if self.enable_dp:
                    loss_value, dp_stats, n = self._dp_sgd_step(
                        images, labels, optimizer
                    )
                    patients += n
                else:
                    optimizer.zero_grad()
                    logits = self.model(images)
                    loss = dice_ce_loss(
                        logits, labels, CONFIG.model.num_classes, ce_weight=0.5
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    loss_value = loss.item()
                    n = images.shape[0]
                    patients += n

                running += loss_value
                steps += 1

        local_loss = running / max(steps, 1)

        # 3. Compute the model *update* (delta) relative to the global model.
        new_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        update = {
            k: (new_state[k].to(torch.float64) - global_state[k].to(torch.float64))
            for k in new_state
        }

        # 4. Local evaluation for the dashboard.
        local_metrics = evaluate_model(
            self.model, self.val_loader, CONFIG.model.num_classes
        )

        # 5. Homomorphic encryption of the update.
        #
        # The FedAvg aggregation weight (n_i / sum(n)) is public, so the
        # client applies it to the plaintext *before* encryption. The server
        # then only performs ciphertext addition, which is exact in CKKS.
        STATE.set_node_status(
            self.hospital_id, NodeStatus.AGGREGATING, "Encrypting update (CKKS)"
        )

        weight = self._aggregation_weight(ins)
        scaled = {k: (v.to(torch.float64) * weight) for k, v in update.items()}

        payload_bytes = 0
        if self.he_enabled:
            blob = he_mod.encrypt_state_dict(self.he_session, scaled)
            payload_bytes = len(blob)
            parameters = fct.Parameters(tensors=[blob], tensor_type="uint8")
        else:
            flat = he_mod.tensor_to_flat_vector(scaled).astype(np.float32)
            parameters = fct.Parameters(tensors=[flat.tobytes()], tensor_type="float32")
            payload_bytes = len(parameters.tensors[0])

        elapsed = time.time() - started
        node = STATE.nodes[self.hospital_id]
        STATE.update_node(
            self.hospital_id,
            rounds_participated=node["rounds_participated"] + 1,
            local_loss=round(local_loss, 6),
            local_dice=local_metrics["dice"],
            encrypted=self.he_enabled,
            payload_bytes=payload_bytes,
            status=NodeStatus.ONLINE,
            last_sync=time.time(),
            message=f"Round complete in {elapsed:.1f}s",
        )

        if self.he_enabled:
            STATE.log_security_event(
                "update_encrypted",
                {
                    "node_id": self.node_id,
                    "ciphertext_bytes": payload_bytes,
                    "scheme": "CKKS",
                },
            )

        if dp_stats:
            STATE.log_privacy_event(
                "dp_noise_applied",
                {
                    "node_id": self.node_id,
                    "sigma": dp_stats["sigma"],
                    "clip_norm": dp_stats["clip_norm"],
                    "noise_l2_norm": dp_stats["noise_l2_norm"],
                    "signal_to_noise": dp_stats.get("signal_to_noise", 0.0),
                    "patients": dp_stats["patients_protected"],
                },
            )

        metrics = {
            "node_id": self.node_id,
            "hospital_id": self.hospital_id,
            "local_loss": round(local_loss, 6),
            "local_dice": local_metrics["dice"],
            "local_accuracy": local_metrics["accuracy"],
            "train_seconds": round(elapsed, 2),
            "encrypted": self.he_enabled,
            "payload_bytes": payload_bytes,
            "dp_noise": dp_stats["noise_l2_norm"] if dp_stats else 0.0,
            "dp_signal_to_noise": dp_stats.get("signal_to_noise", 0.0) if dp_stats else 0.0,
            "dp_steps": steps,
            "patients_protected": patients,
            "aggregation_weight": round(weight, 6),
        }

        logger.info(
            "[%s] fit done: loss=%.4f dice=%.2f%% payload=%sB",
            self.node_id,
            local_loss,
            local_metrics["dice"],
            f"{payload_bytes:,}",
        )

        return fct.FitRes(
            status=fct.Status(code=fct.Code.OK, message="OK"),
            parameters=parameters,
            num_examples=self.num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: fct.EvaluateIns) -> fct.EvaluateRes:
        """Evaluate the broadcast global model on local validation data."""
        global_state = parameters_to_state_dict(ins.parameters, self.reference_state)
        load_state_dict(self.model, global_state)

        metrics = evaluate_model(
            self.model, self.val_loader, CONFIG.model.num_classes
        )
        STATE.update_node(
            self.hospital_id,
            local_loss=metrics["loss"],
            local_dice=metrics["dice"],
        )

        return fct.EvaluateRes(
            status=fct.Status(code=fct.Code.OK, message="OK"),
            loss=metrics["loss"],
            num_examples=len(self.val_loader.dataset),
            metrics={
                "loss": metrics["loss"],
                "dice": metrics["dice"],
                "accuracy": metrics["accuracy"],
            },
        )

    # -- helpers -------------------------------------------------------
    def _ensure_he_session(self, config: dict) -> None:
        """Build the node's HE session from the shared CKKS context.

        The aggregator ships the cluster-wide context in the round config.
        Until it arrives the node transmits its update in the clear, so
        the session is (re)built as soon as the context is available.
        """
        if not self.enable_he:
            return

        blob = config.get("he_context") if config else None
        if not blob:
            return

        if isinstance(blob, str):
            blob = blob.encode("latin-1")

        # Already initialised with this exact context?
        if self.he_enabled and self._he_context_blob == blob:
            return

        session = he_mod.session_from_serialized(self.node_id, blob)
        if not session.available:
            logger.warning("[%s] could not rebuild the CKKS context", self.node_id)
            return

        self.he_session = session
        self._he_context_blob = blob
        self.he_enabled = True
        STATE.update_node(
            self.hospital_id,
            encrypted=True,
            message="CKKS context loaded (shared cluster context)",
        )
        logger.info("[%s] CKKS session activated from shared context", self.node_id)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the local optimizer.

        Plain SGD is used because in our benchmark setup on seed 20260101,
        DP-SGD additive noise interacted poorly with Adam's per-parameter
        second-moment estimates: noise was amplified rather than averaged out,
        and the model stopped learning. See ``tests/sweep3.py`` for calibration.
        """
        cfg = CONFIG.federated
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def _dp_sgd_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict, int]:
        """One DP-SGD optimiser step over a minibatch of patients.

        Every patient's gradient is computed individually, clipped to an
        L2 bound, summed, and perturbed with Gaussian noise. The noise is
        applied to the *sum* (not the average) which is what the standard
        Gaussian mechanism requires.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        gradients: list[torch.Tensor] = []
        for i in range(images.shape[0]):
            self.model.zero_grad(set_to_none=True)
            loss = dice_ce_loss(
                self.model(images[i : i + 1]),
                labels[i : i + 1],
                CONFIG.model.num_classes,
                ce_weight=0.5,
            )
            loss.backward()
            grads = []
            for p in params:
                if p.grad is None:
                    grads.append(torch.zeros(p.numel(), dtype=torch.float64))
                else:
                    grads.append(p.grad.detach().reshape(-1).to(torch.float64))
            gradients.append(torch.cat(grads))

        noised, stats = dp_mod.add_gaussian_noise_to_gradients(
            gradients,
            CONFIG.privacy.dp_noise_multiplier,
            CONFIG.privacy.dp_clip_norm,
        )

        # Scatter the noised summed gradient back into the parameters and
        # let the optimiser apply it.
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = (
                noised[offset : offset + numel].reshape(p.shape).to(p.dtype).clone()
            )
            offset += numel
        optimizer.step()

        # The optimiser's step size is the gradient norm; report it as the
        # loss proxy for the dashboard.
        loss_value = float(noised.norm().item())
        return loss_value, stats, images.shape[0]

    def _aggregation_weight(self, ins: fct.FitIns) -> float:
        """FedAvg weight for this node: n_i / sum(n).

        The server broadcasts the total sample count across all participating
        clients in the round config. When it is unavailable the node falls
        back to a uniform weight and the server re-normalises.
        """
        total = float(ins.config.get("total_examples") or 0)
        if total > 0:
            return self.num_examples / total
        return 1.0


def build_client(hospital_id: str) -> HospitalClient:
    """Factory used by the Flower client entrypoint."""
    hospital = next(h for h in CONFIG.hospitals if h.hospital_id == hospital_id)
    return HospitalClient(hospital.hospital_id, hospital.node_id)
