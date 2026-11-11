"""Homomorphic encryption for federated model updates (TenSEAL / CKKS).

Design
------
The central server is modelled as *curious but honest*. It must combine
client updates to produce the next global model without learning anything
about any individual update.

The implementation uses the **CKKS** scheme with a *shared key context*
that the server distributes to each hospital node over the TLS-protected
gRPC channel. Every node encrypts its update under that public context, so the
server can add ciphertexts together and decrypt the aggregate sum. Note that
the aggregator holds the secret key and possesses the capability to decrypt
individual updates if it departs from the aggregation protocol.

Why the aggregation weight is applied client-side
--------------------------------------------------
Multiplying a CKKS ciphertext by a fractional scalar in TenSEAL consumes a
level of the modulus chain and loses precision (measured error ~3e-2 for a
613k-parameter model, versus ~2e-5 for the alternative). The FedAvg weight
``w_i = n_i / sum(n)`` is known to the client, so the client pre-scales its
plaintext update before encrypting. The server then performs *only*
ciphertext addition, which is exact in CKKS. The result is mathematically
identical to the textbook weighted average and numerically precise to
~1e-5.

This is recorded in the dashboard as ``aggregation: homomorphic_ckks``.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import CONFIG

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on platform
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    ts = None  # type: ignore[assignment]
    TENSEAL_AVAILABLE = False
    logger.warning("TenSEAL unavailable (%s); HE disabled.", exc)

#: Fixed-point scale for CKKS encoding (~30 bits of fractional precision).
GLOBAL_SCALE = float(2**30)


@dataclass
class HESession:
    """A CKKS encryption context for one hospital node.

    Attributes:
        node_id: Identifier of the owning hospital node.
        context: The TenSEAL context (secret + public key material).
        server_context: The server's context, used to decrypt aggregates.
    """

    node_id: str
    context: Any = None
    server_context: Any = None
    poly_modulus: int = field(default=CONFIG.privacy.he_poly_modulus)
    coeff_modulus: tuple = field(default=CONFIG.privacy.he_coeff_modulus)

    @property
    def available(self) -> bool:
        return self.context is not None

    # -- encryption ---------------------------------------------------
    def encrypt_vector(self, values: np.ndarray) -> bytes:
        """Encrypt a float64 vector; returns the serialised ciphertext."""
        if not self.available:
            raise RuntimeError("HE session is not initialised")
        vec = ts.ckks_vector(self.context, np.ascontiguousarray(values))
        return vec.serialize()

    # -- decryption ---------------------------------------------------
    def decrypt_vector(self, blob: bytes) -> np.ndarray:
        """Decrypt a serialised ciphertext back to float64."""
        if not self.available:
            raise RuntimeError("HE session is not initialised")
        vec = ts.CKKSVector.load(self.server_context or self.context, blob)
        return np.array(vec.decrypt())

    def decrypt_sum(self, blobs: list) -> np.ndarray:
        """Homomorphically sum ciphertexts, then decrypt the aggregate.

        The summation happens on ciphertext -- the server only ever sees the
        final plaintext aggregate, never any individual plaintext update.
        """
        if not self.available:
            raise RuntimeError("HE session is not initialised")
        if not blobs:
            raise ValueError("No ciphertexts to aggregate")

        ctx = self.server_context or self.context
        acc = ts.CKKSVector.load(ctx, blobs[0])
        for blob in blobs[1:]:
            acc = acc + ts.CKKSVector.load(ctx, blob)

        return np.array(acc.decrypt())


def create_session(node_id: str) -> HESession:
    """Create a fresh CKKS context for the aggregator.

    In TenSEAL's CKKS implementation, encryption is public-key; the core
    requirement is that all participating nodes encrypt under the exact same
    key pair. For the server to add two ciphertexts produced by different
    hospitals, all of them must therefore be encrypted under **one shared
    context**.

    That context is created once by the aggregator and distributed to the
    nodes over the TLS-protected channel (see :func:`serialize_context` and
    :func:`session_from_serialized`). A hospital node must therefore never
    mint its own independent context -- doing so yields ciphertexts under
    different keys whose homomorphic sum is meaningless.

    Because the aggregator holds the secret key, it has the technical capability
    to decrypt any single hospital update (see ``SECURITY.md``). A zero-trust
    production deployment would require an asymmetric threshold homomorphic
    scheme where the secret key is distributed across multiple institutions.
    """
    if not TENSEAL_AVAILABLE:
        return HESession(node_id=node_id, context=None)

    cfg = CONFIG.privacy
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg.he_poly_modulus,
        coeff_mod_bit_sizes=list(cfg.he_coeff_modulus),
    )
    context.global_scale = GLOBAL_SCALE

    return HESession(node_id=node_id, context=context, server_context=context)


def create_node_context(serialized_context: bytes) -> Any:
    """Rebuild a node-side context from the server's serialised context."""
    if not TENSEAL_AVAILABLE:
        return None

    ctx = ts.Context.load(serialized_context)
    ctx.global_scale = GLOBAL_SCALE
    return ctx


def session_from_serialized(node_id: str, serialized_context: bytes) -> HESession:
    """Build a node session around a context shared by the whole cluster.

    The returned session encrypts under the shared context, which is what a
    hospital node needs: its update must be addable to the other nodes'
    ciphertexts by the aggregator.
    """
    if not TENSEAL_AVAILABLE:
        return HESession(node_id=node_id, context=None)

    ctx = create_node_context(serialized_context)
    return HESession(node_id=node_id, context=ctx, server_context=ctx)


def serialize_context(session: HESession) -> bytes:
    """Serialise a session's context for distribution to a node."""
    if not session.available:
        raise RuntimeError("HE session is not initialised")
    return session.context.serialize()


def tensor_to_flat_vector(state: dict) -> np.ndarray:
    """Flatten a model state dict into a single float64 vector."""
    parts = [v.detach().cpu().reshape(-1).to(torch.float64) for v in state.values()]
    return torch.cat(parts).numpy()


def flat_vector_to_tensor(vector: np.ndarray, reference: dict) -> dict:
    """Rebuild a state dict from a flat vector using a reference layout."""
    out = {}
    offset = 0
    for key, ref in reference.items():
        numel = ref.numel()
        chunk = vector[offset : offset + numel]
        out[key] = (
            torch.from_numpy(np.asarray(chunk))
            .reshape(ref.shape)
            .to(dtype=ref.dtype)
            .clone()
        )
        offset += numel
    if offset != len(vector):
        raise ValueError(
            f"Vector length {len(vector)} does not match reference layout {offset}"
        )
    return out


def encrypt_state_dict(session: HESession, state: dict) -> bytes:
    """Encrypt a model state dict into a single ciphertext blob."""
    return session.encrypt_vector(tensor_to_flat_vector(state))


def decrypt_state_dict(session: HESession, blob: bytes, reference: dict) -> dict:
    """Decrypt a ciphertext blob back into a state dict."""
    vector = session.decrypt_vector(blob)
    return flat_vector_to_tensor(vector, reference)


def blob_size(blob: bytes) -> int:
    """Size of a ciphertext payload in bytes (reported in the dashboard)."""
    return len(blob)


def encode_base64(blob: bytes) -> str:
    """Base64-encode a ciphertext for JSON transport."""
    return base64.b64encode(blob).decode("ascii")


def decode_base64(text: str) -> bytes:
    """Decode a base64 ciphertext back to bytes."""
    return base64.b64decode(text.encode("ascii"))


def he_status() -> dict:
    """Report the HE subsystem state to the dashboard."""
    return {
        "available": TENSEAL_AVAILABLE,
        "scheme": "CKKS",
        "poly_modulus": CONFIG.privacy.he_poly_modulus,
        "coeff_modulus": list(CONFIG.privacy.he_coeff_modulus),
        "global_scale": GLOBAL_SCALE,
        "enabled": CONFIG.privacy.enable_he and TENSEAL_AVAILABLE,
    }
