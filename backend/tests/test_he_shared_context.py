"""Regression test for the shared-context homomorphic aggregation.

Bug
---
Every hospital node used to mint its **own** independent CKKS context.
In TenSEAL CKKS, ciphertexts encrypted under different keys cannot be
added: the homomorphic sum silently produced values around 1e34, which
poisoned the global model and made the federated run diverge to NaN.

The requirement is that all nodes encrypt under the exact same key pair
created by the aggregator and distributed over the TLS-protected round
configuration.

This test locks the behaviour in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedmed.privacy import he as he_mod

pytestmark = pytest.mark.skipif(
    not he_mod.TENSEAL_AVAILABLE, reason="TenSEAL is not installed"
)


def _model_vector(size: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1e-3, size=size)


def test_independent_contexts_cannot_be_added():
    """Documents the original bug: mixing keys yields garbage."""
    server = he_mod.create_session("server")
    node_a = he_mod.create_session("node-a")
    node_b = he_mod.create_session("node-b")

    v_a = _model_vector(64, seed=1)
    v_b = _model_vector(64, seed=2)

    blob_a = node_a.encrypt_vector(v_a)
    blob_b = node_b.encrypt_vector(v_b)

    # Mixing keys does not raise: TenSEAL happily produces a number.
    # That number is meaningless, which is exactly the bug this guards.
    acc = he_mod.ts.CKKSVector.load(server.context, blob_a)
    acc = acc + he_mod.ts.CKKSVector.load(server.context, blob_b)
    out = np.array(acc.decrypt())

    assert not np.allclose(out, v_a + v_b, atol=1e-3), (
        "expected the mixed-key sum to be meaningless"
    )


def test_shared_context_allows_exact_homomorphic_sum():
    """Nodes sharing the aggregator's context can be summed exactly."""
    server = he_mod.create_session("server")
    blob = he_mod.serialize_context(server)

    node_a = he_mod.session_from_serialized("node-a", blob)
    node_b = he_mod.session_from_serialized("node-b", blob)

    v_a = _model_vector(256, seed=1)
    v_b = _model_vector(256, seed=2)

    blob_a = node_a.encrypt_vector(v_a)
    blob_b = node_b.encrypt_vector(v_b)

    acc = he_mod.ts.CKKSVector.load(server.context, blob_a)
    acc = acc + he_mod.ts.CKKSVector.load(server.context, blob_b)
    out = np.array(acc.decrypt())

    assert np.allclose(out, v_a + v_b, atol=1e-3)


def test_shared_context_sum_of_three_clients():
    """Three hospitals (the FedMed topology) aggregate correctly."""
    server = he_mod.create_session("server")
    blob = he_mod.serialize_context(server)

    vectors = [_model_vector(256, seed=s) for s in (11, 22, 33)]
    blobs = [
        he_mod.session_from_serialized(f"node-{i}", blob).encrypt_vector(v)
        for i, v in enumerate(vectors)
    ]

    total = server.decrypt_sum(blobs)
    expected = np.sum(vectors, axis=0)

    assert np.allclose(total, expected, atol=1e-3)


def test_state_dict_round_trip_through_ciphertext():
    """A full model update survives encryption -> sum -> decryption."""
    import torch

    from fedmed.model import build_model

    model = build_model()
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    server = he_mod.create_session("server")
    blob = he_mod.serialize_context(server)
    node = he_mod.session_from_serialized("node-0", blob)

    # Two "clients", each contributing half of an update.
    torch.manual_seed(0)
    delta_a = {
        k: torch.randn_like(v.float()) * 1e-3 for k, v in reference.items()
    }
    torch.manual_seed(1)
    delta_b = {
        k: torch.randn_like(v.float()) * 1e-3 for k, v in reference.items()
    }

    enc_a = he_mod.encrypt_state_dict(node, delta_a)
    enc_b = he_mod.encrypt_state_dict(node, delta_b)

    total_vec = server.decrypt_sum([enc_a, enc_b])
    total = he_mod.flat_vector_to_tensor(total_vec, reference)

    for key in reference:
        expected = (delta_a[key].double() + delta_b[key].double()).to(
            reference[key].dtype
        )
        assert torch.allclose(total[key], expected, atol=1e-4), key


def test_aggregated_update_magnitude_is_finite_and_sane():
    """Guards the NaN regression: the aggregate must be finite and small.

    A poisoned aggregate shows up as ~1e34 values or NaN. This asserts
    the decrypted aggregate of realistic model deltas stays in a sane
    range.
    """
    import torch

    from fedmed.model import build_model

    model = build_model()
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    server = he_mod.create_session("server")
    blob = he_mod.serialize_context(server)

    nodes = [he_mod.session_from_serialized(f"node-{i}", blob) for i in range(3)]
    deltas = []
    for i in range(3):
        torch.manual_seed(100 + i)
        deltas.append(
            {k: torch.randn_like(v.float()) * 1e-3 for k, v in reference.items()}
        )

    blobs = [
        he_mod.encrypt_state_dict(nodes[i], deltas[i]) for i in range(3)
    ]
    total = server.decrypt_sum(blobs)

    assert np.all(np.isfinite(total)), "aggregated update contains NaN/Inf"
    # A realistic weighted update has an L2 norm well under 100.
    assert float(np.linalg.norm(total)) < 100.0
