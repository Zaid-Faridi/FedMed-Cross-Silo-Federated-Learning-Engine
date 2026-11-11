"""Tests for Phase 1 verification requirements.

Validates:
a) A node with HE enabled and no CKKS context raises RuntimeError and sends nothing in plaintext.
b) A round containing one unencrypted update is rejected, the global model is unchanged,
   and a 'round_rejected' security event is logged.
c) Missing TLS certs raise FileNotFoundError unless --insecure / use_tls=False is passed.
d) dp_accountant.per_node is present in the state returned by the API.
e) Poisson batches: over many draws the mean batch size is about batch_size and empty
   batches occur and are handled.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
from flwr.compat.common import typing as fct

from fedmed.client import HospitalClient
from fedmed.config import CONFIG
from fedmed.server import _server_certificates
from fedmed.state import STATE, RuntimeState
from fedmed.strategy import FederatedStrategy


def test_he_node_without_context_raises_fail_closed():
    """Requirement 2a: A node with HE enabled and no CKKS context raises RuntimeError

    and sends nothing in plaintext.
    """
    client = HospitalClient("hospital-1", "HOSP-001", enable_he=True, local_epochs=1)
    # Ensure context is not loaded
    client.he_enabled = False
    client.he_session = MagicMock()
    client.he_session.available = False

    # Create valid parameters matching client's model
    valid_params = client.get_parameters(fct.GetParametersIns({})).parameters
    ins = fct.FitIns(
        parameters=valid_params,
        config={},  # No he_context provided
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.fit(ins)

    assert "Homomorphic encryption is enabled but no valid CKKS context is available" in str(
        exc_info.value
    )
    assert "refusing to send plaintext update" in str(exc_info.value)


def test_round_with_unencrypted_update_rejected():
    """Requirement 2b: A round containing one unencrypted update is rejected,

    the global model is unchanged, and a 'round_rejected' security event is logged.
    """
    strategy = FederatedStrategy(num_rounds=1, enable_he=True)
    initial_params = strategy.initialize_parameters(MagicMock())
    initial_bytes = initial_params.tensors[0]

    # Clear previous events
    STATE.security_events.clear()

    # Create results: 2 encrypted, 1 unencrypted
    results = [
        (
            MagicMock(cid="1"),
            fct.FitRes(
                status=fct.Status(code=fct.Code.OK, message="OK"),
                parameters=fct.Parameters(tensors=[b"fake_cipher_1"], tensor_type="uint8"),
                num_examples=10,
                metrics={"node_id": "HOSP-001", "encrypted": True, "aggregation_weight": 0.33},
            ),
        ),
        (
            MagicMock(cid="2"),
            fct.FitRes(
                status=fct.Status(code=fct.Code.OK, message="OK"),
                parameters=fct.Parameters(tensors=[b"fake_cipher_2"], tensor_type="uint8"),
                num_examples=10,
                metrics={"node_id": "HOSP-002", "encrypted": True, "aggregation_weight": 0.33},
            ),
        ),
        (
            MagicMock(cid="3"),
            fct.FitRes(
                status=fct.Status(code=fct.Code.OK, message="OK"),
                parameters=fct.Parameters(tensors=[b"fake_plaintext_3"], tensor_type="float32"),
                num_examples=10,
                metrics={"node_id": "HOSP-003", "encrypted": False, "aggregation_weight": 0.34},
            ),
        ),
    ]

    out_params, out_metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    # Global model must be unchanged
    assert out_params.tensors[0] == initial_bytes
    assert out_metrics == {}

    # 'round_rejected' security event must be logged
    _, sec_events = STATE.events_snapshot()
    rejected_events = [e for e in sec_events if e.get("event") == "round_rejected"]
    assert len(rejected_events) >= 1
    assert rejected_events[-1]["detail"]["reason"] == "unencrypted_update_received"
    assert rejected_events[-1]["detail"]["round"] == 1


def test_missing_tls_certs_raise_file_not_found(tmp_path):
    """Requirement 2c: Missing TLS certs raise FileNotFoundError unless --insecure is passed."""
    non_existent = tmp_path / "does_not_exist.pem"
    orig_ca = CONFIG.server.ca_cert
    object.__setattr__(CONFIG.server, "ca_cert", non_existent)
    try:
        # When use_tls is True, must raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            _server_certificates(use_tls=True)
        assert "TLS material incomplete" in str(exc_info.value)

        # When use_tls is False (--insecure), returns None without raising
        assert _server_certificates(use_tls=False) is None
    finally:
        object.__setattr__(CONFIG.server, "ca_cert", orig_ca)


def test_dp_accountant_per_node_in_api_state():
    """Requirement 2d: dp_accountant.per_node is present in the state returned by the API."""
    from fedmed.api import get_status

    # Set some per-node state
    STATE.dp_epsilon = 1.25
    STATE.dp_delta = 1e-5
    STATE.dp_rounds = 3
    STATE.dp_per_node = {
        "hospital-1": {"epsilon": 1.2, "steps": 10},
        "hospital-2": {"epsilon": 1.1, "steps": 8},
        "hospital-3": {"epsilon": 0.9, "steps": 6},
    }

    status = get_status()
    assert "dp_accountant" in status
    dp_acct = status["dp_accountant"]
    assert "per_node" in dp_acct
    assert "hospital-1" in dp_acct["per_node"]
    assert dp_acct["per_node"]["hospital-1"]["epsilon"] == 1.2
    assert dp_acct["per_node"]["hospital-2"]["steps"] == 8


def test_poisson_batches_properties_and_empty_batch_handling():
    """Requirement 2e: Poisson batches: over many draws the mean batch size is about

    batch_size and empty batches occur and are handled.
    """
    client = HospitalClient("hospital-1", "HOSP-001", batch_size=4)
    n = client.num_examples
    target_bs = 4
    q = min(target_bs / n, 1.0)

    # Draw many batches using Poisson sampling
    num_draws = 2000
    batch_sizes = []
    empty_batches_count = 0

    for _ in range(num_draws):
        batch = client._sample_poisson_batch()
        if batch is None:
            empty_batches_count += 1
            batch_sizes.append(0)
        else:
            images, labels = batch
            batch_sizes.append(len(images))

    mean_batch_size = float(np.mean(batch_sizes))
    # Over 2000 draws, mean batch size should be close to target_bs (within 0.25)
    assert abs(mean_batch_size - target_bs) < 0.25, f"Expected ~{target_bs}, got {mean_batch_size}"

    # Verify that empty batches occur with non-zero probability (1-q)^n > 0
    # For n=48, target_bs=4, q=4/48=1/12, P(empty) = (11/12)^48 ~ 0.0146 (~1.5%)
    assert empty_batches_count > 0, f"Expected some empty batches, got {empty_batches_count}"

    # Verify empty batch handling in _dp_sgd_step:
    optimizer = MagicMock()
    loss_val, stats, n_patients = client._dp_sgd_step(
        torch.zeros(0, 1, 16, 16, 16),
        torch.zeros(0, 16, 16, 16),
        optimizer,
    )
    assert loss_val == 0.0
    assert stats == {}
    assert n_patients == 0
    # Optimizer step should NOT be called for an empty batch
    assert optimizer.step.call_count == 0
