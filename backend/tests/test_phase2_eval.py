"""Phase 2 evaluation tests:
- Train/val/test patient IDs are disjoint across all hospital partitions.
- Same seed produces identical metrics twice (reproducibility).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from fedmed.baseline import train_baseline
from fedmed.config import CONFIG, DATA_ROOT
from fedmed.data import build_partition


def test_train_val_test_patient_ids_disjoint(tmp_path: Path):
    """Verify that patient sample IDs in train, val, and test are strictly disjoint."""
    # Ensure partition is generated in tmp_path
    partitions = build_partition(data_root=tmp_path)

    train_ids = set()
    val_ids = set()
    test_ids = set()

    total_train = 0
    total_val = 0
    total_test = 0

    for hospital in CONFIG.hospitals:
        train_blob = torch.load(hospital.data_dir / "train.pt", map_location="cpu", weights_only=False)
        val_blob = torch.load(hospital.data_dir / "val.pt", map_location="cpu", weights_only=False)
        test_blob = torch.load(hospital.data_dir / "test.pt", map_location="cpu", weights_only=False)

        h_train_ids = {m["sample_id"] for m in train_blob.get("metadata", [])}
        h_val_ids = {m["sample_id"] for m in val_blob.get("metadata", [])}
        h_test_ids = {m["sample_id"] for m in test_blob.get("metadata", [])}

        total_train += len(train_blob["images"])
        total_val += len(val_blob["images"])
        total_test += len(test_blob["images"])

        # Check disjointness within hospital
        assert len(h_train_ids & h_val_ids) == 0, f"{hospital.hospital_id} has overlap between train and val"
        assert len(h_train_ids & h_test_ids) == 0, f"{hospital.hospital_id} has overlap between train and test"
        assert len(h_val_ids & h_test_ids) == 0, f"{hospital.hospital_id} has overlap between val and test"

        train_ids.update(h_train_ids)
        val_ids.update(h_val_ids)
        test_ids.update(h_test_ids)

    # Check disjointness globally across all nodes
    assert len(train_ids & val_ids) == 0, "Global overlap between train and val IDs"
    assert len(train_ids & test_ids) == 0, "Global overlap between train and test IDs"
    assert len(val_ids & test_ids) == 0, "Global overlap between val and test IDs"

    # Check totals match configuration
    assert total_train == CONFIG.data.num_samples == 120
    assert total_val == CONFIG.data.num_val_samples == 36
    assert total_test == CONFIG.data.num_test_samples == 24
    assert len(train_ids) == 120
    assert len(val_ids) == 36
    assert len(test_ids) == 24


def test_same_seed_gives_identical_baseline_metrics(tmp_path: Path):
    """Verify that training baseline twice with the same seed yields identical loss and Dice metrics."""
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    seed = 424242
    epochs = 2

    res1 = train_baseline(epochs=epochs, seed=seed, output_dir=out_dir_1)
    res2 = train_baseline(epochs=epochs, seed=seed, output_dir=out_dir_2)

    assert res1["best_dice"] == res2["best_dice"], f"best_dice mismatch: {res1['best_dice']} vs {res2['best_dice']}"
    assert res1["final_dice"] == res2["final_dice"], f"final_dice mismatch: {res1['final_dice']} vs {res2['final_dice']}"
    assert res1["test_dice"] == res2["test_dice"], f"test_dice mismatch: {res1['test_dice']} vs {res2['test_dice']}"
    assert res1["final_loss"] == res2["final_loss"], f"final_loss mismatch: {res1['final_loss']} vs {res2['final_loss']}"

    assert len(res1["history"]) == len(res2["history"]) == epochs
    for h1, h2 in zip(res1["history"], res2["history"]):
        assert h1["train_loss"] == h2["train_loss"]
        assert h1["val_loss"] == h2["val_loss"]
        assert h1["dice"] == h2["dice"]

