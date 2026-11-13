"""Centralized baseline trainer (Week 1 of the project plan).

Trains the same 3D U-Net on the *pooled* dataset (all three hospitals'
data combined in one process) to establish the accuracy ceiling that the
federated run is audited against.

The comparison is only meaningful if both runs see identical data and
identical hyper-parameters, so this module reuses the exact loss,
optimizer and schedule used by the federated clients.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import CONFIG, MODEL_ROOT
from .data import load_global_test, load_global_validation, load_pooled_training_set
from .metrics import dice_ce_loss, evaluate_model
from .model import build_model, count_parameters
from .seed import set_seed

logger = logging.getLogger(__name__)


def train_baseline(
    epochs: int = 30,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
    ce_weight: float = 0.5,
    dice_weight: float = 1.0,
    output_dir: Path | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> dict:
    """Train the centralized baseline model.

    Returns:
        A metrics dict with the per-epoch history and the final scores.
    """
    effective_seed = set_seed(seed)
    out_dir = Path(output_dir) if output_dir else MODEL_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # The baseline must use the *same* hyper-parameters as the federated
    # nodes, otherwise the accuracy comparison is meaningless.
    if learning_rate is None:
        learning_rate = CONFIG.federated.learning_rate
    if batch_size is None:
        batch_size = CONFIG.federated.batch_size
    if weight_decay is None:
        weight_decay = CONFIG.federated.weight_decay

    train_x, train_y = load_pooled_training_set()
    val_x, val_y = load_global_validation()

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )

    logger.info(
        "Baseline: pooled train=%d val=%d", len(train_x), len(val_x)
    )

    model = build_model().to(device)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    if CONFIG.federated.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=CONFIG.federated.momentum,
            weight_decay=weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )

    history: list[dict] = []
    best_dice = -1.0
    best_state = None
    best_epoch = 0
    started = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = dice_ce_loss(
                logits,
                labels,
                CONFIG.model.num_classes,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()

        scheduler.step()

        metrics = evaluate_model(model, val_loader, CONFIG.model.num_classes, device)
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(running / len(train_loader), 6),
            "val_loss": metrics["loss"],
            "dice": metrics["dice"],
            "accuracy": metrics["accuracy"],
            "per_class_dice": metrics["per_class_dice"],
        }
        history.append(entry)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        logger.info(
            "epoch %2d/%d loss=%.4f dice=%.2f%% acc=%.2f%%",
            epoch + 1,
            epochs,
            entry["train_loss"],
            entry["dice"],
            entry["accuracy"],
        )

    elapsed = time.time() - started

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate the best validation model on the held-out test split
    try:
        test_x, test_y = load_global_test()
        test_loader = DataLoader(
            TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False
        )
        test_metrics = evaluate_model(
            model, test_loader, CONFIG.model.num_classes, device
        )
        test_dice = test_metrics["dice"]
        test_loss = test_metrics["loss"]
        test_accuracy = test_metrics["accuracy"]
    except FileNotFoundError:
        test_dice = 0.0
        test_loss = 0.0
        test_accuracy = 0.0

    torch.save(
        {
            "state_dict": best_state,
            "config": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": CONFIG.federated.optimizer,
                "num_classes": CONFIG.model.num_classes,
            },
            "metrics": {
                "best_dice": round(best_dice, 4),
                "best_epoch": best_epoch,
                "test_dice": round(test_dice, 4),
                "test_loss": round(test_loss, 4),
                "test_accuracy": round(test_accuracy, 4),
                "final_dice": history[-1]["dice"] if history else 0.0,
                "final_accuracy": history[-1]["accuracy"] if history else 0.0,
                "final_loss": history[-1]["val_loss"] if history else 0.0,
                "per_class_dice": history[-1]["per_class_dice"] if history else {},
            },
            "history": history,
            "duration_seconds": round(elapsed, 2),
        },
        out_dir / "baseline_model.pt",
    )

    summary = {
        "best_dice": round(best_dice, 4),
        "best_epoch": best_epoch,
        "test_dice": round(test_dice, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "final_dice": history[-1]["dice"] if history else 0.0,
        "final_accuracy": history[-1]["accuracy"] if history else 0.0,
        "final_loss": history[-1]["val_loss"] if history else 0.0,
        "per_class_dice": history[-1]["per_class_dice"] if history else {},
        "epochs": epochs,
        "duration_seconds": round(elapsed, 2),
        "parameters": count_parameters(model),
        "history": history,
        "model_path": str(out_dir / "baseline_model.pt"),
    }

    with open(out_dir / "baseline_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "Baseline complete: best_val_dice=%.2f%% (epoch %d), test_dice=%.2f%% in %.1fs",
        best_dice,
        best_epoch,
        test_dice,
        elapsed,
    )
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FedMed centralized baseline trainer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to FEDMED_SEED or 20260101)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    train_baseline(epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
