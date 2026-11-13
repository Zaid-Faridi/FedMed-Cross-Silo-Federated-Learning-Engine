"""Generate the partitioned synthetic BraTS-style dataset.

Usage:
    python -m fedmed.prepare_data
"""

from __future__ import annotations

import logging

from .config import CONFIG, DATA_ROOT
from .data import build_partition

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("fedmed.prepare_data")


def main() -> None:
    logger.info("Generating synthetic BraTS-style dataset...")
    logger.info(
        "  samples=%s  val=%s  test=%s  patch=%s",
        CONFIG.data.num_samples,
        CONFIG.data.num_val_samples,
        CONFIG.data.num_test_samples,
        CONFIG.data.patch_size,
    )

    partitions = build_partition(DATA_ROOT)

    logger.info("Partitioned across %d hospital nodes:", len(partitions))
    for hospital_id, info in partitions.items():
        logger.info(
            "  %-12s node=%s train=%2d val=%2d test=%2d scanner_bias=%+.2f",
            hospital_id,
            info["node_id"],
            info["train_samples"],
            info["val_samples"],
            info.get("test_samples", 0),
            info["scanner_bias"],
        )

    logger.info("Manifest written to %s", DATA_ROOT / "manifest.json")

    total_train = sum(p["train_samples"] for p in partitions.values())
    total_val = sum(p["val_samples"] for p in partitions.values())
    total_test = sum(p.get("test_samples", 0) for p in partitions.values())
    logger.info("Totals: train=%d val=%d test=%d", total_train, total_val, total_test)


if __name__ == "__main__":
    main()
