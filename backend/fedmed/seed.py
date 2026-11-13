"""Deterministic seeding utilities for reproducibility."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int | None = None) -> int:
    """Seed random, numpy and torch.

    If seed is None, looks for FEDMED_SEED environment variable,
    defaulting to 20260101 if not set.

    Returns:
        The integer seed used.
    """
    if seed is None:
        env_seed = os.getenv("FEDMED_SEED")
        if env_seed is not None and env_seed.strip():
            seed = int(env_seed.strip())
        else:
            seed = 20260101

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
