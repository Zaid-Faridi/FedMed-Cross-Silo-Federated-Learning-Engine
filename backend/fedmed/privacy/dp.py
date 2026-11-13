"""Differential privacy for federated model updates.

Implements **DP-SGD** (Abadi et al., 2016) on the hospital nodes:

1. **Per-sample gradient clipping.** Each patient's gradient is clipped
   individually to an L2 norm bound ``C``. This bounds any single
   record's influence, which is the property that makes the guarantee
   meaningful.
2. **Gaussian noise.** A single noise vector with std ``sigma * C`` is
   added to the *summed* clipped gradients.
3. **Accounting.** The sampled-Gaussian RDP accountant
   (:mod:`fedmed.privacy.accountant`) tracks the cumulative ``(epsilon,
   delta)`` budget consumed by the run.

Why per-sample rather than clipping the whole update
-----------------------------------------------------
Clipping the *aggregated* update to a fixed norm and adding noise of the
same relative scale destroys the signal: the update direction is randomised
almost completely and the global model never converges. Clipping each
*patient's* gradient and summing keeps the expected update unbiased while
still bounding every individual contribution, which is exactly the
guarantee the project needs. The aggregated clip bound therefore grows
with the batch size (``C * sqrt(B)``) rather than being constant.

The privacy guarantee is therefore stated per *patient record*, which is
the unit HIPAA/GDPR actually cares about.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

from ..config import CONFIG
from .accountant import SampledGaussianAccountant

logger = logging.getLogger(__name__)


@dataclass
class DPAccountant:
    """Tracks the cumulative privacy budget consumed by the federated run.

    This is a thin convenience wrapper over the sampled-Gaussian RDP
    accountant that also tracks the *event-level* statistics that make the
    dashboard readable (how many patients were protected, how much noise
    was injected, and so on).
    """

    noise_multiplier: float = field(
        default=CONFIG.privacy.dp_noise_multiplier
    )
    clip_norm: float = field(default=CONFIG.privacy.dp_clip_norm)
    delta: float = field(default=CONFIG.privacy.dp_delta)
    rounds: int = 0
    participation_ratio: float = 1.0
    sample_rate: float = field(default=CONFIG.privacy.dp_sample_rate)
    patients_protected: int = 0
    per_node: dict[str, SampledGaussianAccountant] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._accountant = SampledGaussianAccountant(
            noise_multiplier=self.noise_multiplier,
            delta=self.delta,
            sample_rate=self.sample_rate,
        )
        if not self.per_node:
            for hospital in CONFIG.hospitals:
                self.per_node[hospital.hospital_id] = SampledGaussianAccountant(
                    noise_multiplier=self.noise_multiplier,
                    delta=self.delta,
                    sample_rate=self.sample_rate,
                )

    # -- accounting ---------------------------------------------------
    def spend(self, rounds: int = 1) -> None:
        """Record that ``rounds`` additional rounds of noise were added."""
        self.rounds += rounds
        self._accountant.spend(rounds, sample_rate=self.sample_rate)

    def spend_steps(self, steps: int, sample_rate: float | None = None) -> None:
        """Charge optimiser steps (the unit the RDP bound is defined over)."""
        self._accountant.spend(steps, sample_rate=sample_rate)

    def spend_node_steps(
        self, hospital_id: str, steps: int, sample_rate: float | None = None
    ) -> None:
        """Charge optimiser steps for a specific hospital node."""
        if hospital_id not in self.per_node:
            self.per_node[hospital_id] = SampledGaussianAccountant(
                noise_multiplier=self.noise_multiplier,
                delta=self.delta,
                sample_rate=sample_rate if sample_rate is not None else self.sample_rate,
            )
        self.per_node[hospital_id].spend(
            steps,
            sample_rate=sample_rate if sample_rate is not None else self.sample_rate,
        )

    def get_per_node_dict(self) -> dict[str, dict]:
        """Return serialized per-node DP accounting summaries."""
        return {
            h_id: acct.to_dict()
            for h_id, acct in self.per_node.items()
        }

    @property
    def epsilon(self) -> float:
        """Best epsilon across the evaluated RDP orders."""
        return self._accountant.epsilon

    @property
    def delta_value(self) -> float:
        """The configured delta parameter."""
        return self.delta

    def to_dict(self) -> dict:
        data = self._accountant.to_dict()
        data.update(
            {
                "rounds": self.rounds,
                "clip_norm": self.clip_norm,
                "participation_ratio": self.participation_ratio,
                "patients_protected": self.patients_protected,
                "per_node": self.get_per_node_dict(),
            }
        )
        return data


# ---------------------------------------------------------------------------
# Per-sample gradient clipping and noise
# ---------------------------------------------------------------------------


def clip_gradients_per_sample(
    gradients: list[torch.Tensor],
    clip_norm: float | None = None,
) -> tuple[list[torch.Tensor], list[float]]:
    """Clip each sample's gradient to an L2 norm bound.

    Args:
        gradients: One flat gradient tensor per patient in the batch.
        clip_norm: The per-sample L2 bound ``C``.

    Returns:
        ``(clipped_gradients, original_norms)``.
    """
    limit = clip_norm if clip_norm is not None else CONFIG.privacy.dp_clip_norm

    clipped: list[torch.Tensor] = []
    norms: list[float] = []
    for grad in gradients:
        norm = float(grad.detach().to(torch.float64).norm().item())
        norms.append(norm)
        if norm <= limit or norm == 0.0:
            clipped.append(grad.detach().clone())
            continue
        scale = limit / (norm + 1e-12)
        clipped.append((grad.detach().to(torch.float64) * scale).to(grad.dtype).clone())
    return clipped, norms


def add_gaussian_noise_to_gradients(
    gradients: list[torch.Tensor],
    noise_multiplier: float | None = None,
    clip_norm: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Apply the sampled Gaussian mechanism to a batch of per-sample gradients.

    The mechanism is:

        1. clip every per-sample gradient to L2 norm ``C``;
        2. sum the clipped gradients;
        3. add ``N(0, (sigma * C)^2 I)`` noise to the sum.

    The expected value of the result is the sum of the *unclipped*
    gradients, so the mechanism is unbiased while still bounding every
    single patient's contribution.

    Args:
        gradients: One flat gradient tensor per patient.
        noise_multiplier: ``sigma``; the injected std is ``sigma * C``.
        clip_norm: Per-sample L2 clip bound ``C``.
        generator: Optional RNG for reproducible noise.

    Returns:
        ``(noised_summed_gradient, stats)``.
    """
    mult = (
        noise_multiplier
        if noise_multiplier is not None
        else CONFIG.privacy.dp_noise_multiplier
    )
    limit = clip_norm if clip_norm is not None else CONFIG.privacy.dp_clip_norm

    clipped, norms = clip_gradients_per_sample(gradients, limit)

    if not clipped:
        return torch.zeros(0), {
            "original_norm": 0.0,
            "mean_original_norm": 0.0,
            "clipped": False,
            "sigma": round(mult * limit, 6),
            "noise_l2_norm": 0.0,
            "noise_multiplier": mult,
            "clip_norm": limit,
            "batch_size": 0,
            "patients_protected": 0,
            "signal_l2_norm": 0.0,
        }

    # Sum the clipped gradients.
    total = clipped[0].to(torch.float64).clone()
    for grad in clipped[1:]:
        total += grad.to(torch.float64)

    signal_l2 = float(total.pow(2).sum().sqrt().item())

    # Inject the Gaussian noise.
    sigma = mult * limit
    noise = torch.normal(
        mean=0.0,
        std=sigma,
        size=total.shape,
        dtype=torch.float64,
        generator=generator,
    )
    noise_l2 = float(noise.pow(2).sum().sqrt().item())
    total = total + noise

    stats = {
        "original_norm": round(max(norms), 6),
        "mean_original_norm": round(sum(norms) / len(norms), 6),
        "clipped": any(n > limit for n in norms),
        "sigma": round(sigma, 6),
        "noise_l2_norm": round(noise_l2, 6),
        "noise_multiplier": mult,
        "clip_norm": limit,
        "batch_size": len(gradients),
        "patients_protected": len(gradients),
        "signal_l2_norm": round(signal_l2, 6),
        "signal_to_noise": round(signal_l2 / noise_l2, 4) if noise_l2 > 0 else 0.0,
    }

    # Standard DP-SGD averages the noised sum: the gradient is
    #   (1/B) * ( sum_i clip(g_i) + N(0, (sigma*C)^2 I) )
    # so the noise per unit of signal stays at sigma*C/B.
    averaged = total / len(gradients)
    stats["averaged_noise_l2_norm"] = round(noise_l2 / len(gradients), 6)
    stats["averaged_signal_l2_norm"] = round(signal_l2 / len(gradients), 6)
    return averaged.to(clipped[0].dtype), stats


def dp_status() -> dict:
    """Report the DP subsystem configuration to the dashboard."""
    cfg = CONFIG.privacy
    return {
        "enabled": cfg.enable_dp,
        "mechanism": "Sampled Gaussian (DP-SGD, per-sample clipping)",
        "noise_multiplier": cfg.dp_noise_multiplier,
        "clip_norm": cfg.dp_clip_norm,
        "delta": cfg.dp_delta,
        "sample_rate": cfg.dp_sample_rate,
        "projected_epsilon": cfg.dp_epsilon_bound,
        "accountant": "RDP (Mironov, Talwar & Zhang 2019)",
    }
