"""Central configuration for the FedMed federated learning engine.

Every tunable for the federated run lives here so the server, the hospital
nodes and the API layer all agree on the same hyper-parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("FEDMED_DATA_ROOT", BACKEND_ROOT / "data"))
MODEL_ROOT = Path(os.getenv("FEDMED_MODEL_ROOT", BACKEND_ROOT / "models"))
CERT_ROOT = Path(os.getenv("FEDMED_CERT_ROOT", BACKEND_ROOT / "certs"))
LOG_ROOT = Path(os.getenv("FEDMED_LOG_ROOT", BACKEND_ROOT / "logs"))

for _d in (DATA_ROOT, MODEL_ROOT, CERT_ROOT, LOG_ROOT):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """3D U-Net segmentation model configuration."""

    in_channels: int = 1
    out_channels: int = 3  # background + edema + enhancing tumour
    spatial_size: tuple[int, int, int] = (64, 64, 64)
    # A 3-level encoder/decoder keeps the parameter count at ~600k. This is a
    # deliberate trade-off: a CKKS ciphertext expands the float32 parameter vector
    # by roughly 20.3x (~10.1x relative to float64), so every extra channel
    # multiplies the per-round upload volume. At this size one encrypted update is
    # ~48.7 MB, which fits comfortably inside Flower's 512 MB gRPC message limit.
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 1
    dropout: float = 0.2

    @property
    def num_classes(self) -> int:
        return self.out_channels


@dataclass(frozen=True)
class DataConfig:
    """Synthetic BraTS-style dataset configuration."""

    num_samples: int = 120
    num_val_samples: int = 36
    patch_size: tuple[int, int, int] = (64, 64, 64)
    seed: int = 20260101
    # Hospital split ratios must sum to 1.0.
    hospital_split: tuple = (0.4, 0.35, 0.25)


@dataclass(frozen=True)
class FederatedConfig:
    """Federated learning hyper-parameters."""

    num_rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 8
    # SGD (not Adam) is the optimiser used by the nodes: in our benchmark
    # setup on seed 20260101, DP-SGD noise interacted poorly with Adam's
    # per-parameter second-moment estimates, amplifying injected noise rather
    # than averaging it out. Calibrated over a (sigma, lr) sweep -- see tests/sweep3.py.
    learning_rate: float = 5e-2
    weight_decay: float = 1e-5
    optimizer: str = "sgd"
    momentum: float = 0.9
    strategy: str = "fedavg"
    min_fit_clients: int = 1
    min_available_clients: int = 1
    accept_failures: bool = True
    # Simulate a hospital dropping offline at this round (0 disables).
    dropout_round: int = 0
    dropout_node: int = 3
    # Seconds to wait for a node to answer a round before giving up on it.
    round_timeout: int = 900


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy-preserving mechanism configuration."""

    # -- Homomorphic encryption (TenSEAL CKKS) ------------------------
    enable_he: bool = True
    # Polynomial modulus degree (power of two). Must be 8192/16384/32768.
    he_poly_modulus: int = 16384
    # Bit sizes of the coefficient moduli chain. The first and last primes
    # carry the message modulus; the middle primes are the key-switch
    # levels consumed by the ciphertext additions performed on the server.
    he_coeff_modulus: tuple = (60, 40, 40, 60)

    # -- Differential privacy (sampled Gaussian mechanism) ------------
    enable_dp: bool = True
    # sigma = noise_multiplier * clip_norm. Calibrated in tests/sweep3.py:
    # sigma=0.5 converges (epsilon ~127), sigma=2.0 destroys the signal.
    dp_noise_multiplier: float = 0.5
    # L2 clipping bound applied to the per-sample gradient.
    dp_clip_norm: float = 1.0
    dp_delta: float = 1e-5
    # Sampling rate q = batch_size / local_dataset_size. The RDP accountant
    # needs this to convert the noise level into an (epsilon, delta) pair.
    dp_sample_rate: float = 0.16
    # Rounds used for the projected-epsilon estimate shown in the UI.
    dp_max_rounds: int = 10

    @property
    def dp_epsilon_bound(self) -> float:
        """Projected epsilon for the configured noise level and round count.

        Uses the sampled-Gaussian RDP bound (see
        :mod:`fedmed.privacy.accountant`). This is a *projection* for the
        configured round count, shown next to the live accountant value.
        """
        from .privacy.accountant import quick_epsilon

        steps = max(self.dp_max_rounds, 1) * 60
        return quick_epsilon(
            noise_multiplier=self.dp_noise_multiplier,
            steps=steps,
            sample_rate=self.dp_sample_rate,
            delta=self.dp_delta,
        )


@dataclass(frozen=True)
class ServerConfig:
    """Transport / API configuration.

    Two distinct listeners are used:

    * ``port`` -- the **REST + WebSocket API** consumed by the React
      dashboard.
    * ``fl_port`` -- the **Flower aggregator** (gRPC) that the hospital
      nodes connect to.

    They must not share a port: the orchestrator starts the aggregator and
    the API side by side, so a shared port makes one of them fail to bind.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    fl_port: int = 8001
    grpc_port: int = 50051
    # Certificate authority + server certificate for gRPC/TLS.
    ca_cert: Path = CERT_ROOT / "ca.crt"
    server_cert: Path = CERT_ROOT / "server.crt"
    server_key: Path = CERT_ROOT / "server.key"
    client_cert: Path = CERT_ROOT / "client.crt"
    client_key: Path = CERT_ROOT / "client.key"

    @property
    def fl_address(self) -> str:
        """Address of the Flower aggregator the hospital nodes dial."""
        return f"{self.host}:{self.fl_port}"


@dataclass(frozen=True)
class HospitalConfig:
    """A single simulated hospital node."""

    hospital_id: str
    node_id: str
    port: int
    data_dir: Path
    samples: int = 0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    hospitals: tuple[HospitalConfig, ...] = (
        HospitalConfig(
            hospital_id="hospital-1",
            node_id="HOSP-001",
            port=9101,
            data_dir=DATA_ROOT / "hospital-1",
        ),
        HospitalConfig(
            hospital_id="hospital-2",
            node_id="HOSP-002",
            port=9102,
            data_dir=DATA_ROOT / "hospital-2",
        ),
        HospitalConfig(
            hospital_id="hospital-3",
            node_id="HOSP-003",
            port=9103,
            data_dir=DATA_ROOT / "hospital-3",
        ),
    )


CONFIG = AppConfig()
