"""Synthetic BraTS-style MRI dataset generation and hospital partitioning.

The real BraTS dataset requires credentialed access and hundreds of gigabytes,
so this module generates a *structurally faithful* surrogate: 3D volumetric
MRI-like scans with multi-class tumour masks. Volumes are shaped like real
BraTS pre-processing output (240x240x155 -> cropped/resampled to a 64^3
patch) and carry the same tumour sub-regions used for evaluation:

    label 0 -> background
    label 1 -> peritumoral edema
    label 2 -> enhancing tumour

A per-hospital *scanner bias* is injected into every volume so the three
nodes are genuinely non-IID. This matters: without distribution shift, FedAvg
trivially matches the centralized baseline and the federated audit becomes
meaningless.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG, DATA_ROOT

LABEL_BACKGROUND = 0
LABEL_EDEMA = 1
LABEL_ENHANCING = 2


@dataclass(frozen=True)
class VolumeMetadata:
    """Provenance record for one synthetic volume."""

    sample_id: str
    hospital_id: str
    shape: tuple[int, int, int]
    seed: int
    scanner_bias: float
    noise_sigma: float
    tumour_volume_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def _smooth_3d(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing (no SciPy dependency)."""
    if sigma <= 0:
        return field

    radius = max(1, int(math.ceil(sigma * 3)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    out = field
    for axis in range(3):
        pad = [(radius, radius) if a == axis else (0, 0) for a in range(3)]
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis, np.pad(out, pad, mode="edge")
        )
    return out


def _make_brain_mask(shape: tuple[int, int, int]) -> np.ndarray:
    """Ellipsoidal brain envelope matching a rough skull-stripped MRI volume."""
    grid = np.stack(
        np.meshgrid(
            *[np.linspace(-1.0, 1.0, s, dtype=np.float32) for s in shape],
            indexing="ij",
        )
    )
    radii = np.array([0.86, 0.98, 0.80], dtype=np.float32)
    dist = np.sqrt(np.sum((grid / radii[:, None, None, None]) ** 2, axis=0))
    return np.clip(1.6 - dist, 0.0, 1.0).astype(np.float32)


def _make_tumour(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a core tumour with a nested enhancing sub-region.

    Returns:
        (edema_mask, enhancing_mask) as uint8 arrays.
    """
    grid = np.stack(
        np.meshgrid(
            *[np.linspace(-1.0, 1.0, s, dtype=np.float32) for s in shape],
            indexing="ij",
        )
    )

    center = rng.uniform(-0.35, 0.35, size=3).astype(np.float32)
    # BraTS core tumour occupies roughly 1-8% of the brain volume. The
    # normalised radius below is tuned so the generated foreground fraction
    # lands in that realistic band rather than the sub-0.1% range that
    # makes global Dice learning degenerate.
    core_radius = scale * rng.uniform(0.34, 0.52)
    anisotropy = rng.uniform(0.7, 1.35, size=3).astype(np.float32)

    offset = grid - center[:, None, None, None]
    dist = np.sqrt(
        np.sum((offset / (core_radius * anisotropy[:, None, None, None])) ** 2, axis=0)
    )
    core = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    core = _smooth_3d(core, sigma=2.2)
    core = (core > 0.40).astype(np.float32)

    # Enhancing sub-region: an off-centre blob inside the core.
    enh_center = center + rng.uniform(-0.30, 0.30, size=3).astype(np.float32) * core_radius
    enh_radius = core_radius * rng.uniform(0.62, 0.88)
    enh_dist = np.sqrt(
        np.sum(
            (
                (grid - enh_center[:, None, None, None])
                / (enh_radius * anisotropy[:, None, None, None])
            )
            ** 2,
            axis=0,
        )
    )
    enhancing = np.clip(1.0 - enh_dist, 0.0, 1.0).astype(np.float32)
    enhancing = _smooth_3d(enhancing, sigma=1.4)
    enhancing = (enhancing > 0.5).astype(np.float32) * core

    edema = np.clip(core - enhancing, 0.0, 1.0)
    return edema.astype(np.uint8), enhancing.astype(np.uint8)


def generate_volume(
    seed: int,
    shape: tuple[int, int, int] = CONFIG.data.patch_size,
    scanner_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, label) pair.

    Args:
        seed: Deterministic seed for this volume.
        shape: Spatial dimensions of the volume.
        scanner_bias: Per-hospital intensity offset, simulating different
            scanner vendors / protocols. Non-IID by construction.

    Returns:
        image: float32 volume in [0, 1], shape ``shape``.
        label: uint8 volume with class ids, shape ``shape``.
    """
    rng = np.random.default_rng(seed)

    brain = _make_brain_mask(shape)

    # Synthetic MRI appearance: smooth intensity texture inside the brain.
    texture = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    texture = _smooth_3d(texture, sigma=2.4)
    texture = (texture - texture.mean()) / (texture.std() + 1e-6)

    intensity = 0.42 + 0.16 * texture
    intensity = intensity * brain
    intensity = np.clip(intensity, 0.0, 1.0)

    # Scanner bias: multiplicative gain + additive offset, then clipped.
    intensity = np.clip(
        intensity * (1.0 + scanner_bias * 0.18) + scanner_bias * 0.05, 0.0, 1.0
    )

    edema, enhancing = _make_tumour(rng, shape, scale=rng.uniform(0.75, 1.25))

    label = np.zeros(shape, dtype=np.uint8)
    label[edema > 0] = LABEL_EDEMA
    label[enhancing > 0] = LABEL_ENHANCING

    # Tumour hyper-intensity on the image so the task is learnable.
    intensity = np.clip(intensity + 0.32 * edema + 0.46 * enhancing, 0.0, 1.0)

    # MRI acquisition noise.
    noise_sigma = float(rng.uniform(0.012, 0.035))
    intensity = np.clip(intensity + rng.normal(0.0, noise_sigma, size=shape), 0.0, 1.0)

    return intensity.astype(np.float32), label


class BrainTumorDataset(Dataset):
    """In-memory dataset of generated volumes.

    Volumes are generated lazily and cached in RAM; with 64^3 patches a
    full split is only a few MB.
    """

    def __init__(
        self,
        hospital_id: str,
        samples: int,
        seed: int,
        scanner_bias: float,
        shape: tuple[int, int, int] = CONFIG.data.patch_size,
        preload: bool = True,
    ) -> None:
        self.hospital_id = hospital_id
        self.samples = samples
        self.seed = seed
        self.scanner_bias = scanner_bias
        self.shape = shape
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] = []

        if preload:
            for i in range(samples):
                img, lbl = generate_volume(
                    seed=seed + i * 7919,
                    shape=shape,
                    scanner_bias=scanner_bias,
                )
                self._cache.append(
                    (
                        torch.from_numpy(img)[None],  # add channel dim
                        torch.from_numpy(lbl.astype(np.int64)),
                    )
                )

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache:
            return self._cache[index]
        img, lbl = generate_volume(
            seed=self.seed + index * 7919,
            shape=self.shape,
            scanner_bias=self.scanner_bias,
        )
        return torch.from_numpy(img)[None], torch.from_numpy(lbl.astype(np.int64))


def _scanner_biases() -> list[float]:
    """Fixed per-hospital scanner biases -> reproducible non-IID split."""
    return [-0.35, 0.05, 0.42]


def build_partition(
    data_root: Path | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """Generate the dataset and partition it across the hospital nodes.

    Each hospital receives a disjoint slice of the global sample space and
    a distinct scanner bias. Also writes a central ``manifest.json`` so the
    baseline trainer can reconstruct the *pooled* dataset: the centralized
    baseline must be trained on exactly the same data as the federated run
    for the comparison to be meaningful.

    Returns:
        Mapping of hospital_id -> partition info.
    """
    root = Path(data_root) if data_root else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    cfg = CONFIG.data
    total_train = cfg.num_samples
    total_val = cfg.num_val_samples
    total_test = cfg.num_test_samples
    biases = _scanner_biases()

    splits = cfg.hospital_split
    train_counts = [int(total_train * s) for s in splits]
    train_counts[-1] += total_train - sum(train_counts)
    val_counts = [int(total_val * s) for s in splits]
    val_counts[-1] += total_val - sum(val_counts)
    test_counts = [int(total_test * s) for s in splits]
    test_counts[-1] += total_test - sum(test_counts)

    manifest: dict = {
        "patch_size": list(cfg.patch_size),
        "train": {},
        "validation": {},
        "test": {},
        "scanner_biases": biases,
        "num_train": total_train,
        "num_val": total_val,
        "num_test": total_test,
    }

    partitions: dict[str, dict] = {}

    for h, hospital in enumerate(CONFIG.hospitals):
        bias = biases[h]
        seed = cfg.seed + h * 104729
        n_train = train_counts[h]
        n_val = val_counts[h]
        n_test = test_counts[h]

        hospital.data_dir.mkdir(parents=True, exist_ok=True)

        train_ds = BrainTumorDataset(
            hospital_id=hospital.hospital_id,
            samples=n_train,
            seed=seed,
            scanner_bias=bias,
        )
        val_ds = BrainTumorDataset(
            hospital_id=hospital.hospital_id,
            samples=n_val,
            seed=seed + 500009,
            scanner_bias=bias,
        )
        test_ds = BrainTumorDataset(
            hospital_id=hospital.hospital_id,
            samples=n_test,
            seed=seed + 900001,
            scanner_bias=bias,
        )

        train_path = hospital.data_dir / "train.pt"
        val_path = hospital.data_dir / "val.pt"
        test_path = hospital.data_dir / "test.pt"

        torch.save(
            {
                "images": torch.stack([train_ds[i][0] for i in range(len(train_ds))]),
                "labels": torch.stack([train_ds[i][1] for i in range(len(train_ds))]),
                "metadata": [
                    VolumeMetadata(
                        sample_id=f"{hospital.node_id}-TRAIN-{i:04d}",
                        hospital_id=hospital.hospital_id,
                        shape=cfg.patch_size,
                        seed=seed + i * 7919,
                        scanner_bias=bias,
                        noise_sigma=0.0,
                        tumour_volume_fraction=0.0,
                    ).to_dict()
                    for i in range(len(train_ds))
                ],
            },
            train_path,
        )
        torch.save(
            {
                "images": torch.stack([val_ds[i][0] for i in range(len(val_ds))]),
                "labels": torch.stack([val_ds[i][1] for i in range(len(val_ds))]),
                "metadata": [
                    VolumeMetadata(
                        sample_id=f"{hospital.node_id}-VAL-{i:04d}",
                        hospital_id=hospital.hospital_id,
                        shape=cfg.patch_size,
                        seed=(seed + 500009) + i * 7919,
                        scanner_bias=bias,
                        noise_sigma=0.0,
                        tumour_volume_fraction=0.0,
                    ).to_dict()
                    for i in range(len(val_ds))
                ],
            },
            val_path,
        )
        torch.save(
            {
                "images": torch.stack([test_ds[i][0] for i in range(len(test_ds))]),
                "labels": torch.stack([test_ds[i][1] for i in range(len(test_ds))]),
                "metadata": [
                    VolumeMetadata(
                        sample_id=f"{hospital.node_id}-TEST-{i:04d}",
                        hospital_id=hospital.hospital_id,
                        shape=cfg.patch_size,
                        seed=(seed + 900001) + i * 7919,
                        scanner_bias=bias,
                        noise_sigma=0.0,
                        tumour_volume_fraction=0.0,
                    ).to_dict()
                    for i in range(len(test_ds))
                ],
            },
            test_path,
        )

        manifest["train"][hospital.hospital_id] = {
            "node_id": hospital.node_id,
            "count": n_train,
            "scanner_bias": bias,
            "seed": seed,
        }
        manifest["validation"][hospital.hospital_id] = {
            "node_id": hospital.node_id,
            "count": n_val,
            "scanner_bias": bias,
            "seed": seed + 500009,
        }
        manifest["test"][hospital.hospital_id] = {
            "node_id": hospital.node_id,
            "count": n_test,
            "scanner_bias": bias,
            "seed": seed + 900001,
        }

        partitions[hospital.hospital_id] = {
            "hospital_id": hospital.hospital_id,
            "node_id": hospital.node_id,
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "scanner_bias": bias,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
        }

    with open(root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return partitions


def load_node_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a node's saved split from disk."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["images"], blob["labels"]


def load_global_validation(
    data_root: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the pooled held-out validation set for evaluation."""
    root = Path(data_root) if data_root else DATA_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run `python -m fedmed.prepare_data` first."
        )

    images, labels = [], []
    for hospital in CONFIG.hospitals:
        blob = torch.load(
            hospital.data_dir / "val.pt", map_location="cpu", weights_only=False
        )
        images.append(blob["images"])
        labels.append(blob["labels"])

    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def load_global_test(
    data_root: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the pooled held-out test set for final evaluation."""
    root = Path(data_root) if data_root else DATA_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run `python -m fedmed.prepare_data` first."
        )

    images, labels = [], []
    for hospital in CONFIG.hospitals:
        test_path = hospital.data_dir / "test.pt"
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test split not found at {test_path}. Run `python -m fedmed.prepare_data` first."
            )
        blob = torch.load(test_path, map_location="cpu", weights_only=False)
        images.append(blob["images"])
        labels.append(blob["labels"])

    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def load_pooled_training_set(
    data_root: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the pooled training set (used by the centralized baseline)."""
    root = Path(data_root) if data_root else DATA_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. Run `python -m fedmed.prepare_data` first."
        )

    images, labels = [], []
    for hospital in CONFIG.hospitals:
        blob = torch.load(
            hospital.data_dir / "train.pt", map_location="cpu", weights_only=False
        )
        images.append(blob["images"])
        labels.append(blob["labels"])

    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def make_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader over in-memory tensors."""
    dataset = torch.utils.data.TensorDataset(images, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def build_local_loaders(
    hospital_id: str,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load one hospital node's local train/val loaders."""
    cfg_hospital = next(h for h in CONFIG.hospitals if h.hospital_id == hospital_id)
    train_images, train_labels = load_node_dataset(cfg_hospital.data_dir / "train.pt")
    val_images, val_labels = load_node_dataset(cfg_hospital.data_dir / "val.pt")

    bs = batch_size or CONFIG.federated.batch_size
    return (
        make_loader(train_images, train_labels, bs, shuffle=True),
        make_loader(val_images, val_labels, bs, shuffle=False),
    )
