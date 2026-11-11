"""MRI segmentation inference and mask rendering.

Runs the trained 3D U-Net over a held-out volume and renders the result
for the dashboard: a grayscale MRI slice, a colour-coded tumour mask
overlay, and per-class Dice statistics.

The rendering is done with pure NumPy so the backend has no image
dependency (Pillow/OpenCV) and the frontend receives plain PNG bytes
encoded as base64.
"""

from __future__ import annotations

import base64
import io
import logging
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from .config import CONFIG, DATA_ROOT, MODEL_ROOT
from .data import load_global_validation
from .metrics import dice_score
from .model import build_model, load_state_dict

logger = logging.getLogger(__name__)

#: Display colours for the three label classes (RGBA).
CLASS_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),  # background
    1: (34, 211, 238, 170),  # edema / core tumour -> cyan
    2: (244, 63, 94, 200),  # enhancing tumour -> rose
}

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Edema",
    2: "Enhancing Tumour",
}


# ---------------------------------------------------------------------------
# Minimal PNG encoder (avoids a Pillow dependency)
# ---------------------------------------------------------------------------


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as a PNG file."""
    height, width, _ = rgb.shape
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw.extend(rgb[row].tobytes())

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pick_slice(mask: np.ndarray, target: int = 1) -> int:
    """Pick the axial slice containing the most tumour voxels."""
    counts = (mask == target).sum(axis=(0, 1))
    return int(np.argmax(counts))


def render_slice(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    axis: int = 2,
    slice_index: int | None = None,
) -> dict:
    """Render one slice as an MRI image plus a mask overlay.

    Args:
        image: Volume of shape ``(D, H, W)`` with intensities in [0, 1].
        label: Ground-truth volume of the same shape.
        prediction: Predicted class volume of the same shape.
        axis: Axis to slice along (2 == axial).
        slice_index: Explicit slice; ``None`` picks the most tumour-heavy one.

    Returns:
        Dict with the base64 PNGs, the slice index and the per-class stats.
    """
    if slice_index is None:
        slice_index = _pick_slice(label)

    def take(arr: np.ndarray) -> np.ndarray:
        return np.take(arr, slice_index, axis=axis)

    img_slice = take(image)
    gt_slice = take(label)
    pred_slice = take(prediction)

    # Grayscale MRI rendering with a simple windowing.
    lo, hi = float(img_slice.min()), float(img_slice.max())
    rng = max(hi - lo, 1e-6)
    gray = np.clip((img_slice - lo) / rng, 0.0, 1.0)
    gray_rgb = (gray[..., None] * np.array([255.0, 255.0, 255.0])).astype(np.uint8)

    # Colour overlay for the prediction.
    overlay = gray_rgb.astype(np.float32).copy()
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask = pred_slice == cls
        if not mask.any():
            continue
        alpha = color[3] / 255.0
        for c in range(3):
            overlay[..., c][mask] = (
                (1 - alpha) * overlay[..., c][mask] + alpha * color[c]
            )
    overlay_rgb = overlay.astype(np.uint8)

    # Dice of this slice alone.
    slice_dice, per_class = dice_score(
        torch.from_numpy(pred_slice.astype(np.int64)).unsqueeze(0),
        torch.from_numpy(gt_slice.astype(np.int64)).unsqueeze(0),
        CONFIG.model.num_classes,
    )

    return {
        "slice_index": slice_index,
        "axis": axis,
        "shape": list(image.shape),
        "mri_png": base64.b64encode(encode_png(gray_rgb)).decode("ascii"),
        "mask_png": base64.b64encode(encode_png(overlay_rgb)).decode("ascii"),
        "slice_dice": round(slice_dice, 2),
        "per_class_dice": {str(k): round(v * 100, 2) for k, v in per_class.items()},
        "classes": [
            {"id": cls, "name": CLASS_NAMES[cls],
             "color": f"rgba({color[0]},{color[1]},{color[2]},{color[3]})"}
            for cls, color in CLASS_COLORS.items()
        ],
    }


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def available_models() -> list[dict]:
    """List the model checkpoints that have been trained so far."""
    out = []
    for name, filename, label in (
        ("baseline", "baseline_model.pt", "Centralized baseline"),
        ("federated", "federated_model.pt", "Federated global model"),
    ):
        path = Path(MODEL_ROOT) / filename
        out.append(
            {
                "id": name,
                "label": label,
                "available": path.exists(),
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return out


def load_model_checkpoint(which: str = "federated") -> tuple[torch.nn.Module | None, dict]:
    """Load a trained checkpoint; returns ``(model, metadata)``."""
    filename = {
        "baseline": "baseline_model.pt",
        "federated": "federated_model.pt",
    }.get(which, "federated_model.pt")
    path = Path(MODEL_ROOT) / filename

    if not path.exists():
        return None, {"available": False, "reason": f"{filename} not found"}

    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    state = blob.get("state_dict")
    if state is None:
        return None, {"available": False, "reason": "checkpoint has no state_dict"}
    load_state_dict(model, state)
    model.eval()

    meta = {
        "available": True,
        "which": which,
        "metrics": blob.get("metrics", {}),
        "config": blob.get("config", {}),
    }
    return model, meta


def run_inference(
    which: str = "federated",
    sample_index: int | None = None,
) -> dict:
    """Segment a held-out validation volume with the selected model.

    Args:
        which: ``"federated"`` or ``"baseline"``.
        sample_index: Index into the pooled validation set; ``None`` picks
            the sample with the most tumour voxels (the most informative).

    Returns:
        A payload with the rendered slices, the volume-level Dice and the
        per-class statistics.
    """
    model, meta = load_model_checkpoint(which)
    if model is None:
        return {
            "available": False,
            "reason": meta.get("reason", "model unavailable"),
            "models": available_models(),
        }

    val_x, val_y = load_global_validation()

    if sample_index is None:
        # Pick the volume with the largest foreground fraction.
        fg = (val_y > 0).flatten(1).sum(dim=1)
        sample_index = int(torch.argmax(fg))

    image = val_x[sample_index].numpy()[0]  # drop the channel dimension
    label = val_y[sample_index].numpy()

    with torch.no_grad():
        logits = model(val_x[sample_index : sample_index + 1])
        prediction = torch.argmax(logits, dim=1)[0].numpy()

    volume_dice, per_class = dice_score(
        torch.from_numpy(prediction.astype(np.int64)).unsqueeze(0),
        torch.from_numpy(label.astype(np.int64)).unsqueeze(0),
        CONFIG.model.num_classes,
    )

    rendered = render_slice(image, label, prediction)

    tumor_voxels = int((prediction > 0).sum())
    truth_voxels = int((label > 0).sum())

    return {
        "available": True,
        "model": which,
        "sample_index": sample_index,
        "total_samples": int(val_x.shape[0]),
        "volume_dice": round(volume_dice, 2),
        "per_class_dice": {str(k): round(v * 100, 2) for k, v in per_class.items()},
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "tumor_voxels": tumor_voxels,
        "truth_voxels": truth_voxels,
        "tumor_fraction": round(tumor_voxels / max(label.size, 1) * 100, 3),
        "metrics": meta.get("metrics", {}),
        **rendered,
    }


def dataset_overview() -> dict:
    """Summary statistics of the generated dataset for the MRI page."""
    manifest_path = Path(DATA_ROOT) / "manifest.json"
    if not manifest_path.exists():
        return {"available": False}

    import json

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    return {
        "available": True,
        "num_train": manifest.get("num_train", 0),
        "num_val": manifest.get("num_val", 0),
        "patch_size": manifest.get("patch_size", []),
        "scanner_biases": manifest.get("scanner_biases", []),
        "hospitals": [
            {
                "hospital_id": hid,
                "node_id": info.get("node_id"),
                "train": info.get("count", 0),
                "scanner_bias": info.get("scanner_bias", 0.0),
            }
            for hid, info in manifest.get("train", {}).items()
        ],
    }
