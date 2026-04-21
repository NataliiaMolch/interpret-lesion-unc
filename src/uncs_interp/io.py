"""I/O helpers for nnU-Net-style datasets and predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


NIFTI_SUFFIXES = (".nii.gz", ".nii")


def strip_nii_suffix(path: Path) -> str:
    """Return a NIfTI stem without ``.nii`` or ``.nii.gz``."""

    name = path.name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def case_id_from_image(path: Path) -> str:
    """Return nnU-Net case id by removing the channel suffix ``_0000``."""

    stem = strip_nii_suffix(path)
    return re.sub(r"_\d{4}$", "", stem)


def discover_images(images_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover nnU-Net images grouped by case id and channel id."""

    cases: dict[str, dict[str, Path]] = {}
    for path in sorted(images_dir.glob("*.nii*")):
        stem = strip_nii_suffix(path)
        match = re.match(r"(.+)_(\d{4})$", stem)
        if not match:
            continue
        case_id, channel = match.groups()
        cases.setdefault(case_id, {})[channel] = path
    return cases


def load_dataset_json(dataset_dir: Path) -> dict:
    """Load nnU-Net ``dataset.json`` when present."""

    path = dataset_dir / "dataset.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def channel_names(dataset_dir: Path, channels: list[str]) -> dict[str, str]:
    """Resolve human-readable channel names from ``dataset.json`` when available."""

    dataset_json = load_dataset_json(dataset_dir)
    raw = dataset_json.get("channel_names") or dataset_json.get("modality") or {}
    resolved: dict[str, str] = {}
    for channel in channels:
        index = int(channel)
        resolved[channel] = str(raw.get(str(index), raw.get(index, f"modality_{channel}")))
    return resolved


def discover_labels(labels_dir: Path | None) -> dict[str, Path]:
    """Discover label files keyed by case id."""

    if labels_dir is None:
        return {}
    return {strip_nii_suffix(path): path for path in sorted(labels_dir.glob("*.nii*"))}


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI image as float data plus the image object."""

    image = nib.load(str(path))
    return image.get_fdata(), image


def load_probability(path: Path, class_index: int = 1) -> tuple[np.ndarray, nib.Nifti1Image | None]:
    """Load a lesion-class probability map from nnU-Net ``.npz`` or NIfTI output."""

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ("probabilities", "softmax", "pred_probs", "arr_0"):
            if key in data.files:
                values = data[key]
                break
        else:
            raise ValueError(f"No known probability key found in {path}. Keys: {data.files}")
        if values.ndim == 4:
            if values.shape[0] <= class_index:
                raise ValueError(f"Class index {class_index} not available in {path} with shape {values.shape}.")
            values = values[class_index]
        if values.ndim != 3:
            raise ValueError(f"Expected a 3D probability map in {path}, got shape {values.shape}.")
        return np.asarray(values, dtype=np.float32), None

    values, image = load_nifti(path)
    return np.asarray(values, dtype=np.float32), image


def prediction_path(member_dir: Path, case_id: str) -> Path:
    """Find a member prediction for ``case_id``."""

    candidates = [
        member_dir / f"{case_id}.npz",
        member_dir / f"{case_id}.nii.gz",
        member_dir / f"{case_id}.nii",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No prediction found for case {case_id} in {member_dir}. Expected .npz, .nii.gz, or .nii."
    )


def load_member_probabilities(
    member_dirs: list[Path],
    case_id: str,
    class_index: int = 1,
) -> np.ndarray:
    """Load all member probability maps for one case."""

    probabilities = []
    for member_dir in member_dirs:
        probability, _ = load_probability(prediction_path(member_dir, case_id), class_index=class_index)
        probabilities.append(probability)
    shapes = {probability.shape for probability in probabilities}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes differ for {case_id}: {sorted(shapes)}")
    return np.stack(probabilities, axis=0)


def save_like(reference: nib.Nifti1Image, values: np.ndarray, output_path: Path) -> None:
    """Save ``values`` with affine/header copied from ``reference``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values, reference.affine, reference.header), str(output_path))
