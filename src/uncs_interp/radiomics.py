"""Radiomics feature extraction for arbitrary nnU-Net image channels."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .segmentation import labels_without_background


def normalized_sitk_image(path: Path):
    """Load and z-score normalize non-zero voxels in an image."""

    import SimpleITK as sitk

    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    values = sitk.GetArrayFromImage(image)
    non_zero = values[values != 0]
    if non_zero.size:
        mean = float(non_zero.mean())
        std = float(non_zero.std()) or 1.0
        image = (image - mean) / std
    return image


def lesion_radiomics_table(
    image_paths: dict[str, Path],
    lesion_labels: np.ndarray,
    affine: np.ndarray,
    surrounding_iterations: int = 4,
) -> pd.DataFrame:
    """Compute PyRadiomics features for each lesion and each image channel."""

    try:
        import radiomics.featureextractor
    except ImportError as exc:
        raise ImportError("Install pyradiomics to compute radiomics features.") from exc
    import SimpleITK as sitk

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(verbose=False)
    rows: list[dict[str, float]] = []
    lesion_ids = labels_without_background(lesion_labels).astype(int)
    if lesion_ids.size == 0:
        return pd.DataFrame(columns=["lesion label"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        label_path = tmpdir_path / "lesions.nii.gz"
        nib.save(nib.Nifti1Image(lesion_labels.astype(np.uint16), affine), str(label_path))
        label_sitk = sitk.ReadImage(str(label_path), sitk.sitkUInt16)
        label_array = sitk.GetArrayFromImage(label_sitk)

        modality_images = {name: normalized_sitk_image(path) for name, path in image_paths.items()}
        for lesion_id in lesion_ids:
            row: dict[str, float] = {"lesion label": int(lesion_id)}
            lesion_mask_array = (label_array == lesion_id).astype(np.uint8)
            surrounding_array = (
                ndimage.binary_dilation(lesion_mask_array, iterations=surrounding_iterations)
                & ~lesion_mask_array.astype(bool)
            ).astype(np.uint8)

            lesion_mask = sitk.GetImageFromArray(lesion_mask_array)
            lesion_mask.CopyInformation(label_sitk)
            surrounding_mask = sitk.GetImageFromArray(surrounding_array)
            surrounding_mask.CopyInformation(label_sitk)

            for modality, image in modality_images.items():
                _append_radiomics(row, modality, _safe_execute(extractor, image, lesion_mask))
                _append_radiomics(
                    row,
                    f"surrounding {modality}",
                    _safe_execute(extractor, image, surrounding_mask),
                )
            row["# pred voxels"] = int((lesion_labels == lesion_id).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def patient_radiomics_row(
    image_paths: dict[str, Path],
    prediction_mask: np.ndarray,
    affine: np.ndarray,
    brain_mask: np.ndarray | None = None,
    surrounding_iterations: int = 4,
) -> dict[str, float]:
    """Compute patient-scale radiomics for the full predicted mask, surrounding, and brain."""

    try:
        import radiomics.featureextractor
    except ImportError as exc:
        raise ImportError("Install pyradiomics to compute radiomics features.") from exc
    import SimpleITK as sitk

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(verbose=False)
    row: dict[str, float] = {}
    if brain_mask is None:
        brain_mask = np.any(
            np.stack([nib.load(str(path)).get_fdata() != 0 for path in image_paths.values()], axis=0),
            axis=0,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pred_path = tmpdir_path / "prediction.nii.gz"
        brain_path = tmpdir_path / "brain.nii.gz"
        surrounding_path = tmpdir_path / "surrounding.nii.gz"
        surrounding = (
            ndimage.binary_dilation(prediction_mask.astype(bool), iterations=surrounding_iterations)
            & ~prediction_mask.astype(bool)
        )
        nib.save(nib.Nifti1Image(prediction_mask.astype(np.uint8), affine), str(pred_path))
        nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), affine), str(brain_path))
        nib.save(nib.Nifti1Image(surrounding.astype(np.uint8), affine), str(surrounding_path))

        pred_mask = sitk.ReadImage(str(pred_path), sitk.sitkUInt16)
        brain_mask_sitk = sitk.ReadImage(str(brain_path), sitk.sitkUInt16)
        surrounding_mask = sitk.ReadImage(str(surrounding_path), sitk.sitkUInt16)
        for modality, path in image_paths.items():
            image = normalized_sitk_image(path)
            if prediction_mask.any():
                _append_radiomics(row, f"patient {modality}", _safe_execute(extractor, image, pred_mask))
                _append_radiomics(
                    row,
                    f"patient surrounding {modality}",
                    _safe_execute(extractor, image, surrounding_mask),
                )
            _append_radiomics(row, f"patient brain {modality}", _safe_execute(extractor, image, brain_mask_sitk))
        row["patient # pred voxels"] = int(prediction_mask.sum())
    return row


def _safe_execute(extractor, image, mask) -> dict:
    try:
        return dict(extractor.execute(image, mask))
    except Exception:
        return {}


def _append_radiomics(row: dict[str, float], prefix: str, features: dict) -> None:
    for key, value in features.items():
        if "original" in key and "diagnostics" not in key:
            row[f"{prefix} {key}"] = float(value)
