"""End-to-end feature extraction for nnU-Net-style uncertainty analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import (
    channel_names,
    discover_images,
    discover_labels,
    load_member_probabilities,
    load_nifti,
)
from .location import (
    apply_transform_to_mni,
    atlas_feature_table,
    load_atlases,
    register_to_mni_affine,
)
from .radiomics import lesion_radiomics_table, patient_radiomics_row
from .segmentation import (
    LesionDetectionConfig,
    label_components,
    lesion_quality_features,
    threshold_probability,
)
from .uncertainty import lesion_uncertainty_features, voxel_uncertainties


@dataclass
class FeatureExtractionConfig:
    """Configuration for public feature extraction."""

    dataset_dir: Path
    predictions: list[Path]
    output_csv: Path
    images_subdir: str = "imagesTs"
    labels_subdir: str | None = None
    class_index: int = 1
    ensemble_threshold: float = 0.5
    member_thresholds: list[float] | None = None
    min_lesion_size: int = 2
    detection_threshold: float = 0.1
    detection_method: str = "iou_adj"
    atlas_config: Path | None = None
    transforms_dir: Path | None = None
    transform_pattern: str = "{case_id}_to_mni.mat"
    fixed_mni: Path | None = None
    register_missing_transforms: bool = False
    mriqc_csv: Path | None = None
    skip_radiomics: bool = False
    n_dilation: int = 0


def parse_thresholds(path: Path | None, n_members: int, default: float = 0.5) -> list[float]:
    """Parse member thresholds from JSON/CSV/text or return defaults."""

    if path is None:
        return [default] * n_members
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            values = list(data.values())
        else:
            values = list(data)
    else:
        table = pd.read_csv(path)
        if "threshold" in table.columns:
            values = table["threshold"].tolist()
        else:
            values = table.iloc[:, -1].tolist()
    if len(values) != n_members:
        raise ValueError(f"Expected {n_members} member thresholds, got {len(values)} from {path}.")
    return [float(value) for value in values]


def extract_features(config: FeatureExtractionConfig) -> pd.DataFrame:
    """Extract all requested lesion-level features and save a single CSV."""

    images_dir = config.dataset_dir / config.images_subdir
    labels_dir = config.dataset_dir / config.labels_subdir if config.labels_subdir else None
    cases = discover_images(images_dir)
    labels = discover_labels(labels_dir)
    if not cases:
        raise ValueError(f"No nnU-Net images found in {images_dir}.")

    thresholds = config.member_thresholds or [0.5] * len(config.predictions)
    atlases = load_atlases(config.atlas_config)
    mriqc = _load_mriqc(config.mriqc_csv)
    all_rows: list[pd.DataFrame] = []

    for case_id, image_channels in cases.items():
        print(f"Extracting features for {case_id}")
        reference_data, reference_image = load_nifti(next(iter(image_channels.values())))
        image_names = channel_names(config.dataset_dir, sorted(image_channels))
        modality_paths = {image_names[channel]: path for channel, path in image_channels.items()}

        member_probabilities = load_member_probabilities(config.predictions, case_id, config.class_index)
        ensemble_probability = member_probabilities.mean(axis=0)
        ensemble_mask = threshold_probability(
            ensemble_probability,
            threshold=config.ensemble_threshold,
            min_size=config.min_lesion_size,
        )
        ensemble_labels, _ = label_components(ensemble_mask)
        member_labels = np.stack(
            [
                label_components(
                    threshold_probability(probability, threshold=threshold, min_size=config.min_lesion_size)
                )[0]
                for probability, threshold in zip(member_probabilities, thresholds)
            ],
            axis=0,
        )

        gt_mask = None
        gt_labels = None
        if case_id in labels:
            gt_mask = load_nifti(labels[case_id])[0] > 0
            gt_labels, _ = label_components(gt_mask)

        rows = _case_feature_table(
            case_id=case_id,
            filename=case_id,
            ensemble_labels=ensemble_labels,
            member_labels=member_labels,
            member_probabilities=member_probabilities,
            gt_mask=gt_mask,
            gt_labels=gt_labels,
            modality_paths=modality_paths,
            affine=reference_image.affine,
            reference_image_path=next(iter(image_channels.values())),
            atlases=atlases,
            config=config,
        )
        if case_id in mriqc:
            for key, value in mriqc[case_id].items():
                rows[f"mriqc {key}"] = value
        all_rows.append(rows)
        del reference_data

    result = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv, index=False)
    return result


def _case_feature_table(
    case_id: str,
    filename: str,
    ensemble_labels: np.ndarray,
    member_labels: np.ndarray,
    member_probabilities: np.ndarray,
    gt_mask: np.ndarray | None,
    gt_labels: np.ndarray | None,
    modality_paths: dict[str, Path],
    affine: np.ndarray,
    reference_image_path: Path,
    atlases: list,
    config: FeatureExtractionConfig,
) -> pd.DataFrame:
    if not np.any(ensemble_labels > 0):
        return pd.DataFrame(columns=["case_id", "filename", "lesion label"])

    voxel_maps = voxel_uncertainties(member_probabilities)
    uncertainty_df = pd.DataFrame(
        lesion_uncertainty_features(
            ensemble_labels=ensemble_labels,
            member_labels=member_labels,
            member_probabilities=member_probabilities,
            voxel_maps=voxel_maps,
            gt_labels=gt_labels,
            n_dilation=config.n_dilation,
        )
    )
    quality_df = pd.DataFrame(
        lesion_quality_features(
            ensemble_labels,
            gt_mask,
            LesionDetectionConfig(config.detection_method, config.detection_threshold),
        )
    )
    frames = [quality_df, uncertainty_df]

    if not config.skip_radiomics:
        lesion_rad = lesion_radiomics_table(modality_paths, ensemble_labels, affine=affine)
        patient_rad = patient_radiomics_row(modality_paths, ensemble_labels > 0, affine=affine)
        for key, value in patient_rad.items():
            lesion_rad[key] = value
        frames.append(lesion_rad)

    if atlases:
        labels_for_location = ensemble_labels
        if config.fixed_mni is not None:
            transform_path = _transform_path(case_id, config)
            if not transform_path.exists() and config.register_missing_transforms:
                register_to_mni_affine(reference_image_path, config.fixed_mni, transform_path)
            if not transform_path.exists():
                raise FileNotFoundError(
                    f"No MNI transform for {case_id}: {transform_path}. "
                    "Provide transforms or use --register-missing-transforms."
                )
            labels_for_location = apply_transform_to_mni(
                ensemble_labels,
                affine=affine,
                transform_path=transform_path,
                fixed_image_path=config.fixed_mni,
            )
        frames.append(atlas_feature_table(labels_for_location, atlases))

    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, on="lesion label", how="outer")
    data.insert(0, "filename", filename)
    data.insert(0, "case_id", case_id)
    return data


def _transform_path(case_id: str, config: FeatureExtractionConfig) -> Path:
    if config.transforms_dir is None:
        raise ValueError("MNI transforms require --transforms-dir or --register-missing-transforms.")
    return config.transforms_dir / config.transform_pattern.format(case_id=case_id)


def _load_mriqc(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    table = pd.read_csv(path, sep=None, engine="python")
    if "case_id" not in table.columns:
        if "filename" in table.columns:
            table["case_id"] = table["filename"].astype(str).str.replace(".nii.gz", "", regex=False)
        else:
            raise ValueError("MRIQC CSV must contain a 'case_id' or 'filename' column.")
    rows = {}
    for _, row in table.iterrows():
        values = row.drop(labels=[column for column in ("case_id", "filename") if column in row.index])
        rows[str(row["case_id"])] = values.to_dict()
    return rows
