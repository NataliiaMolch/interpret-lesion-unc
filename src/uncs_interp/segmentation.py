"""Segmentation utilities used across feature extraction and analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LesionDetectionConfig:
    """Configuration for optional GT-aware lesion typing."""

    method: str = "iou_adj"
    threshold: float = 0.1


def remove_small_components(mask: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Remove connected components with voxel count <= ``min_size``."""

    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")
    if min_size <= 0:
        return mask.astype(np.uint8)

    labels, n_labels = label_components(mask)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels + 1):
        component = labels == label
        if int(component.sum()) > min_size:
            cleaned[component] = 1
    return cleaned


def threshold_probability(probability: np.ndarray, threshold: float, min_size: int = 2) -> np.ndarray:
    """Threshold a probability map and remove tiny connected components."""

    return remove_small_components((probability >= threshold).astype(np.uint8), min_size=min_size)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return 26-connected labels for a 3D binary mask."""

    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    return ndimage.label(mask.astype(bool), structure=structure)


def iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Intersection over union for binary masks."""

    union = np.logical_or(y_pred, y_true).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(y_pred, y_true).sum() / union)


def iou_adjusted(
    component_mask: np.ndarray,
    prediction_mask: np.ndarray,
    gt_labels: np.ndarray,
) -> float:
    """Adjusted IoU from the paper code.

    ``component_mask`` is one predicted lesion. ``prediction_mask`` is the full
    binary prediction. ``gt_labels`` is the connected-component-labeled GT mask.
    Overlap with GT regions already claimed by other predicted lesions is removed
    from the denominator.
    """

    component_mask = component_mask.astype(bool)
    prediction_mask = prediction_mask.astype(bool)
    intersecting_gt = np.unique(gt_labels[component_mask])
    intersecting_gt = intersecting_gt[intersecting_gt != 0]
    if intersecting_gt.size == 0:
        return 0.0

    gt_union = np.isin(gt_labels, intersecting_gt)
    other_predictions = np.logical_and(prediction_mask, np.logical_not(component_mask))
    gt_minus_other_predictions = np.logical_and(gt_union, np.logical_not(other_predictions))
    numerator = np.logical_and(component_mask, gt_union).sum()
    denominator = np.logical_or(component_mask, gt_minus_other_predictions).sum()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def lesion_type(
    component_mask: np.ndarray,
    prediction_labels: np.ndarray,
    gt_labels: np.ndarray,
    config: LesionDetectionConfig,
) -> str:
    """Classify a predicted lesion as TPL/FPL/USL when GT is available."""

    if config.method == "non_zero":
        return "TPL" if np.any(gt_labels[component_mask.astype(bool)] > 0) else "FPL"
    if config.method == "iou":
        values = [
            iou(component_mask, gt_labels == gt_label)
            for gt_label in np.unique(gt_labels[component_mask.astype(bool)])
            if gt_label != 0
        ]
        score = max(values) if values else 0.0
    elif config.method == "iou_adj":
        score = iou_adjusted(component_mask, prediction_labels > 0, gt_labels)
    else:
        raise ValueError(f"Unsupported lesion detection method: {config.method}")

    if score >= config.threshold:
        return "TPL"
    if score > 0:
        return "USL"
    return "FPL"


def lesion_quality_features(
    prediction_labels: np.ndarray,
    gt_mask: np.ndarray | None,
    config: LesionDetectionConfig | None = None,
) -> list[dict[str, object]]:
    """Compute optional GT-aware quality features for predicted lesions."""

    rows: list[dict[str, object]] = []
    if gt_mask is None:
        for label in labels_without_background(prediction_labels):
            rows.append({"lesion label": int(label), "lesion type": "predicted"})
        return rows

    gt_labels, _ = label_components(gt_mask)
    pred_binary = prediction_labels > 0
    config = config or LesionDetectionConfig()

    for label in labels_without_background(prediction_labels):
        component = prediction_labels == label
        max_iou = max(
            [
                iou(component, gt_labels == gt_label)
                for gt_label in np.unique(gt_labels[component])
                if gt_label != 0
            ]
            or [0.0]
        )
        rows.append(
            {
                "lesion label": int(label),
                "lesion type": lesion_type(component, prediction_labels, gt_labels, config),
                "IoU": iou(component, gt_mask > 0),
                "maxIoU": max_iou,
                "IoUadj": iou_adjusted(component, pred_binary, gt_labels),
            }
        )
    return rows


def labels_without_background(label_image: np.ndarray) -> np.ndarray:
    """Return non-zero labels from a labeled image."""

    labels = np.unique(label_image)
    return labels[labels != 0]
