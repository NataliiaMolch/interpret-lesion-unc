"""Voxel- and lesion-scale uncertainty measures."""

from __future__ import annotations

import itertools

import numpy as np

from .segmentation import iou, iou_adjusted, labels_without_background


def voxel_uncertainties(probabilities: np.ndarray, epsilon: float = 1e-10) -> dict[str, np.ndarray]:
    """Compute voxel uncertainty maps from member-wise binary-class probabilities.

    Parameters
    ----------
    probabilities:
        Array with shape ``(n_members, x, y, z)`` for the lesion class.
    """

    probs = np.stack([probabilities, 1.0 - probabilities], axis=-1)
    mean_probs = probs.mean(axis=0)
    mean_log_probs = np.log(probs + epsilon).mean(axis=0)
    entropy_of_expected = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
    expected_entropy = -np.mean(np.sum(probs * np.log(probs + epsilon), axis=-1), axis=0)
    mutual_information = entropy_of_expected - expected_entropy
    epkl = -np.sum(mean_probs * mean_log_probs, axis=-1) - expected_entropy

    return {
        "confidence": -np.max(mean_probs, axis=-1),
        "entropy_of_expected": entropy_of_expected,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "epkl": epkl,
        "reverse_mutual_information": epkl - mutual_information,
    }


def lesion_uncertainty_features(
    ensemble_labels: np.ndarray,
    member_labels: np.ndarray,
    member_probabilities: np.ndarray,
    voxel_maps: dict[str, np.ndarray],
    gt_labels: np.ndarray | None = None,
    n_dilation: int = 0,
) -> list[dict[str, float]]:
    """Compute lesion-scale uncertainty rows for each predicted lesion."""

    rows: list[dict[str, float]] = []
    for label in labels_without_background(ensemble_labels):
        rows.append(
            single_lesion_uncertainty(
                ensemble_labels=ensemble_labels,
                lesion_label=int(label),
                member_labels=member_labels,
                member_probabilities=member_probabilities,
                voxel_maps=voxel_maps,
                gt_labels=gt_labels,
                n_dilation=n_dilation,
            )
        )
    return rows


def single_lesion_uncertainty(
    ensemble_labels: np.ndarray,
    lesion_label: int,
    member_labels: np.ndarray,
    member_probabilities: np.ndarray,
    voxel_maps: dict[str, np.ndarray],
    gt_labels: np.ndarray | None,
    n_dilation: int = 0,
    epsilon: float = 1e-5,
) -> dict[str, float]:
    """Compute uncertainty features for one ensemble lesion."""

    component = ensemble_labels == lesion_label
    row: dict[str, float] = {"lesion label": int(lesion_label)}
    for dilation in range(n_dilation + 1):
        if dilation > 0:
            from scipy import ndimage

            component = ndimage.binary_dilation(component, iterations=1)
        suffix = "" if dilation == 0 else f"__{dilation}"
        row.update(_single_scale_features(component, ensemble_labels, member_labels, member_probabilities, voxel_maps, gt_labels, epsilon, suffix))
    return row


def _single_scale_features(
    component: np.ndarray,
    ensemble_labels: np.ndarray,
    member_labels: np.ndarray,
    member_probabilities: np.ndarray,
    voxel_maps: dict[str, np.ndarray],
    gt_labels: np.ndarray | None,
    epsilon: float,
    suffix: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    voxel_count = max(int(component.sum()), 1)

    for name, values in voxel_maps.items():
        features[f"Average {name}{suffix}"] = float(values[component].sum() / voxel_count)

    member_component_masks = np.stack(
        [_best_overlapping_member_component(component, member_labels[i]) for i in range(member_labels.shape[0])],
        axis=0,
    )
    pairwise_dice = [
        _dice(member_component_masks[i], member_component_masks[j])
        for i, j in itertools.combinations(range(member_component_masks.shape[0]), 2)
    ]
    volumes = member_component_masks.sum(axis=(1, 2, 3))
    union = np.any(member_component_masks, axis=0).sum()
    intersection = np.all(member_component_masks, axis=0).sum()

    features[f"ddu{suffix}"] = float(1.0 - np.mean([iou(component, mask) for mask in member_component_masks]))
    features[f"ddu_adj{suffix}"] = float(
        1.0
        - np.mean(
            [
                iou_adjusted(component, ensemble_labels > 0, member_labels[i])
                for i in range(member_labels.shape[0])
            ]
        )
    )
    features[f"ddu_fast{suffix}"] = float(1.0 - np.mean([iou(component, member_labels[i] > 0) for i in range(member_labels.shape[0])]))
    features[f"avg_pwdice{suffix}"] = float(-np.mean(pairwise_dice)) if pairwise_dice else 0.0
    features[f"vvc{suffix}"] = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0
    features[f"global_iou{suffix}"] = float(-intersection / union) if union > 0 else 0.0
    features[f"srmi{suffix}"] = _structural_rmi(component, member_probabilities, epsilon)
    features[f"smi{suffix}"] = _structural_negated_mi(member_component_masks, member_probabilities, epsilon)
    features[f"seoe{suffix}"] = _structural_entropy(component, member_probabilities, epsilon)

    if gt_labels is not None:
        features[f"IoUadj_from_uncertainty{suffix}"] = iou_adjusted(component, ensemble_labels > 0, gt_labels)
    return features


def _best_overlapping_member_component(component: np.ndarray, member_label_image: np.ndarray) -> np.ndarray:
    best_score = 0.0
    best = np.zeros_like(component, dtype=bool)
    for label in np.unique(member_label_image[component]):
        if label == 0:
            continue
        candidate = member_label_image == label
        score = iou(component, candidate)
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = a.sum() + b.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def _structural_rmi(component: np.ndarray, member_probabilities: np.ndarray, epsilon: float) -> float:
    values = np.asarray([probability[component] for probability in member_probabilities])
    if values.size == 0:
        return 0.0
    return float(np.log(values.mean(axis=0) + epsilon).mean() - np.log(values + epsilon).mean())


def _structural_negated_mi(member_component_masks: np.ndarray, member_probabilities: np.ndarray, epsilon: float) -> float:
    value = 0.0
    n_members = member_component_masks.shape[0]
    for i, mask in enumerate(member_component_masks):
        if not mask.any():
            continue
        stacked = np.stack([member_probabilities[k][mask] for k in range(n_members)], axis=0)
        value += float((np.log(stacked.mean(axis=0) + epsilon) - np.log(member_probabilities[i][mask] + epsilon)).mean())
    return float(-value / n_members)


def _structural_entropy(component: np.ndarray, member_probabilities: np.ndarray, epsilon: float) -> float:
    values = np.asarray([probability[component] for probability in member_probabilities])
    if values.size == 0:
        return 0.0
    p1 = float(values.mean(axis=1).prod())
    return float(-p1 * np.log(p1 + epsilon))
