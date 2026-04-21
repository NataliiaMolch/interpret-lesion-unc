"""Atlas/MNI-derived lesion location features."""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .segmentation import labels_without_background


@dataclass(frozen=True)
class AtlasConfig:
    """One bilateral atlas used for normalized intersection features."""

    name: str
    root: Path
    left: str
    right: str
    labels_xml: str


class Atlas:
    """Atlas that computes per-lesion fractions overlapping each structure."""

    def __init__(self, config: AtlasConfig):
        self.config = config
        self.labels = self._load_labels()

    def _load_labels(self) -> dict[int, str]:
        tree = ET.parse(self.config.root / self.config.labels_xml)
        labels: dict[int, str] = {}
        for child in tree.getroot().find("data").iter("label"):
            labels[int(child.attrib["index"]) + 1] = f"{self.config.name} {child.text}"
        return labels

    def features(self, label_image: np.ndarray) -> pd.DataFrame:
        lesion_labels = labels_without_background(label_image)
        rows = pd.DataFrame({"lesion label": lesion_labels.astype(int)})
        if lesion_labels.size == 0:
            return rows

        feature_columns: dict[str, list[float]] = {}
        for atlas_file, side in [(self.config.left, "L"), (self.config.right, "R")]:
            atlas_data = nib.load(str(self.config.root / atlas_file)).get_fdata()
            for structure_label, structure_name in self.labels.items():
                column = f"{structure_name} {side}"
                values = []
                structure_mask = atlas_data == structure_label
                for lesion_label in lesion_labels:
                    lesion_mask = label_image == lesion_label
                    values.append(float(np.logical_and(lesion_mask, structure_mask).sum() / lesion_mask.sum()))
                feature_columns[column] = values
        return pd.concat([rows, pd.DataFrame(feature_columns)], axis=1)


def load_atlases(config_path: Path | None) -> list[Atlas]:
    """Load atlas definitions from a JSON file."""

    if config_path is None:
        return []
    data = json.loads(config_path.read_text())
    atlases = []
    for item in data.get("atlases", data if isinstance(data, list) else []):
        root = Path(item["root"]).expanduser()
        atlases.append(
            Atlas(
                AtlasConfig(
                    name=item["name"],
                    root=root,
                    left=item["left"],
                    right=item["right"],
                    labels_xml=item["labels_xml"],
                )
            )
        )
    return atlases


def apply_transform_to_mni(
    label_image: np.ndarray,
    affine: np.ndarray,
    transform_path: Path,
    fixed_image_path: Path,
) -> np.ndarray:
    """Apply an ANTs transform to a label image using nearest-neighbor interpolation."""

    try:
        import ants
    except ImportError as exc:
        raise ImportError("Install the optional 'mni' dependencies to use ANTs MNI features.") from exc

    moving = ants.from_nibabel(nib.Nifti1Image(label_image.astype(np.float32), affine=affine))
    fixed = ants.image_read(str(fixed_image_path))
    transformed = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(transform_path)],
        interpolator="nearestNeighbor",
    )
    return np.asarray(transformed.numpy())


def register_to_mni_affine(
    image_path: Path,
    fixed_image_path: Path,
    output_transform_path: Path,
) -> Path:
    """Compute an affine subject-to-MNI transform with ANTs."""

    try:
        import ants
    except ImportError as exc:
        raise ImportError("Install the optional 'mni' dependencies to run ANTs registration.") from exc

    output_transform_path.parent.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(fixed_image_path))
    moving = ants.image_read(str(image_path))
    with tempfile.TemporaryDirectory() as tmpdir:
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine",
            outprefix=str(Path(tmpdir) / "to_mni_"),
        )
        source = Path(result["fwdtransforms"][0])
        output_transform_path.write_bytes(source.read_bytes())
    return output_transform_path


def atlas_feature_table(
    label_image: np.ndarray,
    atlases: list[Atlas],
) -> pd.DataFrame:
    """Compute and merge location features from all configured atlases."""

    if not atlases:
        return pd.DataFrame({"lesion label": labels_without_background(label_image).astype(int)})
    frames = [atlas.features(label_image) for atlas in atlases]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="lesion label", how="outer")
    return result
