"""MRIQC guidance helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SELECTED_MRIQC_FEATURES = [
    "cjv",
    "efc",
    "icvs_gm",
    "icvs_wm",
    "inu_med",
    "inu_range",
    "rpve_gm",
    "rpve_wm",
    "snr_total",
    "snr_gm",
    "snr_wm",
    "wm2max",
    "fwhm_avg",
]


def mriqc_docker_command(bids_dir: Path, output_dir: Path, participant_label: str | None = None) -> str:
    """Return a Docker command for running MRIQC on a BIDS-like dataset."""

    participant = f" --participant-label {participant_label}" if participant_label else ""
    return (
        "docker run --rm -it "
        f"-v {bids_dir.resolve()}:/data:ro "
        f"-v {output_dir.resolve()}:/out "
        "nipreps/mriqc:latest /data /out participant"
        f"{participant}"
    )


def select_mriqc_features(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Keep the MRIQC IQMs used by the uncertainty analysis."""

    table = pd.read_csv(input_csv, sep=None, engine="python")
    id_columns = [column for column in ("case_id", "filename", "bids_name") if column in table.columns]
    selected = [column for column in SELECTED_MRIQC_FEATURES if column in table.columns]
    result = table[id_columns + selected].copy()
    if "case_id" not in result.columns:
        source = "filename" if "filename" in result.columns else "bids_name"
        result["case_id"] = result[source].astype(str).str.replace(".nii.gz", "", regex=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
