# uncs-interp

Public toolkit for lesion-wise uncertainty interpretability from member-wise nnU-Net probability maps.

This repository is a complement to the paper *Explaining Uncertainty in Multiple Sclerosis Cortical Lesion Segmentation Beyond Prediction Errors*.

The package computes lesion-level uncertainty, segmentation-quality features when ground truth is available, multi-modal radiomics, MNI atlas overlap features, optional MRIQC IQMs, novelty features, ElasticNet interpretability models, and counterfactual reports asking:

> What would need to be different about this lesion for the model to be certain about it?

Code for the preliminary MICCAI iMIMIC research is available in the `miccai-imimic-2024` branch.

The nnU-Net v2 model weights used for the associated work are available on Zenodo: `10.5281/zenodo.19485057`.

## Install

```bash
cd interpret-lesion-unc
python -m pip install -e .
```

For MNI registration/location features, install the optional ANTs dependency:

```bash
python -m pip install -e ".[mni]"
```

## Expected Inputs

The dataset should follow the nnU-Net v2 layout:

```text
DatasetXXX_Name/
  dataset.json
  imagesTs/
    case_001_0000.nii.gz
    case_001_0001.nii.gz
  labelsTs/                 # optional
    case_001.nii.gz
```

Predictions should be one directory per model member/fold. Each directory must contain `case_id.npz`, `case_id.nii.gz`, or `case_id.nii`. For `.npz`, keys named `probabilities`, `softmax`, `pred_probs`, or `arr_0` are supported. If the array is 4D, `--class-index` selects the lesion class.

## Feature Extraction

```bash
uncs-interp extract-features \
  --dataset-dir /path/to/DatasetXXX_Name \
  --predictions /path/to/fold_0 /path/to/fold_1 /path/to/fold_2 /path/to/fold_3 /path/to/fold_4 \
  --labels-subdir labelsTs \
  --output-csv outputs/features.csv \
  --ensemble-threshold 0.5
```

Ground truth is optional. Without `--labels-subdir`, IoU/IoUadj and lesion type are skipped, but uncertainty, radiomics, location, MRIQC, and novelty-compatible features can still be computed.

To include MNI-derived location features:

```bash
uncs-interp extract-features \
  --dataset-dir /path/to/DatasetXXX_Name \
  --predictions /path/to/fold_0 /path/to/fold_1 \
  --output-csv outputs/features_mni.csv \
  --atlas-config configs/example_atlas_config.json \
  --fixed-mni /path/to/MNI152lin_T1_1mm_brain.nii.gz \
  --transforms-dir /path/to/transforms \
  --transform-pattern "{case_id}_to_mni.mat"
```

If transforms are missing and ANTs is installed, add `--register-missing-transforms` to compute affine transforms from the first nnU-Net channel to MNI.

## MRIQC

MRIQC is intentionally run outside this package. Generate a command:

```bash
uncs-interp mriqc-command --bids-dir /path/to/bids --output-dir outputs/mriqc
```

Then select the IQMs used in the analysis:

```bash
uncs-interp select-mriqc \
  --input-csv outputs/mriqc/group_T1w.tsv \
  --output-csv outputs/mriqc_selected.csv
```

Pass the selected CSV to feature extraction with `--mriqc-csv`.

## Novelty Features

Use the training split as the reference distribution:

```bash
uncs-interp add-novelty \
  --reference-csv outputs/train_features.csv \
  --target-csv outputs/test_features.csv \
  --output-csv outputs/test_features_with_novelty.csv
```

## ElasticNet Interpretability

Train without IoUadj:

```bash
uncs-interp train-elasticnet \
  --train-csv outputs/train_features.csv \
  --model-out outputs/elasticnet.joblib
```

Train with IoUadj as an explanatory feature:

```bash
uncs-interp train-elasticnet \
  --train-csv outputs/train_features.csv \
  --model-out outputs/elasticnet_with_iou.joblib \
  --include-iou-adj
```

The command writes:

- `elasticnet.joblib`: trained model artifact
- `elasticnet.feature_importance.csv`: selected ElasticNet coefficients
- `elasticnet.metrics.csv`: cross-validation and training metrics

## Counterfactual Explanations

Given a trained model and features for a new scan:

```bash
uncs-interp counterfactual \
  --model outputs/elasticnet.joblib \
  --features-csv outputs/new_scan_features.csv \
  --output-csv outputs/new_scan_counterfactuals.csv \
  --target-uncertainty 0
```

The output ranks one-feature counterfactuals by the smallest required change. These are descriptive explanations of model behavior, not clinically actionable instructions.

## Important Notes

- Probability thresholds default to `0.5`; pass `--member-thresholds thresholds.json` for fold/member-specific thresholds.
- Multiple MRI modalities are supported through nnU-Net channel files and `dataset.json` channel names.
- PyRadiomics features are computed for each predicted lesion, its surrounding region, and patient-level predicted lesion/brain regions.
- MNI atlas features require either existing transforms or optional ANTs-based affine registration.
- IoUadj follows the adjusted IoU definition from the research code and is only available when ground truth labels are provided.

## Citation

If you use this repository, please cite:

```bibtex
@misc{molchanova2026explaininguncertaintymultiplesclerosis,
      title={Explaining Uncertainty in Multiple Sclerosis Cortical Lesion Segmentation Beyond Prediction Errors}, 
      author={Nataliia Molchanova and Pedro M. Gordaliza and Alessandro Cagol and Mario Ocampo--Pineda and Po--Jui Lu and Matthias Weigel and Xinjie Chen and Erin S. Beck and Haris Tsagkas and Daniel Reich and Anna Stölting and Pietro Maggi and Delphine Ribes and Adrien Depeursinge and Cristina Granziera and Henning Müller and Meritxell Bach Cuadra},
      year={2026},
      eprint={2504.04814},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2504.04814}, 
}
```

Related work:

```bibtex
@inproceedings{molchanovaiee2023,
   title={Novel Structural-Scale Uncertainty Measures and Error Retention Curves: Application to Multiple Sclerosis},
   url={http://dx.doi.org/10.1109/ISBI53787.2023.10230563},
   DOI={10.1109/isbi53787.2023.10230563},
   booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
   publisher={IEEE},
   author={Molchanova, Nataliia and Raina, Vatsal and Malinin, Andrey and La Rosa, Francesco and Muller, Henning and Gales, Mark and Granziera, Cristina and Graziani, Mara and Cuadra, Meritxell Bach},
   year={2023},
   month=apr 
}

@misc{molchanova2024preprint,
      title={Structural-Based Uncertainty in Deep Learning Across Anatomical Scales: Analysis in White Matter Lesion Segmentation}, 
      author={Nataliia Molchanova and Vatsal Raina and Andrey Malinin and Francesco La Rosa and Adrien Depeursinge and Mark Gales and Cristina Granziera and Henning Muller and Mara Graziani and Meritxell Bach Cuadra},
      year={2024},
      eprint={2311.08931},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.08931}, 
}
```
