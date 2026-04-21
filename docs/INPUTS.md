# Input Contract

## Images

Use an nnU-Net v2 dataset folder with `imagesTs` or `imagesTr` channel files:

```text
case_001_0000.nii.gz
case_001_0001.nii.gz
```

Channel names are read from `dataset.json` when available. Otherwise channels are named `modality_0000`, `modality_0001`, etc.

## Predictions

Provide one folder per ensemble member/fold:

```text
predictions/
  fold_0/case_001.npz
  fold_1/case_001.npz
```

Supported probability formats:

- nnU-Net `.npz` with key `probabilities`
- `.npz` with key `softmax`, `pred_probs`, or `arr_0`
- NIfTI lesion-class probability maps

## Ground Truth

Ground-truth masks are optional. If provided through `--labels-subdir`, the package computes lesion type, IoU, maxIoU, and IoUadj. If omitted, regression and counterfactual analysis can still be run without IoUadj features.

## Thresholds

The ensemble threshold and all member thresholds default to `0.5`. Member thresholds can be supplied as:

```json
[0.55, 0.55, 0.5, 0.45, 0.55]
```

or as a CSV with a `threshold` column.
