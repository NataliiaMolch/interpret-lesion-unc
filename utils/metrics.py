"""
All the metrics assume similar input y_pred and y:
* both y_pred and y are binary masks
* numpy.ndarray
* of the same dimensionality: [H, W, D]
* function returns a dictionary in a form dict(zip(metrics_names, metrics_values))

Update:
metrics that are computed separately for CL and WML lesions (while the segmentation was 1 classes)
should take as additional input `cl_mask: np.ndarray, wml_mask: np.ndarray` that satisfy same conditions as y_pred and y,
and have the same shape as y and y_pred
"""
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from scipy import ndimage


def bootstrap_stand_err(a, n_iter=1000, sample_frac=0.5):
    """ For an array `a` compute a bootstrap Ste"""
    np.random.seed(0)
    boot_means = []
    for _ in range(n_iter):
        boot_means.append(np.random.choice(a, size=round(len(a) * sample_frac)).mean())
    return np.std(boot_means)


#### HELPERS ###
def check_inputs(y_pred, y):
    def check_binary_mask(mask):
        unique = np.unique(mask)
        if np.sum(np.isin(unique, test_elements=[0.0, 1.0], invert=True)) != 0.0:
            return False
        return True

    instance = bool(isinstance(y_pred, np.ndarray) * isinstance(y, np.ndarray))

    binary_mask = bool(check_binary_mask(y_pred) * check_binary_mask(y))

    dimensionality = bool(y_pred.shape == y.shape)

    if not instance * binary_mask * dimensionality:
        raise ValueError(f"Inconsistent input to metric function. Failed in instance: {instance},"
                         f"binary mask: {binary_mask}, dimensionality: {dimensionality}.")


### NON-QUALITY METRICS ###

def IoU_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)
    return {'IoU': np.sum(y_pred * y) / np.sum(y_pred + y - y_pred * y)}


def IoU_adjusted_old_metric(cc_pred: np.ndarray, y_pred: np.ndarray = None, y: np.ndarray = None,
                        y_pred_multi: np.ndarray = None, y_multi: np.ndarray = None,
                        check: bool = False):
    if (y_pred is not None and y is not None) or (y_pred_multi is not None and y_multi is not None):
        if y_pred_multi is None and y_multi is None:
            if check:
                check_inputs(y_pred, y)
            struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
            y_pred_multi, n_les_pred = ndimage.label(y_pred - cc_pred, structure=struct_el)
            y_multi, n_les_gt = ndimage.label(y, structure=struct_el)
        # labels of the gt lesions that have an overlap with the predicted lesion
        K_prime_labels: list = np.unique(y_multi * cc_pred).tolist()
        K_prime_labels.remove(0)
        if K_prime_labels:
            K_prime = np.isin(y_multi, test_elements=K_prime_labels).astype(float)
            # labels of the predicted lesions that have an overlap with the K' lesions, except from the predicted lesion
            Q_labels: list = np.unique(K_prime * y_pred_multi).tolist()
            Q_labels.remove(0)
            Q = np.isin(y_pred_multi, test_elements=Q_labels)
            nominator = np.sum(cc_pred * K_prime)
            denominator = np.sum(cc_pred + K_prime - K_prime * Q > 0.0)
            return {'IoUadj': nominator / denominator}
        return {'IoUadj': 0.0}
    else:
        raise ValueError("Either `y_pred` and `y` or `y_pred_multi` and `y_multi` must be not none. "
                         f"Got `y_pred`: {type(y_pred)}, `y`: {type(y)}, `y_pred_multi`: {type(y_pred_multi)}, `y_multi`: {type(y_multi)}")


def IoU_adjusted_metric(cc_pred: np.ndarray, y_pred: np.ndarray = None, y: np.ndarray = None,
                        y_multi: np.ndarray = None, check: bool = False):
    if (y_pred is not None and y is not None) or (y_multi is not None and y_pred is not None):
        if y_multi is None:
            if check:
                check_inputs(y_pred, y)
            struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
            y_multi, n_les_gt = ndimage.label(y, structure=struct_el)
        # labels of the gt lesions that have an overlap with the predicted lesion
        K_prime_labels: list = np.unique(y_multi * cc_pred).tolist()
        K_prime_labels.remove(0)
        if K_prime_labels:
            K_prime_mask = np.isin(y_multi, test_elements=K_prime_labels).astype(float)
            # mask of intersection between the ground truth intersecting with the predicted lesion and other predicted lesions
            K_prime_intersec_Q_mask = (y_pred - cc_pred) * K_prime_mask
            # remove from K' this intersection
            K_prime_minus_Q_mask = K_prime_mask - K_prime_intersec_Q_mask
            nominator = np.sum(cc_pred * K_prime_mask)
            denominator = np.sum(cc_pred + K_prime_minus_Q_mask > 0.0)
            return {'IoUadj': nominator / denominator}
        return {'IoUadj': 0.0}
    else:
        raise ValueError("Either `y_pred` and `y` or `y_pred` and `y_multi` must be not none. "
                         f"Got `y_pred`: {type(y_pred)}, `y`: {type(y)}, `y_multi`: {type(y_multi)}")


### OVERALL SEGMENTATION QUALITY METRICS ###

def DSC_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        return {'DSC': 2 * (y_pred * y).sum() / (y_pred + y).sum()}
    return {'DSC': 1.0}


def nDSC_metric(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        scaling_factor = 1.0 if (np.sum(y) == 0 or len(y.flatten()) == np.sum(y)) else (1 - r) * np.sum(y) / (
                    r * (len(y.flatten()) - np.sum(y)))
        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        fp_scaled = scaling_factor * fp
        return {'nDSC': 2 * tp / (fp_scaled + 2 * tp + fn)}
    return {'nDSC': 1.0}


def voxel_scale_metric(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        scaling_factor = 1.0 if np.sum(y) == 0 else (1 - r) * np.sum(y) / (r * (len(y.flatten()) - np.sum(y)))

        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        tn = np.sum(y[y_pred == 0] == 0)

        fp_scaled = scaling_factor * fp

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fdr = fp / (fp + tp)
        return {'TPRvox': tpr, 'FPRvox': fpr, 'FDRvox': fdr, 'nDSC': 2 * tp / (fp_scaled + 2 * tp + fn),
                'DSC': 2 * (y_pred * y).sum() / (y_pred + y).sum()}
    return {'TPRvox': 1.0, 'FPRvox': 0.0, 'FDRvox': 0.0, 'nDSC': 1.0, 'DSC': 1.0}


def voxel_rates_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False):
    if check: check_inputs(y_pred, y)

    if np.sum(y_pred) + np.sum(y) > 0:
        tp = np.sum(y_pred[y == 1])
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        tn = np.sum(y[y_pred == 0] == 0)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fdr = fp / (fp + tp)
        return {'TPRvox': tpr, 'FPRvox': fpr, 'FDRvox': fdr}
    return {'TPRvox': 1.0, 'FPRvox': 0.0, 'FDRvox': 0.0}

### NEW DETECTION QUALITY METRICS ###

def lesion_detection_metric(y_pred: np.ndarray, y: np.ndarray, check: bool = False,
                            method: str = 'iou_adj', threshold: float = 0.25,
                            n_jobs: int = None,
                            y_multi: np.ndarray = None, n_gt_labels: int = None):
    if check: check_inputs(y_pred, y)
    from .lesions_extraction import decide_pred_lesion_type, decide_gt_lesion_type

    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    y_pred_multi, n_pred_labels = ndimage.label(y_pred, structure=struct_el)
    if y_multi is None or n_gt_labels is None:
        y_multi, n_gt_labels = ndimage.label(y, structure=struct_el)

    process_predicted = partial(decide_pred_lesion_type,
                                y_pred_multi=y_pred_multi,
                                y_multi=y_multi,
                                method=method,
                                threshold=threshold)
    process_gt = partial(decide_gt_lesion_type, y_pred=y_pred)

    with Parallel(n_jobs=n_jobs) as parallel:
        fnl = parallel(delayed(process_gt)(cc_gt=(y_multi == l).astype(y_multi.dtype))
                       for l in range(1, n_gt_labels + 1))
        fnl = len([l for l, t in zip(range(1, n_gt_labels + 1), fnl) if t == 'FNL'])

        tpl_fpl_usl = parallel(delayed(process_predicted)(cc_pred=(y_pred_multi == l).astype(y_pred_multi.dtype))
                               for l in range(1, n_pred_labels + 1))
        tpl = len([l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'TPL'])
        fpl = len([l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'FPL'])
        usl = len([l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'USL'])

    results = {
        'F1-score': 2. * tpl / (2 * tpl + fpl + usl + fnl) if 2 * tpl + fpl + usl + fnl > 0.0 else 1.0,
        'Recall': tpl / (tpl + fnl) if tpl + fnl > 0.0 else 1.0,
        'Precision': tpl / (tpl + fpl + usl) if tpl + fpl + usl > 0.0 else (1.0 if fnl == 0.0 else 0.0),
        'USL-rate': usl / (tpl + fpl + usl) if tpl + fpl + usl > 0.0 else (0.0 if fnl == 0.0 else 1.0),
        'USL-count': usl, 'FPL-count': fpl, 'FNL-count': fnl,
        '# gt lesions': n_gt_labels, '# pred lesions': n_pred_labels
    }

    results_m = dict()
    for k, v in results.items():
        results_m[f"{k} ({method})"] = v
    return results_m


# MODEL CALIBRATION

def model_calibration_metrics(y_pred_probs: np.ndarray, y_pred: np.ndarray,
                              y: np.ndarray, n_bins: int = 30, check: bool=False):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    if check: check_inputs(y_pred, y)
    results = dict()
    y_, y_pred_, y_pred_probs_ = y.copy().flatten(), \
                                 y_pred.copy().flatten(), \
                                 y_pred_probs.copy().flatten()

    # Compute the calibration error
    prob_true, prob_pred = calibration_curve(y_, y_pred_probs_, n_bins=n_bins)
    results['CE'] = np.mean(np.abs(prob_pred - prob_true))
    # Compute Brier score
    results['Brier score'] = brier_score_loss(y_, y_pred_probs_)
    # Compute calibration error
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_pred_probs_)
    bin_sums = np.bincount(binids, weights=y_pred_probs_, minlength=len(bins))
    bin_acc = np.bincount(binids, weights=(y_ == y_pred_).astype(float), minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    non_zero = bin_total != 0.0
    bin_sums, bin_acc, bin_total = bin_sums[non_zero] / bin_total[non_zero], \
                                   bin_acc[non_zero] / bin_total[non_zero], \
                                    bin_total[non_zero]
    # results['ECE'] = np.average(np.abs(bin_acc - bin_sums), weights=bin_total / bin_total.sum())
    # results['MCE'] = np.max(np.abs(bin_acc - bin_sums))
    # for positive class only
    bin_acc = np.bincount(binids, weights=(y_ == 1).astype(float), minlength=len(bins))[non_zero] / bin_total
    results['ECE bin'] = np.average(np.abs(bin_acc - bin_sums), weights=bin_total / bin_total.sum())
    results['MCE bin'] = np.max(np.abs(bin_acc - bin_sums))
    return results
