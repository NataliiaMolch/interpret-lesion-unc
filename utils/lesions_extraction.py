from scipy import ndimage
import numpy as np
from joblib import Parallel, delayed
from .metrics import IoU_metric, IoU_adjusted_metric
from functools import partial


def decide_pred_lesion_type(cc_pred, y_pred_multi, y_multi, method, threshold=None):
    """
    For method non_zero:
        * TPL, if exists an overlap with the GT
        * FPL, if there is no overlap with the GT
    For the method iou:
        * TPL, if IoU(cc_pred, cc_max_y) >= `threshold`
        * USL, if 0 < IoU(cc_pred, cc_max_y) < `threshold`
        * FPL, if there is no overlap with the GT
    For the method iou_adj:
        * TPL, if IoU_adj(cc_pred, y_pred, y) >= threshold
        * USL, if 0 < IoU_adj(cc_pred, y_pred, y) < threshold
        * FPL, if there is no overlap with the GT
    :param cc_pred: predicted connected component
    :param y_pred_multi: predicted binary lesion mask for non_zero and iou_adj, predicted multi_mask for iou
    :param y_multi: gt binary lesion mask
    :param threshold: threshold for IoU / IoU_adj for a lesion to be considered detected
    :param method: non_zero, iou or iou_adj
    :return: type of lesion
    """
    if method == 'non_zero':
        if np.sum(cc_pred * y_multi) > 0.0:
            return 'TPL'
        else:
            return 'FPL'
    elif method == 'iou':
        max_iou = 0.0
        for l_gt in np.unique(y_multi * cc_pred):
            if l_gt != 0.0:
                iou: float = IoU_metric(cc_pred, (y_multi == l_gt).astype(y_multi.dtype))['IoU']
                if iou > max_iou:
                    max_iou = iou
        if max_iou >= threshold:
            return 'TPL'
        elif 0 < max_iou < threshold:
            return 'USL'
        else:
            return 'FPL'
    elif method == 'iou_adj':
        iou_adj = IoU_adjusted_metric(cc_pred=cc_pred,
                                      y_pred=(y_pred_multi > 0.0).astype(float),
                                      y_multi=y_multi)['IoUadj']
        if iou_adj >= threshold:
            return 'TPL'
        elif 0 < iou_adj < threshold:
            return 'USL'
        else:
            return 'FPL'
    else:
        raise ValueError(f"Method {method} not supported.")


def decide_gt_lesion_type(cc_gt, y_pred):
    """
    For the lesions on the ground truth we don't need such precise information (correctly detected / undersegmented),
    this information will be carried by the predicted lesions. We want to recover info that is not in the ground truth.
    For all methods:
        * DL (detected lesion), if exists an overlap with the prediction
        * FNL (false negative lesion), if there is no overlap with the prediction
    # For method non_zero:
    #     * DL (detected lesion), if exists an overlap with the prediction
    #     * FNL (false negative lesion), if there is no overlap with the prediction
    # For the method iou:
    #     * DL, if IoU(cc_gt, cc_max_y_pred) >= `threshold`
    #     * USL_gt, if 0 < IoU(cc_gt, cc_max_y_pred) < `threshold`
    #     * FNL, if there is no overlap with the predicted mask
    # For the method iou_adj:
    #     * DL, if IoU_adj(cc_gt, y_pred, y) >= threshold
    #     * USL_gt, if 0 < IoU_adj(cc_gt, y_pred, y) < threshold
    #     * FNL, if there is no overlap with the predicted mask
    :param cc_gt: ground truth connected component
    :param y_pred: predicted binary lesion mask
    :param method: non_zero, iou or iou_adj
    :return: type of lesion
    """
    if np.sum(cc_gt * y_pred) == 0:
        return 'FNL'
    else:
        return 'DL'


def get_lesion_types_masks(y_pred, y, method, threshold=None, n_jobs: int = None):
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    y_multi, n_gt_labels = ndimage.label(y, structure=struct_el)
    y_pred_multi, n_pred_labels = ndimage.label(y_pred, structure=struct_el)

    process_predicted = partial(decide_pred_lesion_type,
                                y_pred_multi=y_pred_multi,
                                y_multi=y_multi,
                                method=method,
                                threshold=threshold)
    process_gt = partial(decide_gt_lesion_type, y_pred=y_pred)

    with Parallel(n_jobs=n_jobs) as parallel:
        fnl = parallel(delayed(process_gt)(cc_gt=(y_multi == l).astype(y_multi.dtype))
                       for l in range(1, n_gt_labels + 1))
        fnl = [l for l, t in zip(range(1, n_gt_labels + 1), fnl) if t == 'FNL']

        tpl_fpl_usl = parallel(delayed(process_predicted)(cc_pred=(y_pred_multi == l).astype(y_pred_multi.dtype))
                               for l in range(1, n_pred_labels + 1))
        tpl = [l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'TPL']
        fpl = [l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'FPL']
        usl = [l for l, t in zip(range(1, n_pred_labels + 1), tpl_fpl_usl) if t == 'USL']

    return {
        'TPL': y_pred_multi * np.isin(y_pred_multi, tpl),
        'FPL': y_pred_multi * np.isin(y_pred_multi, fpl),
        'FNL': y_multi * np.isin(y_multi, fnl),
        'USL': y_pred_multi * np.isin(y_pred_multi, usl)
    }
