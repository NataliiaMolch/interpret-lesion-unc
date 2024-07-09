import numpy as np
from functools import partial
from collections import Counter
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from utils.metrics import voxel_scale_metric
from scipy.interpolate import interp1d


def voxel_scale_rc(y_pred: np.ndarray, y: np.ndarray, uncertainties: np.ndarray, fracs_retained: np.ndarray, metric_name: str,
                   n_jobs: int = None):
    """
    Compute error retention curve values and nDSC-AUC on voxel-scale.
    :param fracs_retained:
    :param y:
    :param y_pred:
    :param uncertainties:
    :param n_jobs: number of parallel processes
    :return: tuple with nDSC-AAC and nDSC values of the error retention curve
    """

    def compute_metric(frac_, preds_, gts_, N_, metric_name_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate([preds_[:pos], gts_[pos:]])
        return voxel_scale_metric(y=gts_, y_pred=curr_preds)[metric_name_]

    ordering = uncertainties.argsort()
    gts = y[ordering].copy()
    preds = y_pred[ordering].copy()

    process = partial(compute_metric, preds_=preds, gts_=gts, N_=len(gts), metric_name_=metric_name)
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        dsc_norm_scores = parallel_backend(delayed(process)(frac) for frac in fracs_retained)

    return metrics.auc(fracs_retained, np.asarray(dsc_norm_scores)), dsc_norm_scores


def lesion_scale_rc(les_uncs_sub: pd.DataFrame, unc_measure: str, n_fn: int, fracs_retained: np.ndarray, metric_name='LF1'):
    """
    Compute error retention curve values and F1les/PPV-AUC on lesion-scale.
    """
    def compute_metric(lesion_types):
        counter = Counter(lesion_types)
        if metric_name == 'LF1':
            if counter['TPL'] + 0.5 * (counter['FPL'] + counter['USL'] + n_fn) == 0.0:
                return 0
            return counter['TPL'] / (counter['TPL'] + 0.5 * (counter['FPL'] + counter['USL'] + n_fn))
        elif metric_name == 'LPPV':
            if counter['TPL'] + counter['FPL'] + counter['USL'] == 0.0:
                return 1.0
            return counter['TPL'] / (counter['TPL'] + counter['USL'] + counter['FPL'])

    unc_values = np.asarray(les_uncs_sub[unc_measure])
    ltype_values = np.asarray(les_uncs_sub['lesion type'])
    ordering = unc_values.argsort()
    ltype_values = ltype_values[ordering][::-1]
    metric_values = [compute_metric(ltype_values)]
    for i_l, lesion_type in enumerate(ltype_values):
        # like in the previous version of experiments (ISBI, ISMRM)
        # if lesion_type in ['FPL', 'USL']:
        #     ltype_values[i_l] = 'TNL'
        # with redefinition of USL
        if lesion_type == 'FPL':
            ltype_values[i_l] = 'TNL'
        elif lesion_type == 'USL':
            ltype_values[i_l] = 'TPL'
        metric_values.append(compute_metric(ltype_values))

    n_lesions = len(ltype_values)
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=metric_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    f1_interp = spline_interpolator(fracs_retained)
    return metrics.auc(fracs_retained, np.asarray(f1_interp)), f1_interp


def patient_scale_rc(pat_uncs_df: pd.DataFrame, metrics_df:pd.DataFrame, unc_measure: str, metric_name: str, merge_colum: str, replace_with: float = 1.0):
    df = pat_uncs_df[[unc_measure, merge_colum]].copy()
    df = df.merge(metrics_df[[metric_name, merge_colum]], on=merge_colum).sort_values(unc_measure, ascending=False, ignore_index=True)
    rc = [df[metric_name].mean()]
    for idx in df.index:
        df.loc[idx, metric_name] = replace_with
        rc += [df[metric_name].mean()]
    fracs_retained = np.linspace(0, 1, len(rc))
    return metrics.auc(fracs_retained, np.asarray(rc)[::-1]), np.asarray(rc)[::-1]