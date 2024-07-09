"""
Tuning of probability threshold, temperature scaling parameter, minimum lesion size.
"""

import argparse
import os
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set_theme()
import pandas as pd
import sys;
from pathlib import Path
sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from joblib import Parallel, delayed
import numpy as np
from utils.transforms import remove_connected_components
from utils.metrics import voxel_scale_metric, lesion_detection_metric
from utils.logger import save_options
from data_processing.datasets import NpzDataset, get_BIDS_hash
from scipy import ndimage, special
import logging
from sklearn.model_selection import KFold, StratifiedKFold

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--metrics', type=str, nargs='+', default='nDSC,DSC,F1-score (iou_adj)')
parser.add_argument('--quality_metric', type=str, default='F1-score (iou_adj)')
# data
parser.add_argument('--probs', action='set_true', default=False)
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to train set predictions')
parser.add_argument('--stats_csv', type=str, required=True,
                    help='Specify the path to csv with stats')
parser.add_argument('--path_save', type=str, required=True)
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
# parallel computation
parser.add_argument('--n_jobs', type=int, default=10,
                    help='Number of parallel workers for F1 score computation')
parser.add_argument('--ndsc_r', type=float, default=2e-5,
                    help='iou|iou_adj threshold for a lesion to be considered tpl')
parser.add_argument('--det_method', type=str, default='iou_adj', help='non-zero|iou|iou_adj')
parser.add_argument('--det_threshold', type=float, default=0.1,
                    help='iou|iou_adj threshold for a lesion to be considered tpl')
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--n_bins', type=int, default=15, help='for calibration errors')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling parameter')


def create_folds(args_local, pred_filepaths):
    """
    Create folds stratified by lesion load
    :param args_local: 
    :return: 
    """
    np.random.seed(0)
    stats = pd.read_csv(args_local.stats_csv, index_col=0)
    stats = stats[stats['set name'] == 'train'][['subject id', 'CL load']]
    filenames = pd.DataFrame(list(map(str, pred_filepaths)), columns=['filepath'])
    filenames['subject id'] = get_BIDS_hash(filenames['filepath'])
    stats = pd.merge(filenames, stats, on='subject id')

    percent = [0.2, 0.4, 0.6, 0.8]
    quantiles = np.quantile(stats['CL load'], percent)
    strata = np.zeros_like(stats['CL load'])
    limits = np.sort(np.append(quantiles, [stats['CL load'].min(), stats['CL load'].max()]))
    for i_m, _ in enumerate(limits[1:]):
        strata[(limits[i_m - 1] < stats['CL load']) * (stats['CL load'] <= limits[i_m])] = i_m

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42), stats['filepath'].tolist(), strata


def get_metric(y_pred, y, y_multi, brain_mask, n_gt_lesions, proba_threshold, filename,
               args_local):
    seg = y_pred.copy()
    # get a binary mask
    seg[seg >= proba_threshold] = 1.
    seg[seg < proba_threshold] = 0.
    # remove small lesions
    seg = remove_connected_components(seg, args_local.l_min)
    # evaluate and save the results to `row` dictionary
    row = dict()
    row.update(lesion_detection_metric(y_pred=seg, y=y, check=True,
                                       method=args_local.det_method,
                                       threshold=args_local.det_threshold,
                                       y_multi=y_multi, n_gt_labels=n_gt_lesions))
    row.update(voxel_scale_metric(y_pred=seg[brain_mask == 1], y=y[brain_mask == 1],
                                  check=True, r=args_local.ndsc_r))
    row.update({
        'proba_threshold': proba_threshold,
        'l_min': args_local.l_min,
        'filename': filename
    })
    return row


def evaluate_test(data, p, args_local):
    # create binary maps
    if args_local.probs:
        seg = np.mean(data['pred_probs'], axis=0)
    else:
        seg = data['pred_logits']
        seg = [special.softmax(s / args_local.temperature, axis=0)[args_local.class_num] for s in seg]
        seg = np.mean(np.stack(seg, axis=0), axis=0)
    assert len(seg.shape) == 3
    targets_multi, n_gt_lesions = ndimage.label(data['targets'],
                                                structure=ndimage.generate_binary_structure(rank=3, connectivity=2))
    return get_metric(y_pred=seg, y=data['targets'],
                      brain_mask=data['brain_mask'],
                      y_multi=targets_multi, n_gt_lesions=n_gt_lesions,
                      filename=data['filename'],
                      args_local=args_local, proba_threshold=p)


def generate_summary(res_df, metrics, q_metric):
    summary_df = []
    for p in SEARCH_GRID['proba_threshold']:
        row = dict()
        row['proba_threshold'] = p
        for i_m, metric in enumerate(metrics):
            row[f"avg. {metric}"] = res_df[
                (res_df['proba_threshold'] == p)
                ][metric].mean()
            row[f"st.d. {metric}"] = res_df[
                (res_df['proba_threshold'] == p)
                ][metric].std()
        summary_df.append(row)
    summary_df = pd.DataFrame(summary_df)

    best_param = summary_df.sort_values(by=f'avg. {q_metric}', ascending=False)[
        ['proba_threshold', f'avg. {q_metric}', f'st.d. {q_metric}']
    ].head(1)['proba_threshold'].item()
    return summary_df, best_param


def main(args, search_grid):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    npz_dataset = NpzDataset(pred_path=args.path_pred)
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    ''' Evaluation loop '''
    logging.info("Started evaluation")
    cv, filepaths, strata = create_folds(args, npz_dataset.pred_filepaths)
    for i_fold, (train_index, test_index) in enumerate(cv.split(filepaths, strata)):
        logging.info(f"Fold {i_fold}")
        res_df = []
        ''' Evaluate best params on train '''
        for idx in train_index:
            data = npz_dataset[idx]
            # create gt labeled maps
            targets_multi, n_gt_lesions = ndimage.label(data['targets'], structure=struct_el)
            # create binary maps
            if args.probs:
                seg = np.mean(data['pred_probs'], axis=0)
            else:
                seg = data['pred_logits']
                seg = [special.softmax(s / args.temperature, axis=0)[args.class_num] for s in seg]
                seg = np.mean(np.stack(seg, axis=0), axis=0)
            assert len(seg.shape) == 3
            # define evaluation function
            process = partial(get_metric,
                              y_pred=seg, y=data['targets'],
                              brain_mask=data['brain_mask'],
                              y_multi=targets_multi, n_gt_lesions=n_gt_lesions,
                              filename=data['filename'],
                              args_local=args)
            res_df.extend(
                Parallel(n_jobs=args.n_jobs)(delayed(process)(proba_threshold=p)
                                 for p in search_grid['proba_threshold'])
            )
            pd.DataFrame(res_df).to_csv(os.path.join(args.path_save, f"fold{i_fold}_raw.csv"))
        res_df = pd.DataFrame(res_df)
        res_df.to_csv(os.path.join(args.path_save, f"fold{i_fold}_raw.csv"))

        # res_df = pd.read_csv(os.path.join(args.path_save, f"fold{i_fold}_raw.csv"), index_col=0)
        # test_filepaths = [filepaths[idx] for idx in test_index]
        # assert set(res_df['filepath'].unique()).intersection(set(test_filepaths)) == set()

        ''' Figure best parameters based on the trade-off between ECE bin and quality metric'''
        summary_df, best_p = generate_summary(res_df, args.metrics[0].split(','), q_metric=args.quality_metric)
        summary_df.to_csv(os.path.join(args.path_save, f"fold{i_fold}_summary.csv"))

        ''' Run evaluation on test fold '''
        process = partial(evaluate_test, p=best_p, args_local=args)
        test_res_df = Parallel(n_jobs=args.n_jobs)(delayed(process)(data=npz_dataset[idx]) for idx in test_index)
        test_res_df = pd.DataFrame(test_res_df)
        test_res_df['index'] = test_index
        test_res_df = test_res_df.set_index('index')
        test_res_df.to_csv(os.path.join(args.path_save, f"fold{i_fold}_raw_test.csv"))


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.path_save, exist_ok=True)
    save_options(args=args,
                 filepath=os.path.join(args.path_save, "tuning_options.txt"))
    SEARCH_GRID = {
        'proba_threshold': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    }
    main(args, SEARCH_GRID)
