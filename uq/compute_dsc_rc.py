"""
Compute retention curves and save voxel scale uncertainty maps
"""

import argparse
import os
import pickle
from pathlib import Path
import numpy as np
from scipy import special
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from utils.transforms import process_probs
from uq.vox_measures import uncs_measures, uncs_measures_names, ensemble_uncertainties_classification
from uq.retention_curves import voxel_scale_rc
from utils.metrics import bootstrap_stand_err
from data_processing.datasets import NpzDataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;
sns.set_theme()


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--metric', type=str, default='DSC',
                    help='Metric based on which the threshold is tuned')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
parser.add_argument('--ndsc_r', type=float, default=2e-5,
                    help='iou|iou_adj threshold for a lesion to be considered tpl')
# data
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--prefix_pred', default='cl_binary_pred.npz')
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where retention \
                    curves will be saved')
# tuned hyperparameters
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)


def main(args):
    np.random.seed(0)

    os.makedirs(args.path_save, exist_ok=True)

    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))

    fracs_retained = np.linspace(0.0, 1.0, 400)

    results = dict(
        zip(uncs_measures, [{'ret_curve': [], 'AUC': []} for _ in uncs_measures]))
    for k in ['ideal', 'random']:
        results[k] = {'ret_curve': [], 'AUC': []}

    ''' Evaluation loop '''
    for i_f, fn in enumerate(filenames):
        print(fn)
        # prepare the data
        data = npz_dataset[i_f]
        ens_pred_probs, y, bm = data['pred_logits'], data['targets'], data['brain_mask']
        ens_pred_probs = np.stack([
            special.softmax(s / args.temperature, axis=0)[args.class_num] for s in ens_pred_probs
        ], axis=0)
        assert len(ens_pred_probs.shape) == 4
        ens_pred_seg = process_probs(prob_map=np.mean(ens_pred_probs, axis=0),
                                     threshold=args.proba_threshold, l_min=args.l_min)
        # compute uncertainty maps
        uncs_maps = ensemble_uncertainties_classification(np.concatenate(
            (np.expand_dims(ens_pred_probs, axis=-1),
             np.expand_dims(1. - ens_pred_probs, axis=-1)),
            axis=-1))
        # compute DSC-RC
        for um in uncs_measures:
            auc_rc, y_rc = voxel_scale_rc(y_pred=ens_pred_seg[bm == 1], y=y[bm == 1],
                                          uncertainties=uncs_maps[um][bm == 1],
                                          fracs_retained=fracs_retained,
                                          n_jobs=args.n_jobs, metric_name=args.metric)
            results[um]['ret_curve'].append(y_rc)
            results[um]['AUC'].append(auc_rc)

        # compute ideal
        ideal_uncs_map = np.zeros_like(y[bm == 1])
        ideal_uncs_map[(ens_pred_seg[bm == 1] == 1) * (y[bm == 1] == 0)] = 1  # fp
        ideal_uncs_map[(ens_pred_seg[bm == 1] == 0) * (y[bm == 1] == 1)] = 1  # fn
        auc_rc, y_rc = voxel_scale_rc(y_pred=ens_pred_seg[bm == 1], y=y[bm == 1],
                                      uncertainties=ideal_uncs_map,
                                      fracs_retained=fracs_retained,
                                      n_jobs=args.n_jobs, metric_name=args.metric)
        results["ideal"]['ret_curve'].append(y_rc)
        results["ideal"]['AUC'].append(auc_rc)
        # compute random
        random_uncs_map = np.random.permutation(len(y[bm == 1]))
        auc_rc, y_rc = voxel_scale_rc(y_pred=ens_pred_seg[bm == 1], y=y[bm == 1],
                                      uncertainties=random_uncs_map,
                                      fracs_retained=fracs_retained,
                                      n_jobs=args.n_jobs, metric_name=args.metric)
        results["random"]['ret_curve'].append(y_rc)
        results["random"]['AUC'].append(auc_rc)

    results['filenames'] = filenames
    # save results
    with open(os.path.join(args.path_save, f"results_{args.set_name}.pickle"), 'wb') as f:
        pickle.dump(results, f)
    # generate summary and save
    summary_df = []
    aucs = []
    ste = []
    for um in uncs_measures:
        summary_df.append(np.array(results[um]['ret_curve']).mean(axis=0))
        aucs.append(np.mean(results[um]['AUC']))
        ste.append(bootstrap_stand_err(results[um]['AUC']))
    summary_df = pd.DataFrame(summary_df, columns=fracs_retained, index=uncs_measures)
    summary_df['Avg. AUC'] = aucs
    summary_df['St.err. AUC'] = ste
    summary_df.to_csv(os.path.join(args.path_save, f'summary_{args.set_name}.csv'))
    # build plots
    for um in uncs_measures:
        plt.plot(fracs_retained, np.array(results[um]['ret_curve']).mean(axis=0), label=uncs_measures_names[um])
    plt.plot(fracs_retained, np.array(results["ideal"]['ret_curve']).mean(axis=0), linestyle='--', label="Ideal",
             color='black')
    plt.plot(fracs_retained, np.array(results["random"]['ret_curve']).mean(axis=0), linestyle='--', label="Random",
             color='gray')
    plt.xlabel("Fraction of retained voxels")
    plt.ylabel(args.metric)
    plt.title(args.set_name)
    plt.xlim([0.0, 1.01])
    plt.legend()
    plt.savefig(os.path.join(args.path_save, f'plot_{args.set_name}.jpg'), dpi=300)
    plt.clf()


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
