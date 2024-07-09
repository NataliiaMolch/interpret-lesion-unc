"""
Compute retention curves and save voxel scale uncertainty maps
"""

import argparse
import os
import re

from joblib import Parallel, delayed
import json
from functools import partial
import numpy as np
import sys
from pathlib import Path

sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from uq.les_measures import uncs_measures_names
from uq.retention_curves import lesion_scale_rc
from utils.metrics import bootstrap_stand_err
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;

sns.set_theme()
import pickle
from cycler import cycler

custom_cycler = cycler(color=['blue', 'brown', 'coral', 'orange', 'purple', 'm',
                              'green', 'cyan', 'olive', 'greenyellow', 'y',
                              'red', 'plum', 'darkorchid'])

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--metric', type=str, default='LPPV',
                    help='Metric based on which the RC is built: LF1|LPPV')
# data
parser.add_argument('--path_lesion_uncs', type=str, required=True,
                    help='Specify the path to the csv file with lesion uncertainties')
parser.add_argument('--path_fn_count', type=str, required=True,
                    help='Specify the path to the csv file with scans fn counts')
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where retention \
                    curves will be saved')


def create_ideal_uncs(les_uncs_sub):
    ideal = []
    for lt in les_uncs_sub['lesion type']:
        if lt == 'FPL':
            ideal.append({'lesion type': 'FPL',
                          'ideal unc.': 2})
        elif lt == 'USL':
            ideal.append({'lesion type': 'USL',
                          'ideal unc.': 1})
        elif lt == 'TPL':
            ideal.append({'lesion type': 'TPL',
                          'ideal unc.': 0})
        else:
            raise ValueError(lt)
    return pd.DataFrame(ideal)


def create_random_uncs(les_uncs_sub):
    random = les_uncs_sub.copy()
    random['random unc.'] = np.random.permutation(len(les_uncs_sub))
    return random[['random unc.', 'lesion type']]


def main(args):
    np.random.seed(0)
    os.makedirs(args.path_save, exist_ok=True)

    # load the data
    lu_df: pd.DataFrame = pd.read_csv(args.path_lesion_uncs, index_col=0)
    fnc_df: pd.DataFrame = pd.read_csv(args.path_fn_count, index_col=0)

    # define containers
    fracs_retained = np.linspace(0.0, 1.0, 400)
    uncs_measures = [
        c for c in lu_df.columns
        if c not in ['lesion label', 'lesion type', 'filename', 'maxIoU', 'IoUadj', 'IoU']
    ]
    results = dict(
        zip(uncs_measures, [{'ret_curve': [], 'AUC': []} for _ in uncs_measures]))
    for k in ['ideal', 'random']:
        results[k] = {'ret_curve': [], 'AUC': []}
    filenames = list(set(lu_df['filename']))

    ''' Evaluation loop '''
    for filename in filenames:
        les_uncs_sub = lu_df[lu_df['filename'] == filename]
        n_fn = fnc_df[fnc_df['filename'] == filename]
        assert len(n_fn) == 1, filename
        n_fn = n_fn['FNl-count'].item()

        with Parallel(n_jobs=args.n_jobs) as parallel_backend:
            process = partial(lesion_scale_rc,
                              les_uncs_sub=les_uncs_sub,
                              n_fn=n_fn,
                              fracs_retained=fracs_retained,
                              metric_name=args.metric)
            parallel_results = parallel_backend(delayed(process)(unc_measure=un) for un in uncs_measures)
            for um, res in zip(uncs_measures, parallel_results):
                results[um]['ret_curve'].append(res[1])
                results[um]['AUC'].append(res[0])

        auc_rc, y_rc = lesion_scale_rc(les_uncs_sub=create_ideal_uncs(les_uncs_sub),
                                       unc_measure='ideal unc.',
                                       n_fn=n_fn, metric_name=args.metric,
                                       fracs_retained=fracs_retained)
        results["ideal"]['ret_curve'].append(y_rc)
        results["ideal"]['AUC'].append(auc_rc)
        auc_rc, y_rc = lesion_scale_rc(les_uncs_sub=create_random_uncs(les_uncs_sub),
                                       unc_measure='random unc.',
                                       n_fn=n_fn, metric_name=args.metric,
                                       fracs_retained=fracs_retained)
        results["random"]['ret_curve'].append(y_rc)
        results["random"]['AUC'].append(auc_rc)

    results['filenames'] = filenames

    # save and viz
    try:
        with open(os.path.join(args.path_save, f"results_{args.set_name}.json"), 'w') as f:
            json.dump(results, f)
    except:
        with open(os.path.join(args.path_save, f"results_{args.set_name}.pickle"), 'wb') as f:
            pickle.dump(results, f)

    summary_df = []
    aucs = []
    ste = []
    for um in uncs_measures + ["ideal", "random"]:
        summary_df.append(np.array(results[um]['ret_curve']).mean(axis=0))
        aucs.append(np.mean(results[um]['AUC']))
        ste.append(bootstrap_stand_err(results[um]['AUC']))
    summary_df = pd.DataFrame(summary_df, columns=fracs_retained, index=uncs_measures + ["ideal", "random"])
    summary_df['Avg. AUC'] = aucs
    summary_df['St.err. AUC'] = ste
    summary_df.to_csv(os.path.join(args.path_save, f'summary_{args.set_name}.csv'))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_prop_cycle(custom_cycler)
    for um in uncs_measures:
        ax.plot(fracs_retained, np.array(results[um]['ret_curve']).mean(axis=0), label=uncs_measures_names[um])
    ax.plot(fracs_retained, np.array(results["ideal"]['ret_curve']).mean(axis=0), linestyle='--', label="Ideal",
            color='black')
    ax.plot(fracs_retained, np.array(results["random"]['ret_curve']).mean(axis=0), linestyle='--', label="Random",
            color='gray')

    ax.set_xlabel("Fraction of retained lesions")
    ax.set_ylabel(args.metric)
    ax.set_title(args.set_name)
    ax.set_xlim([0.0, 1.01])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(args.path_save, f'plot_{args.set_name}.jpg'), dpi=300)
    plt.clf()


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
