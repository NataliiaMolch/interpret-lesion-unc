import argparse
import os
from functools import partial
import seaborn as sns;

sns.set_theme()
import pandas as pd
import sys;
from pathlib import Path

sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from joblib import Parallel, delayed
import numpy as np
from data_processing.datasets import NpzDataset
from utils.transforms import process_probs
from utils.metrics import voxel_scale_metric, lesion_detection_metric, model_calibration_metrics
from utils.logger import save_options
import logging
from scipy import special

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# number of samples
parser.add_argument('--n_samples', type=int, required=True)
# data
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to predictions')
parser.add_argument('--path_save', type=str, required=True)
parser.add_argument('--set_name', type=str, required=True)
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
parser.add_argument('--probs', action='store_true', default=False)
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# metrics parameters
parser.add_argument('--ndsc_r', type=float, default=2e-5,
                    help='iou|iou_adj threshold for a lesion to be considered tpl')
parser.add_argument('--det_method', type=str, default='iou_adj', help='non-zero|iou|iou_adj')
parser.add_argument('--det_threshold', type=float, default=0.1,
                    help='iou|iou_adj threshold for a lesion to be considered tpl')
parser.add_argument('--n_bins', type=int, default=30, help='for calibration errors')
# tuned hyperparameters
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)


def get_metric(i_file, args_local, npz_ds):
    # parse data
    data = npz_ds[i_file]
    gt_bin, bm = data['targets'], data['brain_mask']
    # softmax probabilities for all models in ensemble
    if args_local.probs:
        mems_prob = data['pred_probs'][:args_local.n_samples]
    else:
        mems_prob = data['pred_logits'][:args_local.n_samples]
        mems_prob = np.stack([
            special.softmax(s / args.temperature, axis=0)[args_local.class_num]
            for s in mems_prob
        ], axis=0)
    assert len(mems_prob.shape) == 4
    # binary mask from the softmax probabilities of an ensemble model
    ens_pred_seg = process_probs(
        prob_map=np.mean(mems_prob, axis=0),
        threshold=args_local.proba_threshold,
        l_min=args_local.l_min
    )
    # evaluate and save the results to `row` dictionary
    row = {'filename': data['filename']}
    row.update(
        voxel_scale_metric(
            y_pred=ens_pred_seg[bm == 1],
            y=gt_bin[bm == 1],
            r=args_local.ndsc_r
        )
    )
    row.update(
        lesion_detection_metric(
            y_pred=ens_pred_seg,
            y=gt_bin,
            method=args_local.det_method,
            threshold=args_local.det_threshold
        )
    )
    row.update(
        model_calibration_metrics(
            y_pred_probs=np.mean(mems_prob, axis=0)[bm == 1],
            y_pred=ens_pred_seg[bm == 1],
            y=gt_bin[bm == 1],
            n_bins=args_local.n_bins
        )
    )
    return row


def main(args):
    npz_dataset = NpzDataset(pred_path=args.path_pred)

    logging.info("Started evaluation")

    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        process = partial(
            get_metric,
            args_local=args,
            npz_ds=npz_dataset
        )
        res_df = parallel_backend(
            delayed(process)(i_file=i)
            for i in range(len(npz_dataset))
        )
    pd.DataFrame(res_df).to_csv(
        os.path.join(
            args.path_save,
            f"metrics_{args.n_samples}s_{args.set_name}.csv"
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.path_save, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    save_options(
        args=args,
        filepath=os.path.join(
            args.path_save,
            f"{args.set_name}_testing_options.txt"
        )
    )

    main(args)
