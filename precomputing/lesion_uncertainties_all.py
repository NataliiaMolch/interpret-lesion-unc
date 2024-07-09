import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
import sys
from scipy import special

sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from utils.lesions_extraction import get_lesion_types_masks
from utils.transforms import get_cc_mask, process_probs
from utils.metrics import IoU_metric, IoU_adjusted_metric
from uq.vox_measures import ensemble_uncertainties_classification
from utils.logger import save_options
from uq.les_measures import lesions_uncertainty
from data_processing.datasets import NpzDataset

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# number of samples
parser.add_argument('--n_samples', type=int, required=True)
# data
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory with *_pred.npz files')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
parser.add_argument('--probs', action='set_true', default=False)
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
parser.add_argument('--det_threshold', type=float, default=0.1,
                    help='threshold for the intersection over union. see lesion_extraction.py')
parser.add_argument('--det_method', type=str, default='iou_adj',
                    help='method to classify lesions as tp, fp, us. see lesion_extraction.py')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where uncertainties will be saved')
# tuned hyperparameters
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)



def parse_thresholds(thresh_filename):
    with open(thresh_filename, 'r') as f:
        data = f.read().split('\n')
    return [float(d.split('\t')[1]) for d in data[:-1]]


def main(args):
    np.random.seed(0)

    os.makedirs(args.path_save, exist_ok=True)
    save_options(args, os.path.join(args.path_save, "les_uncs_options.txt"))

    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))

    lesions_uncs = []
    fn_count = []
    print(f"Using {args.n_samples} samples for uncertainty quantification.")
    
    for i_f, fn in enumerate(filenames):
        print(fn)
        # extract data from npz
        data = npz_dataset[i_f]
        if args.probs:
            mems_prob = data['pred_probs'][:args.n_samples]
        else:
            mems_prob = data['pred_logits']
            # softmax probabilities for all models in ensemble
            mems_prob = np.stack([
                special.softmax(s / args.temperature, axis=0)[args.class_num]
                for s in mems_prob[:args.n_samples]
            ], axis=0)
        assert len(mems_prob.shape) == 4
        # labeled ground truth mask
        gt_bin = data['targets']
        gt_lab = get_cc_mask(gt_bin)
        # binary mask from the softmax probabilities of an ensemble model
        ens_pred_seg = process_probs(
            prob_map=np.mean(mems_prob, axis=0),
            threshold=args.proba_threshold,
            l_min=args.l_min
        )
        # labeled masks from the softmax probabilities each model in ensemble
        mems_seg_lab = [
            process_probs(
                prob_map=ep,
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
            for ep in mems_prob
        ]
        mems_seg_lab = np.stack(
            [get_cc_mask(ep) for ep in mems_seg_lab],
            axis=0
        )
        # compute uncertainty maps
        vox_uncs = ensemble_uncertainties_classification(
            np.concatenate(
                (np.expand_dims(mems_prob, axis=-1),
                 np.expand_dims(1. - mems_prob, axis=-1)),
                axis=-1
            )
        )
        # labeled masks of TPL, USL, FPL
        lesion_masks: dict = get_lesion_types_masks(
            y_pred=ens_pred_seg,
            y=gt_bin,
            method=args.det_method,
            threshold=args.det_threshold,
            n_jobs=args.n_jobs
        )
        del ens_pred_seg, gt_bin

        # uncertainty evaluation loop
        for les_type, les_mask in lesion_masks.items():
            if les_type != 'FNL':
                les_list: list = lesions_uncertainty(
                    ens_seg_lab=les_mask.astype(float),
                    vox_uncs=vox_uncs,
                    mems_seg_lab=mems_seg_lab,
                    mems_prob=mems_prob,
                    gt_lab=gt_lab,
                    n_jobs=args.n_jobs
                )

                for i_l, el in enumerate(les_list):
                    les_list[i_l]['filename'] = fn
                    les_list[i_l]['lesion type'] = les_type

                lesions_uncs.extend(les_list)
                fn_count.append({'filename': fn,
                                 "FNl-count": len(np.unique(lesion_masks['FNL'])) - 1})

        # saving the data
        pd.DataFrame(lesions_uncs).to_csv(
            os.path.join(args.path_save, f"les_uncs_{args.n_samples}s_{args.set_name}.csv")
        )
        pd.DataFrame(fn_count).to_csv(
            os.path.join(args.path_save, f"FNL-count_{args.n_samples}s_{args.set_name}.csv")
        )


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
