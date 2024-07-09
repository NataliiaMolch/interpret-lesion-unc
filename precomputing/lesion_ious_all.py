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
from data_processing.datasets import NpzDataset


parser = argparse.ArgumentParser(description='Get all command line arguments.')
# data
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory with *_pred.npz files')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
parser.add_argument('--npz_instance', type=str, default='pred_logits',
                    help='pred_logits|pred_probs')
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
parser.add_argument('--det_threshold', type=float, default=0.1,
                    help='threshold for the intersection over union. see lesion_extraction.py')
parser.add_argument('--det_method', type=str, default='iou_adj',
                    help='method to classify lesions as tp, fp, us. see lesion_extraction.py')
parser.add_argument('--n_samples', type=int, default=10,
                    help='the minimum size of the connected components')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where uncertainties will be saved')
# tuned hyperparameters
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)



def single_lesion_iou(y_pred_multi: np.ndarray, cc_label: float, y_multi):
    """ IoU with the GT for a single lesion
    """

    def get_max_ious_ccmasks_ens():
        max_iou = 0.0
        for intersec_label in np.unique(cc_mask * y_multi):
            if intersec_label != 0.0:
                lesion_m = (y_multi == intersec_label).astype(float)
                iou = IoU_metric(cc_mask, lesion_m)['IoU']
                if iou > max_iou:
                    max_iou = iou
        return max_iou

    cc_mask = (y_pred_multi == cc_label).astype(y_pred_multi.dtype)

    return {
        'maxIoU': get_max_ious_ccmasks_ens(),
        'IoUadj': IoU_adjusted_metric(cc_mask, y_pred=(y_pred_multi > 0.0).astype(float), y_multi=y_multi)['IoUadj'],
        'IoU': IoU_metric(y_pred=cc_mask, y=(y_multi > 0.0).astype(float))['IoU'],
        'lesion label': cc_label
    }


def lesion_ious(y_pred_multi: np.ndarray, y_multi, n_jobs: int = None):
    """ Parallel evaluation of all
    """
    from joblib import Parallel, delayed
    from functools import partial

    cc_labels = np.unique(y_pred_multi)
    cc_labels = cc_labels[cc_labels != 0.0]
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        process = partial(single_lesion_iou,
                          y_pred_multi=y_pred_multi,
                          y_multi=y_multi)
        les_ious_list = parallel_backend(delayed(process)(
            cc_label=cc_label
        ) for cc_label in cc_labels)  # returns lists of dictionaries, but need dictionary of lists

    return les_ious_list


def main(args):
    np.random.seed(0)

    os.makedirs(args.path_save, exist_ok=True)

    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))

    lesions_ious = []
    ''' Evaluation loop '''
    for i_f, fn in enumerate(filenames):
        print(fn)
        # prepare the data
        data = npz_dataset[i_f]
        
        if args.npz_instance == "pred_probs":
            mems_prob = data['pred_probs'][:args.n_samples]
            assert len(mems_prob.shape) == 4
            ens_pred_seg = process_probs(
                prob_map=np.mean(mems_prob, axis=0),
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
        elif args.npz_instance == "pred_logits":
            mems_prob = data['pred_logits']
            
            # obtain the predicted labeled lesion maps
            mems_prob = np.stack([
                special.softmax(s / args.temperature, axis=0)[args.class_num]
                for s in mems_prob[:args.n_samples]
            ], axis=0)
            assert len(mems_prob.shape) == 4
            ens_pred_seg = process_probs(
                prob_map=np.mean(mems_prob, axis=0),
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
        else:
            raise NotImplementedError(args.npz_instance)
    
        
        y = data['targets']
        y_multi = get_cc_mask(y)

        lesion_masks: dict = get_lesion_types_masks(y_pred=ens_pred_seg, 
                                                    y=y,
                                                    method=args.det_method,
                                                    threshold=args.det_threshold,
                                                    n_jobs=args.n_jobs)
        del ens_pred_seg
        
        for les_type, les_mask in lesion_masks.items():
            if les_type != 'FNL':
                les_list = lesion_ious(y_pred_multi=les_mask.astype(float),
                                       y_multi=y_multi, n_jobs=args.n_jobs)
                for i_l, el in enumerate(les_list):
                    les_list[i_l]['filename'] = fn
                    les_list[i_l]['lesion type'] = les_type
                lesions_ious.extend(les_list)

        pd.DataFrame(lesions_ious).to_csv(os.path.join(args.path_save, f"les_ious_{args.set_name}.csv"))

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
