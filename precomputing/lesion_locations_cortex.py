import os
import re
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import sys
from scipy import special
sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from utils.transforms import process_probs, get_cc_mask
import ants
from utils.logger import save_options
from data_processing.datasets import NpzDataset
from functools import partial
from joblib import Parallel, delayed

gray_matter_labels = [
    3,  # Left cerebral cortex
    42, # Right cerebral cortex
    8,  # Left cerebellum cortex
    47, # Right cerebellum cortex
    10, # Left thalamus
    49, # Right thalamus
    11, # Left caudate
    50, # Right caudate
    12, # Left putamen
    51, # Right putamen
    13, # Left pallidum
    52, # Right pallidum
    17, # Left hippocampus
    53, # Right hippocampus
    18, # Left amygdala
    54, # Right amygdala
    26, # Left accumbens area
    58  # Right accumbens area
] 


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--n_jobs', type=int, default=1, help='joblib parameter')
# data
parser.add_argument('--path_synthseg_results', type=str, required=True,
                    help='')
parser.add_argument('--path_pred', type=str, required=True,
                    help='path to the directory with npz predictions')
parser.add_argument('--prefix_synthseg', type=str,
                    default='UNIT1_synthseg.nii.gz')
parser.add_argument('--npz_instance', type=str, default='pred_logits',
                    help='pred_logits|targets')
# binary lesion masks processing
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--l_min', type=int, default=2,
                    help='the minimum size of the connected components')
parser.add_argument('--n_samples', type=int, default=10,
                    help='the minimum size of the connected components')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where uncertainties will be saved')

def compute_subject_features(idx, filename, _npz_dataset):
    data = _npz_dataset[idx]
    h = re.search(r"sub-[\d]+_ses-[\d]+", str(filename))[0]
    
    if os.path.exists(os.path.join(args.path_synthseg_results, f"{h}_{args.prefix_synthseg}")):
        cortex_seg = nib.load(os.path.join(args.path_synthseg_results, f"{h}_{args.prefix_synthseg}"))
        cortex_seg = np.isin(cortex_seg.get_fdata(), test_elements=gray_matter_labels).astype(float)
        
        if args.npz_instance == "pred_probs":
            mems_prob = data['pred_probs'][:args.n_samples]

            assert len(mems_prob.shape) == 4
            ens_pred_seg = process_probs(
                prob_map=np.mean(mems_prob, axis=0),
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
            lesion_seg = get_cc_mask(ens_pred_seg)
            
            del mems_prob, ens_pred_seg
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
            lesion_seg = get_cc_mask(ens_pred_seg)
            
            del mems_prob, ens_pred_seg
        elif args.npz_instance == "targets":
            lesion_seg = get_cc_mask(data['targets'])
        else:
            raise NotImplementedError(args.npz_instance)
        
        lesion_labels = np.unique(lesion_seg)
        lesion_labels = lesion_labels[lesion_labels != 0]
        
        if len(lesion_labels) > 0:
            res = list()
            for lesion_label in lesion_labels:
                lesion = (lesion_seg == lesion_label).astype(float)
                row = {
                    'filename': filename, 
                    'lesion label': lesion_label, 
                    'GM overlap': np.sum(cortex_seg * lesion) / np.sum(lesion)} 
                res.append(row)
            return pd.DataFrame(res)
        return None
    else:
        return os.path.join(args.path_synthseg_results, f"{h}_{args.prefix_synthseg}")
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    os.makedirs(args.path_save, exist_ok=True)
    save_options(args, os.path.join(args.path_save, "les_omics_options.txt"))
    
    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))
        
    with Parallel(n_jobs=args.n_jobs) as parallel:
        results = parallel(
            delayed(compute_subject_features)(idx=i_f, filename=fn, _npz_dataset=npz_dataset)
            for i_f, fn in enumerate(filenames)
            )
    
    failed_cases = [_ for _ in results if isinstance(_, str)]
        
    with open(os.path.join(args.path_save, f"no_synthseg_{args.set_name}_{args.npz_instance}.txt"), 'w') as f:
        for fc in failed_cases:
            f.write(f"{fc}\n")
            
    results = [ _ for _ in results if _ is not None and not isinstance(_, str)]
    
    if len(results) > 0:
        pd.concat(results, axis=0).to_csv(os.path.join(args.path_save,
                                            f"cortex_overlap_features_{args.set_name}_{args.npz_instance}.csv"))
    else:
        pd.DataFrame([]).to_csv(os.path.join(args.path_save,
                                            f"cortex_overlap_features_{args.set_name}_{args.npz_instance}.csv"))