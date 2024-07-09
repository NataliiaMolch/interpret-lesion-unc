import os
import re
import copy
from joblib import Parallel, delayed
from functools import partial
from scipy import special
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from utils.transforms import process_probs, get_cc_mask
import ants
import SimpleITK as sitk
import radiomics
from utils.logger import save_options
from data_processing.datasets import NpzDataset



class Atlas:
    def __init__(self, atlas_name, atlas_root, atlas_left_fn, atlas_right_fn, atlas_xml_fn):
        self.atlas_name, self.atlas_root, self.atlas_left_fn, self.atlas_right_fn, self.atlas_xml_fn = \
            atlas_name, atlas_root, atlas_left_fn, atlas_right_fn, atlas_xml_fn
        self.labels = self.get_labels()
        self.mass_centers = self.get_mass_centers()

    def get_labels(self):
        from xml.etree import ElementTree as ET
        tree = ET.parse(os.path.join(self.atlas_root, self.atlas_xml_fn))
        root = tree.getroot()
        labels = dict()
        for child in root.find('data').iter('label'):
            print(child.attrib, child.text)
            labels[int(child.attrib['index']) + 1] = self.atlas_name + ' ' + child.text
        return labels

    def compute_cm(self, binary_mask):
        return np.asarray(np.where(binary_mask == 1)).mean(axis=1)

    def get_mass_centers(self):
        mass_centers = dict()
        for fn, side in zip([self.atlas_left_fn, self.atlas_right_fn], ['L', 'R']):
            mask = nib.load(os.path.join(self.atlas_root, fn)).get_fdata()
            for label, label_name in self.labels.items():
                mass_centers[label_name + ' ' + side] = self.compute_cm((mask == label).astype(float))
        return mass_centers

    def compute_distances(self, binary_structure):
        assert set(np.unique(binary_structure).tolist()) - {0, 1} == set(), np.unique(binary_structure)
        structure_cm = self.compute_cm(binary_structure)
        distances = dict()
        for label_name, label_cm in self.mass_centers.items():
            distances[label_name] = np.linalg.norm(label_cm - structure_cm)
        return distances


def register_to_mni(mask: np.ndarray, original_affine: np.ndarray, transform_path, fixed_img):
    img = ants.from_nibabel(nib.Nifti1Image(mask, affine=original_affine))
    return ants.apply_transforms(fixed=fixed_img, moving=img, transformlist=[transform_path],
                                 interpolator='nearestNeighbor')


parser = argparse.ArgumentParser(description='Get all command line arguments.')
# data
parser.add_argument('--path_flair', type=str, required=True,
                    help='Specify the path to the directory with flair skull stripped images')
parser.add_argument('--path_mp2rage', type=str, required=True,
                    help='Specify the path to the directory with mp2rage skull stripped images')
parser.add_argument('--prefix_flair', type=str, default='FLAIR.nii.gz',
                    help='Specify the path to the directory with flair images')
parser.add_argument('--prefix_mp2rage', type=str, default='UNIT1.nii.gz',
                    help='Specify the path to the directory with mp2rage images')
parser.add_argument('--path_pred', type=str, required=True,
                    help='path to the directory with npz predictions')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--l_min', type=int, default=2,
                    help='the minimum size of the connected components')
parser.add_argument('--n_samples', type=int, default=10,
                    help='the minimum size of the connected components')
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where uncertainties will be saved')
parser.add_argument('--npz_instance', type=str, default='pred_probs',
                    help='pred_logits|targets')


def normalise(sitk_img):
        non_zero_img = sitk.GetArrayFromImage(sitk_img)
        non_zero_img = non_zero_img[non_zero_img != 0.0]
        mu = np.mean(non_zero_img)
        sigma = np.mean(non_zero_img)
        return (sitk_img - mu) / sigma


if __name__ == '__main__':
    args = parser.parse_args()
    
    os.makedirs(args.path_save, exist_ok=True)
    save_options(args, os.path.join(args.path_save, "les_omics_options.txt"))
    
    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))
    
    fe = radiomics.featureextractor.RadiomicsFeatureExtractor(verbose=False)

    results = []
    
    for i_f, fn in enumerate(filenames):
        data = npz_dataset[i_f]

        if args.npz_instance == "pred_probs":
            mems_prob = data['pred_probs'][:args.n_samples]

            assert len(mems_prob.shape) == 4
            ens_pred_seg = process_probs(
                prob_map=np.mean(mems_prob, axis=0),
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
            ens_pred_seg_lab = get_cc_mask(ens_pred_seg)
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
            ens_pred_seg_lab = get_cc_mask(ens_pred_seg)
        else:
            raise NotImplementedError(args.npz_instance)
        

        gt_bin = data['targets']
        
        h = re.search(r"sub-[\d]+_ses-[\d]+", str(fn))[0]
        
        # register lesion mask to the mni space
        nib.save(nib.Nifti1Image(ens_pred_seg_lab, affine=data['affine']),
                 os.path.join(args.path_save, 'tmp_img.nii.gz'))

        flair_img = sitk.ReadImage(os.path.join(args.path_flair,
                                                f"{h}_{args.prefix_flair}"),
                                   sitk.sitkFloat32)
        mp2rage_img = sitk.ReadImage(os.path.join(args.path_mp2rage,
                                                  f"{h}_{args.prefix_mp2rage}"),
                                     sitk.sitkFloat32)
        mask_img = sitk.ReadImage(os.path.join(args.path_save, 'tmp_img.nii.gz'),
                                  sitk.sitkUInt16)
        flair_img, mp2rage_img = normalise(flair_img), normalise(mp2rage_img)

        res = []
        for ll in np.unique(ens_pred_seg_lab):
            if ll != 0.0:
                row = {'filename': fn, 'lesion label': ll}
                for name, img in zip(['flair', 'mp2rage'], [flair_img, mp2rage_img]):
                    try:
                        features = dict(fe.execute(img, mask_img == ll))
                    except ValueError as e:
                        print(e)
                        features = dict()
                    for k, v in features.items():
                        if re.search('original', k) is not None and re.search("diagnostics", k) is None:
                            row[f"{name} {k}"] = float(v)
                res.append(row)

        results.extend(res)

        pd.DataFrame(results).to_csv(
            os.path.join(args.path_save,
                        f"radiomics_{args.set_name}.csv"))
