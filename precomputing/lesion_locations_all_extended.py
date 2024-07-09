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


class Atlas:
    def __init__(self, atlas_name, atlas_root, atlas_left_fn, atlas_right_fn, atlas_xml_fn):
        self.atlas_name, self.atlas_root, self.atlas_left_fn, self.atlas_right_fn, self.atlas_xml_fn = \
            atlas_name, atlas_root, atlas_left_fn, atlas_right_fn, atlas_xml_fn
        self. labels = self.get_labels()
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
        for _fn, side in zip([self.atlas_left_fn, self.atlas_right_fn], ['L', 'R']):
            mask = nib.load(os.path.join(self.atlas_root, _fn)).get_fdata()
            for label, label_name in self.labels.items():
                mass_centers[label_name + ' ' + side] = self.compute_cm((mask == label).astype(float))
        return mass_centers

    # def compute_distances(self, binary_structure):
    #     assert set(np.unique(binary_structure).tolist()) - {0, 1} == set(), np.unique(binary_structure)
    #     structure_cm = self.compute_cm(binary_structure)
    #     distances = dict()
    #     for label_name, label_cm in self.mass_centers.items():
    #         distances[f" Dist. to {label_name}"] = np.linalg.norm(label_cm - structure_cm)
    #     return distances
    
    # def compute_intersection(self, seg_lab):
    #     intersections = dict()
    #     for _fn, side in zip([self.atlas_left_fn, self.atlas_right_fn], ['L', 'R']):
    #         structure_mask = nib.load(os.path.join(self.atlas_root, _fn)).get_fdata()
    #         for structure_label, structure_name in self.labels.items():
    #             intersections['Intersec.' + structure_name + ' ' + side] = list()
    #             for lesion_label in seg_lab.unique():
    #                 intersections['Intersec.' + structure_name + ' ' + side].append(
    #                     np.sum((seg_lab == lesion_label).astype(float) * (structure_mask == structure_label))
    #                     )
    #     return intersections
    
    def compute_features(self, seg_lab):
        lesion_labels = np.unique(seg_lab)
        lesion_labels = lesion_labels[lesion_labels != 0]
        
        feature_names = list()
        
        # compute intersection matrix #lesions x #structures
        intersections = list()
        for _fn, side in zip([self.atlas_left_fn, self.atlas_right_fn], ['L', 'R']):
            structure_mask = nib.load(os.path.join(self.atlas_root, _fn)).get_fdata()
            for structure_label, structure_name in self.labels.items():
                structure_intersections = list()
                feature_names.append(structure_name + ' ' + side)
                for lesion_label in lesion_labels:
                    structure_intersections.append(
                        np.sum((seg_lab == lesion_label).astype(float) * (structure_mask == structure_label))
                        )
                intersections.append(structure_intersections)
        intersections = np.asarray(intersections).T
        assert intersections.shape == (len(lesion_labels), len(feature_names))
        
        # compute distances matrix #lesions x #structures
        distances = list()
        for lesion_label in lesion_labels:
            lesion_cm = self.compute_cm((seg_lab == lesion_label).astype(float))
            lesion_distances = list()
            for feat_name in feature_names:
                lesion_distances.append(np.linalg.norm(self.mass_centers[feat_name] - lesion_cm))
            distances.append(lesion_distances)
        distances = np.asarray(distances)
        assert intersections.shape == distances.shape
        
        # features computation and packing into a dataframe
        features = distances * (intersections == np.max(intersections, axis=1, keepdims=True))
        features = pd.DataFrame(features, columns=feature_names, index=lesion_labels)
        features['lesion label'] = lesion_labels

        distances = pd.DataFrame(distances, columns=['Dist. to ' + feat_name for feat_name in feature_names], index=lesion_labels)  
        intersections = pd.DataFrame(intersections, columns=['Intesec. with ' + feat_name for feat_name in feature_names], index=lesion_labels)
        
        return pd.concat([features, distances, intersections], axis=1)
        
        


def register_to_mni(mask: np.ndarray, original_affine: np.ndarray, transform_path, _fixed_img):
    img = ants.from_nibabel(nib.Nifti1Image(mask, affine=original_affine))
    return ants.apply_transforms(fixed=_fixed_img, moving=img, transformlist=[transform_path],
                                 interpolator='nearestNeighbor')


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--n_jobs', type=int, default=1, help='joblib parameter')
# data
parser.add_argument('--path_atlases', type=str, required=True,
                    help='Specify the path to the directory with atlases from the `atlases` list')
parser.add_argument('--path_pred', type=str, required=True,
                    help='path to the directory with npz predictions')
parser.add_argument('--path_transform', type=str, required=True,
                    help='path to the directory with *fwd.mat files to transform predictions to MNI space')
parser.add_argument('--prefix_transform', type=str,
                    default='toMNI152_affine_transform_fwd.mat')
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

def compute_subject_features(idx, filename, _npz_dataset, atlases: list):
    data = _npz_dataset[idx]
    h = re.search(r"sub-[\d]+_ses-[\d]+", str(filename))[0]
    
    if args.npz_instance == "pred_probs":
        mems_prob = data['pred_probs'][:args.n_samples]

        assert len(mems_prob.shape) == 4
        ens_pred_seg = process_probs(
            prob_map=np.mean(mems_prob, axis=0),
            threshold=args.proba_threshold,
            l_min=args.l_min
        )
        ens_pred_seg_lab = get_cc_mask(ens_pred_seg)
        
        seg_lab_mni = register_to_mni(
            mask=ens_pred_seg_lab,
            original_affine=data['affine'],
            transform_path=os.path.join(args.path_transform, f"{h}_{args.prefix_transform}"),
            _fixed_img=fixed_img
            ).numpy()
        del mems_prob, ens_pred_seg, ens_pred_seg_lab
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
        
        seg_lab_mni = register_to_mni(
            mask=ens_pred_seg_lab,
            original_affine=data['affine'],
            transform_path=os.path.join(args.path_transform, f"{h}_{args.prefix_transform}"),
            _fixed_img=fixed_img
            ).numpy()
        del mems_prob, ens_pred_seg, ens_pred_seg_lab
    elif args.npz_instance == "targets":
        gt_bin = data['targets']
        
        gt_lab = get_cc_mask(gt_bin)
        
        seg_lab_mni = register_to_mni(
            mask=gt_lab,
            original_affine=data['affine'],
            transform_path=os.path.join(args.path_transform, f"{h}_{args.prefix_transform}"),
            _fixed_img=fixed_img
            ).numpy()
    else:
        raise NotImplementedError(args.npz_instance)
    
    if len(np.unique(seg_lab_mni)) > 1:
        res = [atlas.compute_features(seg_lab=seg_lab_mni) for atlas in atlases]
        res = pd.concat(res, axis=1)
        res['filename'] = np.full(len(res), filename)
        
        return res
    return None
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    os.makedirs(args.path_save, exist_ok=True)
    save_options(args, os.path.join(args.path_save, "les_omics_options.txt"))
    
    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))
    
    results = list()
    
    # Read the atlas information
    HOAtlas = Atlas(atlas_name='HOC', atlas_root=args.path_atlases,
                    atlas_left_fn='HarvardOxford-cort-maxprob-thr0-1mm_L.nii.gz',
                    atlas_right_fn='HarvardOxford-cort-maxprob-thr0-1mm_R.nii.gz',
                    atlas_xml_fn='HarvardOxford-Cortical.xml')
    MNIAtlas = Atlas(atlas_name='MNI', atlas_root=args.path_atlases,
                     atlas_left_fn='MNI-maxprob-thr0-1mm_L.nii.gz',
                     atlas_right_fn='MNI-maxprob-thr0-1mm_R.nii.gz',
                     atlas_xml_fn='MNI.xml')
    MNIRefAtlas = Atlas(atlas_name='rMNI', atlas_root=args.path_atlases,
                     atlas_left_fn='MNI-refined-maxprob-thr0-1mm_L.nii.gz',
                     atlas_right_fn='MNI-refined-maxprob-thr0-1mm_R.nii.gz',
                     atlas_xml_fn='MNI-refined.xml')
    fixed_img = ants.image_read(os.path.join(args.path_atlases,
                                             'MNI152lin_T1_1mm_brain.nii.gz'))
    
    process = partial(compute_subject_features, _npz_dataset=npz_dataset, atlases=[MNIAtlas, HOAtlas, MNIRefAtlas])
    
    with Parallel(n_jobs=args.n_jobs) as parallel:
        results = parallel(
            delayed(process)(idx=i_f, filename=fn)
            for i_f, fn in enumerate(filenames)
            )
    
    results = [ _ for _ in results if _ is not None]
        
    if len(results) > 0:
        pd.concat(results, axis=0).to_csv(os.path.join(args.path_save,
                                            f"location_features_{args.set_name}_{args.npz_instance}.csv"))
    else:
        pd.DataFrame([]).to_csv(os.path.join(args.path_save,
                                            f"location_features_{args.set_name}_{args.npz_instance}.csv"))