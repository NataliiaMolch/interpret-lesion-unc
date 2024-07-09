import numpy as np
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Lambdad, Compose, LoadImaged)
from pathlib import Path
import os
from glob import glob
import re
import logging


def check_dataset(filepaths_list, prefixes):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of
    flair image filepath and gts filepath.
    Parameters:
        - flair_filepaths (list) - list of paths to flair images
        - gts_filepaths (list) - list of paths to lesion masks of flair images
        - args (argparse.ArgumentParser) - system arguments, should contain `flair_prefix` and `gts_prefix`
    """
    if len(filepaths_list) > 1:
        filepaths_ref = filepaths_list[0]
        for filepaths, prefix in zip(filepaths_list[1:], prefixes[1:]):
            assert len(filepaths_ref) == len(
                filepaths), f"Found {len(filepaths_ref)} ref files and {len(filepaths)} with prefix: {prefix}"
        for sub_filepaths in list(map(list, zip(*filepaths_list))):
            filepath_ref = sub_filepaths[0]
            prefix_ref = prefixes[0]
            for filepath, prefix in zip(sub_filepaths[1:], prefixes[1:]):
                if os.path.basename(filepath_ref)[:-len(prefix_ref)] != os.path.basename(filepath)[:-len(prefix)]:
                    raise ValueError(f"{filepath_ref} and {filepath} do not match")


def get_filepaths(paths, prefixes):
    if isinstance(paths, list):
        return [sorted(glob(os.path.join(pa, f"*{pre}")), key=lambda i: int(re.sub('\D', '', i)))
                for pa, pre in zip(paths, prefixes)]
    elif isinstance(paths, str):
        return sorted(glob(os.path.join(paths, f"*{prefixes}")),
                      key=lambda i: int(re.sub('\D', '', i)))
    else:
        return None


def get_BIDS_hash(filepaths: list):
    return [re.search('sub-[\d]+_ses-[\d]+', fp)[0] for fp in filepaths]


class NiftiDataset(CacheDataset):
    def __init__(self, input_paths: list, input_prefixes: list, input_names: list,
                 target_path: str, target_prefix: list, transforms,
                 num_workers=0, cache_rate=0.5, bm_path: str = None, bm_prefix: str = None):
        """
        :param input_paths: list of paths to directories where input MR images are stored
        :param target_path: path to the directory where target binary masks are stored
        :param balmask_path: path to the directory where balancing masks are stored
        :param input_prefixes: list of name endings of files in corresponding `input_paths` directories
        :param target_prefix: name endings of files in `target_path` directory
        :param balmask_prefix: name ending of files in `balmask_path` directory
        :param input_names: names of modalities corresponding to files in `input_paths` directories
        :param num_workers: number of parallel processes to preprocess the data
        :param cache_rate: fraction of images that are preprocessed and cached
        """
        if not len(input_paths) == len(input_prefixes) == len(input_names):
            raise ValueError("Input paths, prefixes and names should be of the same length.")

        def get_nonzero_targets(filepaths):
            import nibabel as nib
            return [i_f for i_f, file in enumerate(filepaths)
                    if nib.load(file).get_fdata().sum() > 0.0]

        self.input_filepaths = get_filepaths(input_paths, input_prefixes)
        self.target_filepaths = get_filepaths(target_path, target_prefix)

        idx_stay = get_nonzero_targets(self.target_filepaths)

        self.target_filepaths = [self.target_filepaths[i_f] for i_f in idx_stay]
        for i, filepaths_i in enumerate(self.input_filepaths):
            self.input_filepaths[i] = [filename for i_f, filename in enumerate(filepaths_i)
                                       if i_f in idx_stay]

        to_check_filepaths = self.input_filepaths + [self.target_filepaths]
        to_check_prefix = input_prefixes + [target_prefix]
        modality_names = input_names + ["targets"]

        # if need to load brain masks
        if bm_path is not None and bm_prefix is not None:
            self.bm_filepaths = get_filepaths(bm_path, bm_prefix)
            self.bm_filepaths = [self.bm_filepaths[i_f] for i_f in idx_stay]
            to_check_filepaths += [self.bm_filepaths]
            to_check_prefix += [bm_prefix]
            modality_names += ["brain_mask"]

        check_dataset(to_check_filepaths, to_check_prefix)

        logging.info(f"Initializing the dataset. Number of subjects {len(self.target_filepaths)}")

        self.files = [dict(zip(modality_names, files)) for files in list(zip(*to_check_filepaths))]

        super().__init__(data=self.files, transform=transforms,
                         cache_rate=cache_rate, num_workers=num_workers, hash_as_key=True)

    def __len__(self):
        return len(self.files)


class NpzDataset:
    def __init__(self, pred_path:str, pred_prefix: str = 'pred.npz'):
        """

        :param pred_path: path to the directory with npz files
        :param pred_prefix: prefix of the npz files
        """
        self.pred_filepaths: list = sorted(list(Path(pred_path).glob(f"*{pred_prefix}")))

        logging.info(f"Initializing the dataset. Number of subjects {len(self.pred_filepaths)}")

    def __len__(self):
        return len(self.pred_filepaths)

    def __getitem__(self, idx):
        data: np.lib.npyio.NpzFile = np.load(self.pred_filepaths[idx])
        data_dict = {
            'shape': data['shape'],
            'affine': data['affine'],
            'filename': os.path.basename(self.pred_filepaths[idx])
        }

        # parse brain mask
        bm = np.zeros(data['shape'])
        bm[data['brain_location'][0], data['brain_location'][1], data['brain_location'][2]] = 1
        data_dict['brain_mask'] = bm

        # parse outputs
        output_name = 'pred_logits' if 'pred_logits' in data.files else 'pred_probs'
        output = np.zeros(shape=data['output_shape'])
        output[np.broadcast_to(bm, data['output_shape']) == 1] = data[output_name]
        data_dict[output_name] = output
        # data_dict[output_name] = data[output_name]

        # parse targets
        target = np.zeros(data['shape'])
        target[bm == 1] = data['targets']
        data_dict['targets'] = target

        return data_dict

