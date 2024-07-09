import logging
import os
import torch
from utils.transforms import binarize_mask
from monai.networks.utils import one_hot
from monai.data import write_nifti
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, data_loader, activation, metrics: list, device, inferer, prob_threshold: float, n_classes: int,
                 save_path: str, set_name: str, save_pred: bool = False,
                 include_background: bool = False, to_onehot_y: bool = True, postprocessing=None,
                 inputs_key: str = "inputs", targets_key: str = "targets"):
        self.data_loader = data_loader
        self.activation = activation
        self.metrics = metrics
        self.inferer = inferer
        self.postprocessing = postprocessing
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.device = device
        self.threshold = prob_threshold
        self.include_background = include_background
        self.n_classes = n_classes
        self.to_onehot_y = to_onehot_y
        self.save_pred = save_pred

        if self.metrics:
            os.makedirs(os.path.join(save_path, 'evaluation'), exist_ok=True)
            self.res_filepath = os.path.join(save_path, 'evaluation', f'{set_name}_metrics.csv')

        if save_pred:
            self.save_path_pred = os.path.join(save_path, f"predictions_{set_name}")
            os.makedirs(self.save_path_pred, exist_ok=True)

        if postprocessing is None:
            self.postprocessing = lambda x: x

    def __call__(self, network, *args, **kwargs):
        if isinstance(network, list):
            for n in network:
                n.eval()
        else:
            network.eval()
        metrics_list = []
        filenames = []
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets = data[self.inputs_key].to(self.device), data[self.targets_key].to(self.device)
                if isinstance(network, list):
                    outputs = [self.inferer(inputs=inputs, network=n) for n in network]
                    outputs = [self.activation(o) for o in outputs]
                    outputs = torch.stack(outputs, dim=0).mean(dim=0)
                else:
                    outputs = self.inferer(inputs=inputs, network=network)  # [1, 2, H, W, D]
                    outputs = self.activation(outputs)  # [1, 2, H, W, D]

                if self.to_onehot_y:
                    targets = one_hot(targets, num_classes=self.n_classes)

                outputs = outputs.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets = targets.squeeze(0).cpu().numpy()  # [2, H, W, D]

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                if self.metrics:
                    for c in range(0, self.n_classes):
                        outputs_bin[c] = self.postprocessing(outputs_bin[c])
                    metrics_val = dict()
                    for c in range(0 if self.include_background else 1, self.n_classes):
                        for metric_func in self.metrics:
                            metrics_row = metric_func(y_pred=outputs_bin[c], y=targets[c], check=True)
                            for key in metrics_row:
                                metrics_val['%s_%d' % (key, c)] = metrics_row[key]

                    metrics_list += [metrics_val]

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                filenames += [filename]

                logging.info(filename)

                if self.metrics:
                    pd.DataFrame(metrics_list, index=filenames).to_csv(self.res_filepath)

                if self.save_pred:
                    affine = data['targets_meta_dict']['affine'][0]
                    spatial_shape = data['targets_meta_dict']['spatial_shape'][0]

                    for c in range(0 if self.include_background else 1, self.n_classes):
                        ''' Save binary mask '''
                        new_filepath = os.path.join(self.save_path_pred,
                                                    filename.split('.')[0] + f'_pred_class_{c}.nii.gz')
                        write_nifti(outputs_bin[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        ''' Save probability mask '''
                        new_filepath = os.path.join(self.save_path_pred,
                                                    filename.split('.')[0] + f'_pred_prob_class_{c}.nii.gz')
                        write_nifti(outputs[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        new_filepath = os.path.join(self.save_path_pred,
                                                    filename.split('.')[0] + f'_target_class_{c}.nii.gz')
                        write_nifti(targets[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

        return metrics_list, filenames


class PureEvaluator:
    def __init__(self, data_loader, metrics: list, prob_threshold: float, class_number: int,
                 postprocessing=None):
        """ When the predictions have already been made and saved and are returned by the dataloader
        Works for binary classification only.
        """
        self.data_loader = data_loader
        self.metrics = metrics
        self.postprocessing = postprocessing
        self.outputs_key = "outputs"
        self.targets_key = "targets"
        self.cl_key = "targets_cl"
        self.wml_key = "targets_wml"
        self.threshold = prob_threshold
        self.class_num = class_number

        if postprocessing is None:
            self.postprocessing = lambda x: x

        self.metrics_list = []
        self.filenames = []

    def __call__(self, *args, **kwargs):
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                outputs, targets = data[self.outputs_key], data[self.targets_key]

                outputs = outputs.squeeze(0).numpy()  # [H, W, D]
                targets = targets.squeeze(0).numpy()

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                outputs_bin = self.postprocessing(outputs_bin)

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                self.filenames += [filename]

                try:
                    metrics_val = dict()
                    for metric_func in self.metrics:
                        metrics_row = metric_func(y_pred=outputs_bin, y=targets, check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, self.class_num)] = metrics_row[key]

                    self.metrics_list += [metrics_val]

                    logging.info(filename)
                except Exception as e:
                    logging.warn(f"Exception caught on {filename}: {e}. Subject excluded from evaluation.")
                    self.filenames.remove(filename)

        return self.metrics_list, self.filenames


class PredictorNpzEnsemble:
    def __init__(self, data_loader, activation, device, inferer, class_num: int, n_classes: int,
                 save_path: str, set_name: str, temperature: float = 1,
                 inputs_key: str = "inputs", targets_key: str = "targets", bm_key:str = "brain_mask"):
        """
        Save probability predictions for all models in the ensemble / single model in to npz files.
        :param data_loader:
        :param activation:
        :param device:
        :param inferer:
        :param prob_threshold:
        :param temperature: temperature scaling parameter, applied only if activation is not None
        :param class_num:
        :param save_path:
        :param set_name:
        :param save_pred:
        :param include_background:
        :param postprocessing:
        :param inputs_key:
        :param targets_key:
        """
        self.temperature = temperature
        self.data_loader = data_loader
        self.activation = activation
        self.inferer = inferer
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.bm_key = bm_key
        self.device = device
        self.class_num = class_num
        self.n_classes = n_classes
        self.save_path_pred = os.path.join(save_path, f"predictions_{set_name}_npz")
        os.makedirs(self.save_path_pred, exist_ok=True)

    def __call__(self, network: list, *args, **kwargs):
        for n in network:
            n.eval()
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets, brain_mask = data[self.inputs_key].to(self.device), \
                                              data[self.targets_key].to(self.device), \
                                              data[self.bm_key].squeeze(0).squeeze(0).numpy()
                outputs: list = [self.inferer(inputs=inputs, network=n) for n in network]
                if self.activation is not None:
                    outputs: list = [self.activation(o / self.temperature) for o in outputs]     # list of [1, 2, H, W, D]
                    outputs: list = [o.squeeze(0).cpu().numpy()[self.class_num] for o in outputs]  # [H, W, D]
                else:
                    outputs: list = [o.squeeze(0).cpu().numpy() for o in outputs] # [2, H, W, D]
                outputs: np.ndarray = np.stack(outputs, axis=0)

                targets = one_hot(targets, num_classes=self.n_classes)
                targets = targets.squeeze(0).cpu().numpy()[self.class_num]  # [H, W, D]

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                logging.info(filename)

                to_save = {
                    'shape': targets.shape, 'output_shape': outputs.shape,
                    'brain_location': np.where(brain_mask == 1),
                    'affine': data['targets_meta_dict']['affine'][0],
                    'targets': targets[brain_mask == 1],
                    'pred_logits' if self.activation is None else 'pred_probs': outputs[np.broadcast_to(brain_mask, outputs.shape) == 1]
                }

                new_filename = filename.split('.')[0] + '_pred.npz'
                np.savez_compressed(os.path.join(self.save_path_pred, new_filename), **to_save)


class PredictorNpzMCDP:
    def __init__(self, data_loader, activation, device, inferer, class_num: int, n_classes: int,
                 save_path: str, set_name: str, n_samples: int, temperature: float = 1.,
                 inputs_key: str = "inputs", targets_key: str = "targets", bm_key:str = "brain_mask"):
        """
        Save probability predictions for several samples generated by MCDP model.
        :param data_loader:
        :param activation:
        :param device:
        :param inferer:
        :param prob_threshold:
        :param class_num:
        :param save_path:
        :param set_name:
        :param save_pred:
        :param include_background:
        :param postprocessing:
        :param inputs_key:
        :param targets_key:
        """
        self.temperature = temperature
        self.data_loader = data_loader
        self.activation = activation
        self.inferer = inferer
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.bm_key = bm_key
        self.device = device
        self.class_num = class_num
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.save_path_pred = os.path.join(save_path, f"predictions_{set_name}_npz")
        os.makedirs(self.save_path_pred, exist_ok=True)

    def __call__(self, network, *args, **kwargs):
        def enable_dropout(model):
            """ Function to enable the dropout layers during test-time """
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        network.eval()
        enable_dropout(network)
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets, brain_mask = data[self.inputs_key].to(self.device), \
                                              data[self.targets_key].to(self.device), \
                                              data[self.bm_key].squeeze(0).squeeze(0).numpy()
                outputs = [self.inferer(inputs=inputs, network=network) for _ in range(self.n_samples)]  # [1, 2, H, W, D]
                if self.activation is not None:
                    outputs: list = [self.activation(o / self.temperature) for o in outputs]  # list of [1, 2, H, W, D]
                    outputs = [o.squeeze(0).cpu().numpy()[self.class_num] for o in outputs]  # [H, W, D]
                else:
                    outputs = [o.squeeze(0).cpu().numpy() for o in outputs]  # [2, H, W, D]
                outputs: np.ndarray = np.stack(outputs, axis=0)

                targets = one_hot(targets, num_classes=self.n_classes)
                targets = targets.squeeze(0).cpu().numpy()[self.class_num]  # [H, W, D]

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                logging.info(filename)

                to_save = {
                    'shape': targets.shape, 'output_shape': outputs.shape,
                    'brain_location': np.where(brain_mask == 1),
                    'affine': data['targets_meta_dict']['affine'][0],
                    'targets': targets[brain_mask == 1],
                    'pred_logits' if self.activation is None else 'pred_probs': outputs[
                        np.broadcast_to(brain_mask, outputs.shape) == 1]
                }

                new_filename = filename.split('.')[0] + '_pred.npz'
                np.savez_compressed(os.path.join(self.save_path_pred, new_filename), **to_save)