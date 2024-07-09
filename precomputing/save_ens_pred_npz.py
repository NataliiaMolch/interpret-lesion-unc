import sys, os
from pathlib import Path

sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
import pandas as pd
from joblib import Parallel
from functools import partial
# random seed fix
import random
import numpy as np
# inner imports
from utils.options import get_predict_options
from data_processing.datasets import NiftiDataset
from model_evaluation.evaluator import PredictorNpzEnsemble, PredictorNpzMCDP
from utils.logger import save_options
from utils.metrics import *
from utils.models import get_model
from utils.transforms import remove_connected_components, get_val_transforms
# logging
import logging, sys

if __name__ == '__main__':
    ''' Parse and add model types '''
    args = get_predict_options(class_specific=True)
    ''' Log ans save dirs '''
    save_path = os.path.join(args.path_save, args.exp_name)
    os.makedirs(os.path.join(save_path, f"predictions_{args.eval_set_name}_npz"), exist_ok=True)
    save_options(args=args, filepath=os.path.join(save_path, f"predictions_{args.eval_set_name}_npz", "test_options.txt"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    ''' Fix random seeds '''
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ''' Get default device '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Using device: {device}")
    torch.multiprocessing.set_sharing_strategy('file_system')

    ''' Define data loaders '''
    val_transforms = get_val_transforms(input_keys=args.input_modalities, label_key="targets", binarize_keys=["brain_mask"]).set_random_state(
        seed=seed)
    if args.dataset == 'niftidataset':
        val_dataset = NiftiDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes,
                                   input_names=args.input_modalities,
                                   target_path=args.target_val_path, target_prefix=args.target_prefix,
                                   transforms=val_transforms, num_workers=args.num_workers,
                                   cache_rate=args.cache_rate, bm_path=args.bm_val_path, bm_prefix=args.bm_prefix)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented.")
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    if args.activation == 'softmax':
        activation = torch.nn.Softmax(dim=1)
    elif args.activation == 'none':
        activation = None
    else:
        raise NotImplementedError(f"Activation {args.activation} is not implemented.")
    inferer = SlidingWindowInferer(roi_size=(args.input_size, args.input_size, args.input_size),
                                   sw_batch_size=args.sw_batch_size, mode='gaussian', overlap=0.25)

    ''' Define model '''
    if args.model_type == 'de':
        models = [
            get_model(model_name=args.model,
                      n_classes=args.n_classes,
                      n_input=len(args.input_modalities),
                      input_size=args.input_size).to(device)
            for mc in args.ensemble_checkpoints[0].split(',')
        ]
        for m, mc in zip(models, args.ensemble_checkpoints[0].split(',')):
            m.load_state_dict(torch.load(mc))
        evaluator = PredictorNpzEnsemble(val_dataloader, activation=activation, device=device, temperature=args.temperature,
                                         save_path=save_path, set_name=args.eval_set_name,
                                         n_classes=args.n_classes, class_num=args.class_num, inferer=inferer)
        ''' Run evaluation '''
        logging.info("Started evaluation")
        evaluator(models)
    elif args.model_type == 'mcdp':
        model = get_model(model_name=args.model,
                          n_classes=args.n_classes,
                          n_input=len(args.input_modalities),
                          input_size=args.input_size, dropout=args.dropout_proba).to(device)
        model.load_state_dict(torch.load(args.model_checkpoint))
        evaluator = PredictorNpzMCDP(val_dataloader, activation=activation, device=device, temperature=args.temperature,
                                     save_path=save_path, set_name=args.eval_set_name, n_samples=args.n_samples,
                                     n_classes=args.n_classes, class_num=args.class_num, inferer=inferer)
        evaluator(model)
    else:
        print(f"Argument {args.model_type} not applicable for the model")
