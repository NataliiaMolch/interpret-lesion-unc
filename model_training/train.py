from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
# random seed fix
import random
import sys, os
from pathlib import Path
sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
# inner imports
from utils.options import get_train_options
from data_processing.datasets import NiftiDataset
from utils.transforms import get_val_transforms, get_cltrain_transforms
from data_processing.dataloaders import ShuffleDataLoader
from trainer import Trainer
from callbacks import *
from validator import Validator
from utils.logger import Logger, save_options
from losses import get_loss
from utils.metrics import *
from utils.models import get_model
from scheduling import get_scheduler
# logging
import logging

if __name__ == '__main__':
    ''' Parse and save arguments '''
    args = get_train_options()
    save_path = os.path.join(args.path_save, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    save_options(args=args, filepath=os.path.join(save_path, "train_options.txt"))
    train_log_file = os.path.join(save_path, 'train.log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(train_log_file),
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
    train_transforms = get_cltrain_transforms(input_keys=args.input_modalities, label_key="targets",
                                              generate_instance_mask=True if args.loss in ['bndl',
                                                                                           'detl_cl'] else False,
                                              crop_factor=args.crop_factor,
                                              roi_size=(args.input_size, args.input_size, args.input_size),
                                              n_patches=args.n_patches).set_random_state(seed=seed)

    val_transforms = get_val_transforms(input_keys=args.input_modalities, label_key="targets",
                                        generate_instance_mask=True if args.loss in ['bndl', 'detl_cl'] else False
                                        ).set_random_state(seed=seed)
    train_dataset = NiftiDataset(input_paths=args.input_train_paths, input_prefixes=args.input_prefixes,
                                 input_names=args.input_modalities,
                                 target_path=args.target_train_path, target_prefix=args.target_prefix,
                                 transforms=train_transforms,
                                 num_workers=args.num_workers, cache_rate=args.cache_rate)
    train_eval_dataset = NiftiDataset(input_paths=args.input_train_paths, input_prefixes=args.input_prefixes,
                                      input_names=args.input_modalities,
                                      target_path=args.target_train_path, target_prefix=args.target_prefix,
                                      transforms=val_transforms,
                                      num_workers=args.num_workers, cache_rate=args.cache_rate)
    val_dataset = NiftiDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes,
                               input_names=args.input_modalities,
                               target_path=args.target_val_path, target_prefix=args.target_prefix,
                               transforms=val_transforms, num_workers=args.num_workers,
                               cache_rate=args.cache_rate)
    train_dataloader = ShuffleDataLoader(train_dataset, batch_size=args.batch_size, n_patches=args.n_patches,
                                         monai_num_workers=args.num_workers, shuffle=True, num_workers=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)
    train_eval_dataloader = DataLoader(train_eval_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)
    ''' Define model '''
    model = get_model(model_name=args.model, n_classes=args.n_classes,
                      n_input=len(args.input_modalities), input_size=args.input_size, dropout=args.dropout_proba)
    if args.pretrain_checkpoint is not None:
        model.load_state_dict(torch.load(args.pretrain_checkpoint))
    model = model.to(device)
    print(model)
    activation = torch.nn.Softmax(dim=1)
    loss_function = get_loss(loss_name=args.loss, activation=activation, device=device)

    ''' Define validation actors '''
    inferer = SlidingWindowInferer(roi_size=(args.input_size, args.input_size, args.input_size),
                                   sw_batch_size=args.sw_batch_size, mode='gaussian', overlap=0.25)
    metrics_list = [
        partial(voxel_scale_metric, r=args.ndsc_r, check=False)
        # partial(nDSC_metric, r=args.ndsc_r, check=False),
        # partial(DSC_metric, check=False)#,
        # partial(lesion_detection_metric, method=args.det_method, threshold=args.det_threshold, check=False)
                    ]
    validator = Validator(val_dataloader, activation=activation, metrics_funcs=metrics_list, device=device,
                          prob_threshold=args.threshold, include_background=False, to_onehot_y=True,
                          n_classes=args.n_classes,
                          loss_function=loss_function, inferer=inferer)
    train_validator = Validator(train_eval_dataloader, activation=activation, metrics_funcs=metrics_list, device=device,
                                prob_threshold=args.threshold, include_background=False, to_onehot_y=True,
                                n_classes=args.n_classes, loss_function=loss_function, inferer=inferer)

    ''' Define training actors '''
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.01)
    scheduler = get_scheduler(scheduler_name=args.scheduler, learning_rate=args.learning_rate, optimizer=optimizer,
                              best_min_max=args.best_min_max, step_size=args.step_size, gamma=args.gamma,
                              learning_rate_min=args.learning_rate_min)

    logger = Logger(log_file=os.path.join(save_path, "train_log.csv"))
    # evaluation on the val set + early stopping based on val
    general_callback = GeneralLoggingCallback(logger, validator,
                                              print_name='val',
                                              patience=args.patience, tolerance=args.tolerance,
                                              min_is_good=True if args.best_min_max == 'min' else False,
                                              metric=args.best_metric, train_log_file=train_log_file)
    # evaluation on the training set
    train_log_callback = LoggingCallback(logger, validator=train_validator, print_name='train',
                                         print_freq=args.val_interval)
    # plot val and train metrics and losses
    plotting_callback = PlottingCallback(logger, print_freq=args.val_interval)
    # plot learning rate
    plotting_lr_callback = PlotLRCallback(logger)
    # train
    trainer = Trainer(train_dataloader, network=model, optimiser=optimizer, scheduler=scheduler,
                      loss_function=loss_function, max_epochs=args.n_epochs, device=device, print_freq=args.print_freq,
                      train_log_file=train_log_file,
                      min_lr=args.learning_rate_min if args.scheduler != 'cosine' else 0.,
                      callbacks=[general_callback, train_log_callback, plotting_callback, plotting_lr_callback],
                      warmup_iters=args.warmup_iters)
    ''' Run training '''
    trainer.run()
