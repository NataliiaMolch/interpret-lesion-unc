import argparse


def get_train_options():
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('--exp_name', required=True, type=str, help="name of the experiment")
    # training parameters
    parser.add_argument('--loss', default='dfl', type=str, help='dfl|gdfl|ndfl|bndl')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Specify the number of epochs to train for')
    parser.add_argument('--save_freq', default=1, type=int,
                        help="frequency in epochs of saving mmodel")
    parser.add_argument('--print_freq', default=50, type=int,
                        help="frequency in iteration of printing training loss")
    # scheduler
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Specify the initial learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-6,
                        help='Minimal learning rate')
    parser.add_argument('--scheduler', default='none', type=str, help='none|steplr|cyclic|cosine|plateau')
    parser.add_argument('--best_metric', default='loss', type=str,
                        help='metric based on which quality improvement model will be saved')
    parser.add_argument('--best_min_max', default='min', type=str,
                        help='min if the lower the better, max if otherwise')
    parser.add_argument('--step_size', default=10, type=int, help='for scheduler')
    parser.add_argument('--gamma', default=0.8, type=float, help='for scheduler')
    # Validation measures
    parser.add_argument('--ndsc_r', type=float, default=2e-5, help='nDSC parameter')
    parser.add_argument('--det_method', type=str, default='iou_adj', help='non-zero|iou|iou_adj')
    parser.add_argument('--det_threshold', type=float, default=0.1,
                        help='iou|iou_adj threshold for a lesion to be considered tpl')
    # warmup
    parser.add_argument('--warmup_iters', type=int, default=0,
                        help='number of iterations that learning rate increases from 0 to `learning_rate`')
    # model parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Specify the global random seed')
    parser.add_argument('--model', type=str, default='unet',
                        help="unet|unet_shallow|unet_shallow_dropout|unetr|swin_unetr")
    parser.add_argument('--dropout_proba', type=float, default=None,
                        help='for the unet_shallow_dropout model define the mcdp proba value')
    parser.add_argument('--n_classes', type=str, default=2,
                        help="number of output dimensions including background")
    parser.add_argument('--pretrain_checkpoint',
                        type=str, default=None, help="path to saved model")
    # validation
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for lesion detection')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Evaluation every val_interval epochs')
    parser.add_argument('--sw_batch_size', type=int, default=4, help='monai inferer parameter')
    # early stopping
    parser.add_argument('--patience', type=int, default=5, help='max number of silent epochs')
    parser.add_argument('--tolerance', type=float, default=1e-7, help='neglectable change in loss')
    # data options
    parser.add_argument('--input_modalities', type=str, nargs='+', default=['flair', 'mp2rage'])
    parser.add_argument('--input_train_paths', type=str, nargs='+', required=True)
    parser.add_argument('--target_train_path', type=str, required=True)
    parser.add_argument('--input_val_paths', type=str, nargs='+', required=True)
    parser.add_argument('--target_val_path', type=str, required=True)
    parser.add_argument('--target_prefix', type=str, required=True)
    parser.add_argument('--input_prefixes', type=str, nargs='+', required=True)
    # sub-volumes formation
    parser.add_argument('--crop_factor', type=float, default=4./3.,
                        help="ratio of first cropped patch to final cropped patch")
    parser.add_argument('--input_size', type=float, default=96, help="size of the patch")
    parser.add_argument('--n_patches', type=int, default=32,
                        help="number of patches generated from one subject")
    # data loader
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size in data loader")
    # data handling
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
    parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
    # save
    parser.add_argument('--path_save', type=str, default='', help='Specify the path to the save directory')

    return parser.parse_args()


def get_predict_options(class_specific=False):
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('--exp_name', required=True, type=str, help="name of the experiment")
    parser.add_argument('--eval_set_name', required=True, type=str,
                        help="name of evaluation set")
    parser.add_argument('--seed', type=int, default=1,
                        help='Specify the global random seed')
    # tuned parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for lesion detection')
    parser.add_argument('--l_min', type=float, default=3,
                        help='minimum number of voxels in the predicted lesion')
    parser.add_argument('--model_type', required=True, help="mcdp|simple|de")
    parser.add_argument('--ensemble_checkpoints', type=str, nargs='+')
    parser.add_argument('--model_checkpoint', type=str, help="path to saved model")
    parser.add_argument('--n_samples', type=int, default=10, help="number of mcdp samples")
    parser.add_argument('--n_jobs', type=int, default=1, help='joblib parameter')
    # model parameters
    parser.add_argument('--model', type=str, default='unet', help="unet|unet_shallow|unetr|swin_unetr")
    parser.add_argument('--dropout_proba', type=float, default=None,
                        help='for the unet_shallow_dropout model define the mcdp proba value for the inference')
    parser.add_argument('--n_classes', type=int, default=2,
                        help="number of output dimensions including background")
    if class_specific:
        parser.add_argument('--class_num', type=int, default=1,
                            help="for programs where the computations are done for the specific class")
    # inferer
    parser.add_argument('--sw_batch_size', type=int, default=4, help='monai inferer parameter')
    # data options
    parser.add_argument('--input_modalities', type=str, nargs='+', default=['flair', 'mp2rage'])
    parser.add_argument('--input_val_paths', type=str, nargs='+', required=True)
    parser.add_argument('--target_val_path', type=str, required=True)
    parser.add_argument('--target_prefix', type=str, required=True)
    parser.add_argument('--bm_val_path', type=str, required=True, help='brain mask')
    parser.add_argument('--bm_prefix', type=str, required=True, help='brain mask')
    parser.add_argument('--input_prefixes', type=str, nargs='+', required=True)
    # sub-volumes formation
    parser.add_argument('--input_size', type=float, default=96, help="size of the patch")
    # data loader
    parser.add_argument('--dataset', type=str, default='niftidataset', help='niftidataset')
    # data handling
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
    parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
    # save
    parser.add_argument('--path_save', type=str, default='', help='Specify the path to the save directory')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--include_background', action='store_true')
    parser.add_argument('--ndsc_r', type=float, default=2e-5, help='nDSC parameter')
    parser.add_argument('--det_method', type=str, default='iou_adj', help='non-zero|iou|iou_adj')
    parser.add_argument('--det_threshold', type=float, default=0.1,
                        help='iou|iou_adj threshold for a lesion to be considered tpl')
    # activation type
    parser.add_argument('--activation', type=str, default='softmax', help="softmax|none")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature scaling parameter applied before softmax")
    return parser.parse_args()


def get_test_options(clwml_eval=False):
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    parser.add_argument('--exp_name', required=True, type=str, help="name of the experiment")
    parser.add_argument('--eval_set_name', required=True, type=str,
                        help="name of evaluation set")
    parser.add_argument('--n_jobs', type=int, default=1, help='joblib parameter')
    # model parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Specify the global random seed')
    parser.add_argument('--class_number', type=int, default=1,
                        help="class for which the predictions are made")
    # tuned parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for lesion detection')
    parser.add_argument('--l_min', type=float, default=3,
                        help='minimum number of voxels in the predicted lesion')
    # for cl wml evaluation only
    if clwml_eval:
        parser.add_argument('--clmask_val_path', type=str, default=None)
        parser.add_argument('--clmask_prefix', type=str, default=None)
        parser.add_argument('--wmlmask_val_path', type=str, default=None)
        parser.add_argument('--wmlmask_prefix', type=str, default=None)
    # sub-volumes formation
    parser.add_argument('--input_size', type=float, default=96, help="size of the patch")
    # data loader
    parser.add_argument('--path_pred', type=str, required=True, help="path where the predictions are stored")
    parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
    # save
    parser.add_argument('--path_save', type=str, default='', help='Specify the path to the save directory')
    # metrics parameters
    parser.add_argument('--ndsc_r', type=float, default=2e-5, help='nDSC parameter')
    parser.add_argument('--det_method', type=str, default='iou_adj', help='non-zero|iou|iou_adj')
    parser.add_argument('--det_threshold', type=float, default=0.1,
                        help='iou|iou_adj threshold for a lesion to be considered tpl')
    return parser.parse_args()
