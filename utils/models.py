from monai.networks.nets import UNet, SwinUNETR, UNETR, AttentionUnet
from monai.networks.blocks import Convolution
import torch
from torch import nn


def get_model(model_name, n_classes, n_input, input_size, dropout: float =None):
    if model_name == 'unet':
        model = UNet(spatial_dims=3, in_channels=n_input, out_channels=n_classes,
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), norm='batch', num_res_units=0)
    elif model_name == 'unet_shallow':
        model = UNet(spatial_dims=3, in_channels=n_input, out_channels=n_classes,
                     channels=(32, 64, 128, 256), strides=(2, 2, 2), norm='batch', num_res_units=0)
    elif model_name == 'unet_shallow_dropout':
        model = UNet(spatial_dims=3, in_channels=n_input, out_channels=n_classes, dropout=dropout,
                     channels=(32, 64, 128, 256), strides=(2, 2, 2), norm='batch', num_res_units=0)
    elif model_name == "unetr":
        model = UNETR(img_size=input_size, in_channels=n_input,
                      out_channels=n_classes, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12)
    elif model_name == 'swin_unetr':
        model = SwinUNETR(img_size=input_size, in_channels=n_input,
                          out_channels=n_classes, depths=(2, 2, 2), num_heads=(3, 6, 12), feature_size=12)
    else:
        raise ValueError(f"Can't use option {model_name} for --model argument")
    # weights intialization
    for layer in model.model.modules():
        if type(layer) == nn.Conv3d:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
    return model