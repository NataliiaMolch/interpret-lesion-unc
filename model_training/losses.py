import torch
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
import warnings
from monai.networks import one_hot
from typing import Union
from monai.losses import GeneralizedDiceFocalLoss, DiceFocalLoss, FocalLoss, TverskyLoss


def get_loss(loss_name, activation, device):
    if loss_name == 'gdfl':
        loss_function = GeneralizedDiceFocalLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                                 lambda_gdl=0.5, lambda_focal=1.0)
    elif loss_name == 'dfl':
        loss_function = DiceFocalLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                      lambda_dice=0.5, lambda_focal=1.0)
    elif loss_name == 'ndfl':
        loss_function = NormalisedDiceFocalLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                                lambda_ndscl=0.5, lambda_focal=1.0)
    elif loss_name == 'bndl':
        loss_function = BlobNormalisedDiceLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                               lambda_global=2.0, lambda_blob=1.0, device=device)
    elif loss_name == 'tversky_cl':
        loss_function = TverskyLoss(include_background=False, to_onehot_y=True, sigmoid=False, softmax=False,
                                    other_act=activation, alpha=0.3, beta=0.7)
    elif loss_name == 'focal_cl':
        loss_function = FocalLoss(include_background=False, to_onehot_y=True, gamma=2.0, weight=2e5)
    elif loss_name == 'detl_cl':
        loss_function = DetectionLoss(include_background=False, to_onehot_y=True, sigmoid=False, softmax=False,
                                      other_act=activation, iou_threshold=0.1, avg_lesion_size=6, device=device)
    else:
        raise NotImplementedError(f"Loss {loss_name} not implemented")
    return loss_function


class NormalisedDiceLoss(_Loss):
    """
    Inherited from GeneralizedDiceLoss and DiceLoss
    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act=None,
            reduction=LossReduction.MEAN,
            effective_load: Union[torch.Tensor, float] = 1e-3,
            batch: bool = False,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        self.effective_load = float(effective_load)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
            raise NotImplementedError("Was never debugged")

        ground_o = torch.sum(target, dim=reduce_axis)  # tp + fn
        pred_o = torch.sum(input, dim=reduce_axis)  # tp + fp

        n_element = target.nelement() / target.shape[0] / target.shape[1]

        if isinstance(self.effective_load, torch.Tensor):
            if target.shape[1] != self.effective_load.shape[0]:
                raise ValueError(
                    f"expecting effective_loads length to be equal to the number of classes {target.shape[1]}, got {self.effective_load.shape[0]}.")
            effective_load_b = torch.stack([self.effective_load for _ in range(target.shape[0])], dim=0).to(target.device)
            scaling_factor = (1.0 - effective_load_b) * ground_o / (effective_load_b * (n_element - ground_o))
        elif isinstance(self.effective_load, float):
            scaling_factor = (1.0 - self.effective_load) * ground_o / (self.effective_load * (n_element - ground_o))
        else:
            raise ValueError(f"expecting effective_loads to be float or torch.Tensor, got {type(self.effective_load)}")
        scaling_factor[ground_o == 0.0] = 1.0
        scaling_factor.to(target.device)

        # scaling_factor = (1 - self.effective_load) * ground_o / (self.effective_load * (n_element - ground_o))
        # scaling_factor[ground_o == 0.0] = 1.0

        tp = torch.sum(target * input, dim=reduce_axis)
        fp = pred_o - tp

        f: torch.Tensor = 1.0 - (2.0 * tp) / (scaling_factor * fp + ground_o + tp)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class NormalisedDiceFocalLoss(_Loss):
    def __init__(self,
                 include_background: bool = True,
                 to_onehot_y: bool = False,
                 sigmoid: bool = False,
                 softmax: bool = False,
                 other_act=None,
                 reduction=LossReduction.MEAN,
                 smooth_r: float = 1e-3,
                 batch: bool = False,
                 gamma: float = 2.0,
                 focal_weight=None,
                 lambda_ndscl: float = 1.0,
                 lambda_focal: float = 1.0):
        super().__init__()
        self.normalized_dice = NormalisedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
            effective_load=smooth_r,
            batch=batch,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction,
        )
        if lambda_ndscl < 0.0:
            raise ValueError("lambda_gdl should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_ndscl = lambda_ndscl
        self.lambda_focal = lambda_focal

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target (torch.Tensor): the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When the input and target tensors have different numbers of dimensions, or the target
                channel isn't either one-hot encoded or categorical with the same shape of the input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                f"Input - {input.shape} - and target - {target.shape} - must have the same number of dimensions."
            )

        gdl_loss = self.normalized_dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_ndscl * gdl_loss + self.lambda_focal * focal_loss
        return total_loss


class BlobNormalisedDiceLoss(_Loss):
    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act=None,
            reduction=LossReduction.MEAN,
            effective_load: float = 1e-3,
            batch: bool = False,
            lambda_global: float = 2.0,
            lambda_blob: float = 1.0,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        self.effective_load = float(effective_load)
        self.batch = batch

        self.lambda_global = lambda_global
        self.lambda_blob = lambda_blob

        self.device = device

        self.ndsc_loss = NormalisedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
            effective_load=effective_load,
            batch=batch
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
        blob_loss = torch.tensor(0.0).to(self.device)
        for label in instance_mask.unique()[1:]:
            mask = torch.isin(instance_mask, test_elements=torch.Tensor([0.0, label]).to(self.device)).to(self.device)
            blob_loss += self.ndsc_loss(input=input*mask, target=target*mask)
        blob_loss /= len(instance_mask.unique()[1:])

        global_loss = self.ndsc_loss(input, target)

        return self.lambda_global * global_loss + self.lambda_blob * blob_loss


class DetectionLoss(_Loss):
    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act=None,
            reduction=LossReduction.MEAN,
            iou_threshold: float = 0.25,
            avg_lesion_size: int = 21,
            batch: bool = False,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

        self.iou_threshold = float(iou_threshold)
        self.avg_lesion_size = avg_lesion_size
        self.batch = batch

        self.device = device

    def intersection_over_union(self, mask1: torch.Tensor, mask2: torch.Tensor):
        intersection = (mask1 * mask2).sum()
        union = (mask1 + mask2).sum() - intersection
        return intersection / union

    def forward(self, input: torch.Tensor, target: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
            raise NotImplementedError("Was never debugged")

        fn = torch.tensor(0.0).to(self.device)
        tp = torch.tensor(0.0).to(self.device)

        # compute n of tp and n of fn (undersegmented lesions are the ones that have iou less than threshold)
        for label in instance_mask.unique()[1:]:
            mask = torch.isin(instance_mask, test_elements=label).to(self.device)
            iou = self.intersection_over_union(mask, target)
            if iou > self.iou_threshold:
                tp += 1.
            else:
                fn += 1.

        # approximate n fp
        fp = (input[target == 0].sum() / self.avg_lesion_size).to(self.device)
        det_loss = 2.0 * tp / (2.0 * tp + fp + fn)
        return det_loss
