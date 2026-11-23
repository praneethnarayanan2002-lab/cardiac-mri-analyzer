"""U-Net whose encoder is a ResNet-34 pretrained on ImageNet (segmentation_models_pytorch)."""

import segmentation_models_pytorch as smp
from torch import nn

from cardiac.models.unet import N_CLASSES


def build_pretrained_unet(encoder_weights: str | None = "imagenet") -> nn.Module:
    """ResNet-34 encoder + U-Net decoder, 1 input channel, 4 output classes.

    The ImageNet encoder expects 3 colour channels. With in_channels=1, smp sums the first
    convolution's RGB weights into one channel, so a grey MRI slice activates it the way a grey
    photo would. Everything after that first layer is used unchanged."""
    model: nn.Module = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=N_CLASSES,
    )
    return model
