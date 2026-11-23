"""U-Net trained from scratch (MONAI implementation)."""

from monai.networks.nets import UNet
from torch import nn

N_CLASSES = 4  # background, RV, myocardium, LV


def build_scratch_unet(dropout: float = 0.1) -> nn.Module:
    """2D U-Net: 1 input channel (greyscale MRI), 4 output channels (one score per class).

    Five levels with 32 -> 512 channels; four stride-2 steps take a 192 x 192 slice down to 12 x 12
    at the bottom, where each feature sees most of the heart at once."""
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=N_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
