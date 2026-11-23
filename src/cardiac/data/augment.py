"""Training-time augmentation for 2D slices (MONAI dictionary transforms).

Geometric changes are applied to the image and the label together; the label always uses
nearest-neighbour so it stays a set of whole numbers 0-3. Intensity changes touch the image only.
"""

import numpy as np
from monai.transforms import (
    Compose,
    Rand2DElasticd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)

KEYS = ("image", "label")


def training_transforms() -> Compose:
    """Each transform fires with its own probability, so most slices get a different mix."""
    return Compose(
        [
            RandRotated(
                keys=KEYS,
                range_x=np.deg2rad(15),
                prob=0.5,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandZoomd(
                keys=KEYS,
                min_zoom=0.85,
                max_zoom=1.15,
                prob=0.5,
                mode=("bilinear", "nearest"),
                padding_mode="constant",
            ),
            RandFlipd(keys=KEYS, spatial_axis=1, prob=0.5),
            Rand2DElasticd(
                keys=KEYS,
                spacing=(24, 24),
                magnitude_range=(0.5, 1.5),
                prob=0.2,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandScaleIntensityd(keys="image", factors=0.15, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.15, prob=0.3),
            RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.3),
            RandGaussianNoised(keys="image", std=0.05, prob=0.2),
        ]
    )
