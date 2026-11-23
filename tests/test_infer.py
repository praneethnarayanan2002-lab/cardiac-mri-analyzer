import numpy as np
import torch

from cardiac.data.augment import training_transforms
from cardiac.infer import fill_cavity_holes, keep_largest_component, predict_volume
from cardiac.models.unet import build_scratch_unet


def test_largest_component_removes_stray_blob() -> None:
    label = np.zeros((20, 20, 3), dtype=np.uint8)
    label[5:12, 5:12, :] = 3
    label[17:19, 17:19, 1] = 3  # a stray blob far from the heart
    out = keep_largest_component(label)
    assert out[17:19, 17:19, 1].sum() == 0
    assert (out[5:12, 5:12, :] == 3).all()


def test_fill_holes_fills_cavity_but_keeps_myocardium_ring() -> None:
    label = np.zeros((20, 20, 1), dtype=np.uint8)
    label[3:17, 3:17] = 2  # myocardium ring...
    label[6:14, 6:14] = 3  # ...around the LV
    label[9, 9] = 0  # a speck of background inside the LV
    out = fill_cavity_holes(label)
    assert out[9, 9, 0] == 3
    assert (out[6:14, 6:14] == 3).all()  # LV was not swallowed by the myocardium


def test_predict_volume_returns_original_shape() -> None:
    model = build_scratch_unet()
    image = np.random.rand(200, 180, 5).astype(np.float32)
    label = predict_volume(model, image, (1.25, 1.25, 10.0), torch.device("cpu"))
    assert label.shape == image.shape
    assert label.dtype == np.uint8
    assert set(np.unique(label)) <= {0, 1, 2, 3}


def test_augmentation_keeps_image_and_label_aligned() -> None:
    image = np.zeros((1, 64, 64), dtype=np.float32)
    image[0, 20:40, 10:30] = 1.0
    sample = {"image": image, "label": (image > 0.5).astype(np.float32) * 3}
    transforms = training_transforms()
    for seed in range(10):
        transforms.set_random_state(seed)
        out = transforms(dict(sample))
        label = np.asarray(out["label"])
        assert set(np.unique(label)) <= {0.0, 3.0}  # nearest-neighbour: no in-between labels
        bright = np.asarray(out["image"])[label == 3]
        assert bright.size == 0 or np.median(bright) > 0.3  # the label still sits on the square
