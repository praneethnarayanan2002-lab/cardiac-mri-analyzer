import numpy as np

from cardiac.models.medsam import box_from_mask, jitter_box, to_rgb_uint8


def test_box_is_padded_and_uses_x_for_columns() -> None:
    mask = np.zeros((50, 60), dtype=bool)
    mask[10:20, 30:41] = True  # rows 10-19, columns 30-40
    assert box_from_mask(mask, pad=5).tolist() == [25, 5, 45, 24]


def test_box_is_clipped_to_the_image() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:3, 17:20] = True
    assert box_from_mask(mask, pad=5).tolist() == [12, 0, 19, 7]


def test_zero_jitter_returns_the_same_box() -> None:
    box = np.array([10.0, 12.0, 40.0, 44.0])
    out = jitter_box(box, 0, np.random.default_rng(0), (64, 64))
    assert (out == box).all()


def test_jitter_stays_within_magnitude_and_image() -> None:
    box = np.array([10.0, 12.0, 40.0, 44.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        out = jitter_box(box, 10, rng, (48, 48))
        assert out[0] < out[2] and out[1] < out[3]
        assert out.min() >= 0 and out[2] <= 47 and out[3] <= 47
        assert np.abs(out - np.minimum(box, 47)).max() <= 10


def test_rgb_conversion_shape_and_range() -> None:
    rgb = to_rgb_uint8(np.random.default_rng(0).normal(100, 30, size=(32, 32)))
    assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8
    assert (rgb[..., 0] == rgb[..., 2]).all()
