from collections import Counter
from pathlib import Path

import pytest

from cardiac.data.io import ACDC_GROUPS, PatientMeta
from cardiac.data.splits import assert_disjoint, load_splits, make_splits

COMMITTED_SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits.json"


def _fake_meta(index: int, split: str) -> PatientMeta:
    return PatientMeta(
        pid=f"patient{index:03d}",
        split=split,  # type: ignore[arg-type]
        group=ACDC_GROUPS[(index - 1) // (20 if split == "train" else 10) % 5],
        height_cm=170.0,
        weight_kg=70.0,
        n_frames=30,
        ed_frame=1,
        es_frame=12,
        original_spacing_mm=(1.5, 1.5, 10.0),
        n_slices=10,
    )


@pytest.fixture
def pools() -> tuple[list[PatientMeta], list[PatientMeta]]:
    train_pool = [_fake_meta(i, "train") for i in range(1, 101)]
    test = [_fake_meta(i, "test") for i in range(101, 151)]
    return train_pool, test


def test_no_patient_in_two_splits(pools: tuple[list[PatientMeta], list[PatientMeta]]) -> None:
    splits = make_splits(*pools)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_sizes_and_stratification(pools: tuple[list[PatientMeta], list[PatientMeta]]) -> None:
    train_pool, _ = pools
    splits = make_splits(*pools)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (80, 20, 50)
    group_of = {m.pid: m.group for m in train_pool}
    assert Counter(group_of[p] for p in splits["val"]) == dict.fromkeys(ACDC_GROUPS, 4)


def test_official_test_set_is_unchanged(
    pools: tuple[list[PatientMeta], list[PatientMeta]],
) -> None:
    _, test = pools
    assert make_splits(*pools)["test"] == sorted(m.pid for m in test)


def test_same_seed_gives_same_split(pools: tuple[list[PatientMeta], list[PatientMeta]]) -> None:
    assert make_splits(*pools, seed=0) == make_splits(*pools, seed=0)
    assert make_splits(*pools, seed=0)["val"] != make_splits(*pools, seed=1)["val"]


def test_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        assert_disjoint({"train": ["p1", "p2"], "val": ["p2"], "test": []})


@pytest.mark.skipif(not COMMITTED_SPLITS.exists(), reason="run scripts/make_splits.py first")
def test_committed_splits_file_is_disjoint_and_complete() -> None:
    splits = load_splits(COMMITTED_SPLITS)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (80, 20, 50)
