from pathlib import Path

import numpy as np
import pytest

from ndm_gym.motion.sources import bonesseed_smpl


def _write_motion(root: Path, name: str, frames: int = 3) -> None:
    destination = root / name / "motion.npz"
    destination.parent.mkdir(parents=True)
    np.savez(
        destination,
        root_p=np.zeros((frames, 3), dtype=np.float32),
        root_quat_wxyz=np.tile(
            np.asarray((1, 0, 0, 0), dtype=np.float32), (frames, 1)
        ),
        joints=np.zeros((frames, 69), dtype=np.float32),
        fps=np.asarray(50, dtype=np.float32),
    )


def test_distributed_shards_are_disjoint_and_cover_the_library(tmp_path):
    for index in range(7):
        _write_motion(tmp_path, f"motion_{index:02d}")

    all_names = [clip.name for clip in bonesseed_smpl.load_dir(tmp_path)]
    shards = [
        [clip.name for clip in bonesseed_smpl.load_dir(tmp_path, shard=(rank, 3))]
        for rank in range(3)
    ]

    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert sorted(name for shard in shards for name in shard) == all_names
    assert not (set(shards[0]) & set(shards[1]))
    assert not (set(shards[0]) & set(shards[2]))
    assert not (set(shards[1]) & set(shards[2]))


def test_max_clips_is_a_global_limit_before_sharding(tmp_path):
    for index in range(7):
        _write_motion(tmp_path, f"motion_{index:02d}")

    shards = [
        bonesseed_smpl.load_dir(tmp_path, max_clips=5, shard=(rank, 2))
        for rank in range(2)
    ]
    assert [len(shard) for shard in shards] == [3, 2]


@pytest.mark.parametrize("shard", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_shard_is_rejected(tmp_path, shard):
    _write_motion(tmp_path, "motion")
    with pytest.raises(ValueError, match="invalid motion shard"):
        bonesseed_smpl.load_dir(tmp_path, shard=shard)
