from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_json_result_list_filters_before_sharding(tmp_path):
    import json

    for index in range(7):
        _write_motion(tmp_path, f"day/motion_{index:02d}")
    selected = [
        "day/motion_01/result.npz",
        "day/motion_03/result.npz",
        "day/motion_06/result.npz",
    ]
    manifest = tmp_path / "selected.json"
    manifest.write_text(json.dumps(selected))
    names = bonesseed_smpl.read_clip_list(manifest)
    shards = [
        bonesseed_smpl.load_dir(tmp_path, names=names, shard=(rank, 2))
        for rank in range(2)
    ]
    assert [[clip.name for clip in shard] for shard in shards] == [
        ["day/motion_01", "day/motion_06"],
        ["day/motion_03"],
    ]


def test_selected_shard_only_requires_its_own_files(tmp_path):
    _write_motion(tmp_path, "motion_00")
    _write_motion(tmp_path, "motion_02")
    clips = bonesseed_smpl.load_dir(
        tmp_path,
        names=["motion_00", "motion_01_missing", "motion_02"],
        shard=(0, 2),
    )
    assert [clip.name for clip in clips] == ["motion_00", "motion_02"]


@pytest.mark.parametrize("shard", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_shard_is_rejected(tmp_path, shard):
    _write_motion(tmp_path, "motion")
    with pytest.raises(ValueError, match="invalid motion shard"):
        bonesseed_smpl.load_dir(tmp_path, shard=shard)


def test_obsolete_euler_schema_is_rejected(tmp_path):
    _write_motion(tmp_path, "old")
    path = tmp_path / "old" / "motion.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    np.savez(path, schema=np.asarray("bonesseed-smpl-character-v1"), **arrays)
    with pytest.raises(ValueError, match="obsolete v1 XYZ-Euler motion"):
        bonesseed_smpl.load_dir(tmp_path)


def test_obsolete_v2_coordinate_schema_is_rejected(tmp_path):
    _write_motion(tmp_path, "old_coordinates")
    path = tmp_path / "old_coordinates" / "motion.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    np.savez(
        path,
        schema=np.asarray("bonesseed-smpl-character-v2"),
        joint_vel=np.zeros((3, 69), dtype=np.float32),
        **arrays,
    )
    with pytest.raises(ValueError, match="obsolete v2 motion"):
        bonesseed_smpl.load_dir(tmp_path)


def test_v3_loads_source_authoritative_joint_velocity(tmp_path):
    from ndm_gym.motion.library import MotionLib

    destination = tmp_path / "new" / "motion.npz"
    destination.parent.mkdir(parents=True)
    velocity = np.arange(4 * 69, dtype=np.float32).reshape(4, 69) / 100
    np.savez(
        destination,
        schema=np.asarray("bonesseed-smpl-character-v3"),
        root_p=np.zeros((4, 3), dtype=np.float32),
        root_quat_wxyz=np.tile(
            np.asarray((1, 0, 0, 0), dtype=np.float32), (4, 1)
        ),
        joints=np.zeros((4, 69), dtype=np.float32),
        joint_vel=velocity,
        fps=np.asarray(50, dtype=np.float32),
    )
    clip = bonesseed_smpl.load_dir(tmp_path)[0]
    torch.testing.assert_close(clip.joint_vel, torch.from_numpy(velocity))
    motion = MotionLib([clip], object(), "cpu", cache_body_state=False)
    torch.testing.assert_close(motion.qd, torch.from_numpy(velocity))
