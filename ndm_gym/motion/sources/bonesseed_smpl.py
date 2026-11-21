"""Compact BonesSEED UMR++ motions mapped to the 69-DoF SMPL character."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from . import Clip


def _clip_name(value: str) -> str:
    name = value.strip().replace("\\", "/").lstrip("./")
    for suffix in ("/result.npz", "/motion.npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.rstrip("/")


def read_clip_list(path: str | Path) -> list[str]:
    """Read converted clip names from text or a UMR++ JSON manifest."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text())
        if isinstance(raw, dict):
            raw = raw.get("motions", raw.get("items"))
        if not isinstance(raw, list):
            raise ValueError("SMPL JSON clip list must be a list or contain motions/items")
        values = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("path") or item.get("output_dir") or item.get("result")
                if item is None:
                    raise ValueError("SMPL JSON clip-list item has no path/output_dir/result")
            values.append(str(item))
    else:
        values = [
            line.split("#", 1)[0].strip()
            for line in source.read_text().splitlines()
        ]
    names = [_clip_name(value) for value in values if value.strip()]
    if not names:
        raise ValueError(f"{source} lists no clips")
    if len(set(names)) != len(names):
        raise ValueError(f"{source} lists a clip more than once")
    return names


def load_dir(
    path: str | Path,
    max_clips: int | None = None,
    *,
    names: list[str] | None = None,
    shard: tuple[int, int] | None = None,
) -> list[Clip]:
    """Load converted SMPL clips, optionally taking one deterministic shard.

    Sharding is applied to the sorted file list before any NPZ is opened.  A
    distributed worker therefore reads and retains only its own motions rather
    than constructing the complete library and discarding the other ranks'
    clips afterwards.
    """
    root = Path(path).expanduser().resolve()
    selected_names = sorted(names) if names is not None else None
    if selected_names is None:
        files = sorted(glob.glob(str(root / "**" / "motion.npz"), recursive=True))
        if max_clips is not None:
            files = files[:max_clips]
    else:
        if max_clips is not None:
            selected_names = selected_names[:max_clips]
        files = [str(root / name / "motion.npz") for name in selected_names]
    if shard is not None:
        index, count = shard
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"invalid motion shard {index}/{count}")
        files = files[index::count]
    if selected_names is not None:
        missing = [
            Path(file).parent.relative_to(root).as_posix()
            for file in files
            if not Path(file).is_file()
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} selected SMPL clips are missing under {root}: "
                f"{missing[:10]}"
            )
    if not files:
        raise FileNotFoundError(
            f"no converted SMPL-character motions (motion.npz) under {root}"
        )
    clips: list[Clip] = []
    for name in files:
        file = Path(name)
        with np.load(file, allow_pickle=False) as data:
            schema = str(np.asarray(data["schema"]).item()) if "schema" in data else None
            if schema == "bonesseed-smpl-character-v1":
                raise ValueError(
                    f"{file}: obsolete v1 XYZ-Euler motion; reconvert it with "
                    "convert_bonesseed_umrpp_to_smpl.py"
                )
            if schema not in (None, "bonesseed-smpl-character-v2"):
                raise ValueError(f"{file}: unsupported schema {schema!r}")
            root_p = torch.from_numpy(np.asarray(data["root_p"], dtype=np.float32).copy())
            root_quat = torch.from_numpy(
                np.asarray(data["root_quat_wxyz"], dtype=np.float32).copy()
            )
            joints = torch.from_numpy(np.asarray(data["joints"], dtype=np.float32).copy())
            joint_vel = (
                torch.from_numpy(
                    np.asarray(data["joint_vel"], dtype=np.float32).copy()
                )
                if "joint_vel" in data
                else None
            )
            fps = float(np.asarray(data["fps"]).item())
        if root_p.ndim != 2 or root_p.shape[1] != 3:
            raise ValueError(f"{file}: root_p must be [T,3]")
        if root_quat.shape != (root_p.shape[0], 4):
            raise ValueError(f"{file}: root_quat_wxyz must be [T,4]")
        if joints.shape != (root_p.shape[0], 69):
            raise ValueError(f"{file}: joints must be [T,69]")
        if schema == "bonesseed-smpl-character-v2":
            if joint_vel is None or joint_vel.shape != joints.shape:
                raise ValueError(f"{file}: v2 joint_vel must match joints")
            if abs(fps - 50.0) > 1e-6:
                raise ValueError(f"{file}: v2 clips must be pre-resampled to 50 Hz")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{file}: invalid fps {fps}")
        relative = file.parent.relative_to(root).as_posix()
        clips.append(
            Clip(
                relative,
                root_p,
                root_quat,
                joints,
                fps,
                meta={"source": str(file)},
                joint_vel=joint_vel,
            )
        )
    return clips
