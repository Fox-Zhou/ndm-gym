"""Compact BonesSEED UMR++ motions mapped to the 69-DoF PHC character."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from . import Clip


def load_dir(path: str | Path, max_clips: int | None = None) -> list[Clip]:
    root = Path(path).expanduser().resolve()
    files = sorted(glob.glob(str(root / "**" / "motion.npz"), recursive=True))
    if max_clips is not None:
        files = files[:max_clips]
    if not files:
        raise FileNotFoundError(f"no converted PHC motions (motion.npz) under {root}")
    clips: list[Clip] = []
    for name in files:
        file = Path(name)
        with np.load(file, allow_pickle=False) as data:
            root_p = torch.from_numpy(np.asarray(data["root_p"], dtype=np.float32).copy())
            root_quat = torch.from_numpy(
                np.asarray(data["root_quat_wxyz"], dtype=np.float32).copy()
            )
            joints = torch.from_numpy(np.asarray(data["joints"], dtype=np.float32).copy())
            fps = float(np.asarray(data["fps"]).item())
        if root_p.ndim != 2 or root_p.shape[1] != 3:
            raise ValueError(f"{file}: root_p must be [T,3]")
        if root_quat.shape != (root_p.shape[0], 4):
            raise ValueError(f"{file}: root_quat_wxyz must be [T,4]")
        if joints.shape != (root_p.shape[0], 69):
            raise ValueError(f"{file}: joints must be [T,69]")
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
            )
        )
    return clips
