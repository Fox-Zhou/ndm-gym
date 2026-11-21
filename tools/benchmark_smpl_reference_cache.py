#!/usr/bin/env python3
"""Measure SMPL reference-FK cache memory and eager reward/critic latency."""

from __future__ import annotations

import argparse
import json
import time

import torch

from ndm_gym.motion.library import MotionLib
from ndm_gym.motion.sources import bonesseed_smpl
from ndm_gym.robots.smpl_character import SMPL_CHARACTER_SPEC, kinematics
from ndm_gym.tasks.articulation_tracking import ArticulationTrackingMDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--smpl_mjcf", required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--max_clips", type=int)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def milliseconds(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    if args.batch < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("batch/repeats must be positive and warmup non-negative")
    device = "cuda:0"
    torch.cuda.set_device(device)
    kin = kinematics(args.smpl_mjcf, device)
    clips = bonesseed_smpl.load_dir(args.motion_dir, max_clips=args.max_clips)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    load_start = time.perf_counter()
    motion = MotionLib(clips, kin, device, cache_body_state=args.cache)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    library_mib = (torch.cuda.memory_allocated() - before) / 2**20

    mdp = ArticulationTrackingMDP(
        SMPL_CHARACTER_SPEC,
        kin,
        motion,
        torch.zeros(SMPL_CHARACTER_SPEC.num_joints, device=device),
    )
    frames = torch.arange(args.batch, device=device) % (motion.num_frames - 1)
    state = motion.ref_state(frames)
    action = torch.zeros(
        args.batch, SMPL_CHARACTER_SPEC.action_dim, device=device
    )
    context = torch.zeros(
        args.batch, SMPL_CHARACTER_SPEC.context_dim, device=device
    )

    with torch.inference_mode():
        reward_ms = milliseconds(
            lambda: mdp.reward(state, frames, action, action),
            args.warmup,
            args.repeats,
        )
        critic_ms = milliseconds(
            lambda: mdp.critic_features(state, frames, context),
            args.warmup,
            args.repeats,
        )
    print(
        json.dumps(
            {
                "cache": args.cache,
                "clips": len(clips),
                "frames": motion.num_frames,
                "batch": args.batch,
                "library_mib": library_mib,
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                "load_seconds": load_seconds,
                "reward_ms": reward_ms,
                "critic_features_ms": critic_ms,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
