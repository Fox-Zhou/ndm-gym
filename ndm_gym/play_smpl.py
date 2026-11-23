"""Run an SMPL SuperMimic checkpoint in Newton/MuJoCo-Warp with Viser."""

from __future__ import annotations

import argparse
from pathlib import Path

from .sim.launcher import add_launcher_args, launch_simulation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir", default=None, help="default: checkpoint training motion_dir")
    parser.add_argument("--clips", default=None, help="default: checkpoint training clip list")
    parser.add_argument("--max_clips", type=int, default=256, help="motions exposed in the GUI; 0 loads the complete list")
    parser.add_argument("--smpl_mjcf", default=None, help="default: checkpoint training asset")
    parser.add_argument("--smpl_usd", default=None, help="default: checkpoint training asset")
    parser.add_argument("--start_motion", type=int, default=0)
    parser.add_argument("--wm_rollout_steps", type=int, default=100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--autoplay", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.max_clips < 0:
        parser.error("--max_clips must be non-negative")
    if args.start_motion < 0:
        parser.error("--start_motion must be non-negative")
    return args


def _path(cli_value, checkpoint_args: dict, key: str) -> str:
    value = cli_value or checkpoint_args.get(key)
    if not value:
        raise ValueError(f"--{key} is required because the checkpoint does not record it")
    return str(Path(value).expanduser())


def main() -> None:
    args = parse_args()

    import torch

    from .motion.library import MotionLib
    from .motion.sources import bonesseed_smpl
    from .robots.smpl_character import SMPL_CHARACTER_SPEC, kinematics
    from .sim.rigid.smpl_mjwarp import SMPLTrackingSim, make_sim
    from .smpl_viewer import SMPLSuperMimicRuntime, SMPLSuperMimicViser
    from .tasks.articulation_tracking import ArticulationTrackingMDP

    device = str(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    checkpoint_args = checkpoint.get("args", {})
    motion_dir = _path(args.motion_dir, checkpoint_args, "motion_dir")
    mjcf = _path(args.smpl_mjcf, checkpoint_args, "smpl_mjcf")
    usd = _path(args.smpl_usd, checkpoint_args, "smpl_usd")
    clip_list = args.clips or checkpoint_args.get("clips")
    names = bonesseed_smpl.read_clip_list(clip_list) if clip_list else None
    clips = bonesseed_smpl.load_dir(
        motion_dir,
        max_clips=(args.max_clips or None),
        names=names,
    )
    kin = kinematics(mjcf, device)
    motion = MotionLib(clips, kin, device, cache_body_state=False)
    default_q = checkpoint.get(
        "default_q", torch.zeros(SMPL_CHARACTER_SPEC.num_joints)
    ).to(device)
    mdp = ArticulationTrackingMDP(
        SMPL_CHARACTER_SPEC, kin, motion, default_q
    )
    sim_cfg = make_sim(device)
    with launch_simulation(sim_cfg, args):
        env = SMPLTrackingSim(
            sim_cfg,
            1,
            motion,
            mdp,
            reset_noise=False,
            eval_mode=True,
            usd_path=usd,
        )
        runtime = SMPLSuperMimicRuntime(
            env,
            kin,
            motion,
            mdp,
            checkpoint,
            wm_rollout_steps=args.wm_rollout_steps,
        )
        runtime.clip_index = min(args.start_motion, len(motion.names) - 1)
        viewer = SMPLSuperMimicViser(
            runtime,
            mjcf,
            host=args.host,
            port=args.port,
            fps=args.fps,
            autoplay=args.autoplay,
        )
        viewer.run()


if __name__ == "__main__":
    main()
