"""Render the actual NDM SMPL articulation falling passively in its simulator.

This is an asset/coordinate-system acceptance check, not a policy evaluation.
The scene is rendered by IsaacLab's native visualizer from the same USD rigid
bodies stepped by Newton/MuJoCo-Warp.  No MJCF FK overlay and no learned policy
participates in the result.
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smpl_usd",
    default=(
        "/ALG/zhouyangchen/human_template/character/ndm_smpl/"
        "usd/ndm_smpl/ndm_smpl.usda"
    ),
)
parser.add_argument(
    "--smpl_mjcf",
    default=(
        "/ALG/zhouyangchen/human_template/character/ndm_smpl/ndm_smpl.xml"
    ),
)
parser.add_argument(
    "--motion_dir",
    default=(
        "/ALG/zhouyangchen/dataset/StableMotionTracking/BonesSEED/"
        "ndm_gym_smplx_umr"
    ),
)
parser.add_argument(
    "--clips",
    default=(
        "/ALG/zhouyangchen/dataset/BonesSEED/supermimic/"
        "bonesseed_smplx_umr_filtered.json"
    ),
)
parser.add_argument("--motion_index", type=int, default=0)
parser.add_argument(
    "--fall_seconds",
    type=float,
    default=4.0,
    help="seconds before resetting the passive character for another fall",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=0,
    help="number of falls; zero repeats until interrupted",
)
parser.add_argument(
    "--no_realtime",
    action="store_true",
    help="step as fast as possible instead of matching wall-clock time",
)
add_launcher_args(parser)
args = parser.parse_args()

if args.fall_seconds <= 0:
    parser.error("--fall_seconds must be positive")
if args.cycles < 0:
    parser.error("--cycles must be non-negative")
if args.motion_index < 0:
    parser.error("--motion_index must be non-negative")


import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from ndm_gym.core.rotations import mat_to_quat
from ndm_gym.motion.library import MotionLib
from ndm_gym.motion.sources import bonesseed_smpl
from ndm_gym.robots.smpl_character import kinematics
from ndm_gym.sim.rigid.smpl_mjwarp import (
    CONTROL_DT,
    FORMAL_SIM_JOINT_NAMES,
    SMPL_CFG,
    make_sim,
)


PASSIVE_SMPL_CFG = replace(
    SMPL_CFG,
    spawn=replace(SMPL_CFG.spawn, usd_path=args.smpl_usd),
    actuators={
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            armature=0.02,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
    },
)
REFERENCE_SMPL_CFG = replace(
    PASSIVE_SMPL_CFG,
    spawn=replace(
        PASSIVE_SMPL_CFG.spawn,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, linear_damping=0.0, angular_damping=0.0
        ),
    ),
)


@configclass
class FreeFallSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
    )
    robot = replace(PASSIVE_SMPL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    reference = replace(
        REFERENCE_SMPL_CFG, prim_path="{ENV_REGEX_NS}/Reference"
    )


def reset_character(scene, sim) -> None:
    robot = scene["robot"]
    env_ids = torch.arange(scene.num_envs, device=robot.device)
    pose = robot.data.default_root_state.torch[:, :7].clone()
    pose[:, :3] += scene.env_origins
    pose[:, 0] -= 0.8
    velocity = torch.zeros(scene.num_envs, 6, device=robot.device)
    joint_position = torch.zeros_like(robot.data.joint_pos.torch)
    joint_velocity = torch.zeros_like(robot.data.joint_vel.torch)
    robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(
        root_velocity=velocity, env_ids=env_ids
    )
    robot.write_joint_position_to_sim_index(
        position=joint_position, env_ids=env_ids
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=joint_velocity, env_ids=env_ids
    )
    scene.reset(env_ids)
    sim.forward()
    scene.update(0.0)


def write_reference(scene, motion, frame: int, initial_xy) -> None:
    reference = scene["reference"]
    env_ids = torch.arange(scene.num_envs, device=reference.device)
    frames = torch.full(
        (scene.num_envs,), frame, dtype=torch.long, device=reference.device
    )
    p, rotation, velocity, angular_velocity, q, qd = (
        value.clone() for value in motion.ref_state(frames)
    )
    p[:, :2] -= initial_xy
    p[:, 0] += 0.8
    p += scene.env_origins
    quat_wxyz = mat_to_quat(rotation)
    pose = torch.cat((p, quat_wxyz[:, 1:4], quat_wxyz[:, :1]), -1)
    sim_names = tuple(reference.joint_names)
    formal_to_sim = torch.tensor(
        [sim_names.index(name) for name in FORMAL_SIM_JOINT_NAMES],
        dtype=torch.long,
        device=reference.device,
    )
    sim_to_formal = torch.argsort(formal_to_sim)
    reference.write_root_link_pose_to_sim_index(
        root_pose=pose, env_ids=env_ids
    )
    reference.write_root_link_velocity_to_sim_index(
        root_velocity=torch.cat((velocity, angular_velocity), -1),
        env_ids=env_ids,
    )
    reference.write_joint_position_to_sim_index(
        position=q[:, sim_to_formal], env_ids=env_ids
    )
    reference.write_joint_velocity_to_sim_index(
        velocity=qd[:, sim_to_formal], env_ids=env_ids
    )


def main() -> None:
    sim_cfg = make_sim(args.device or "cuda:0")
    with launch_simulation(sim_cfg, args), torch.no_grad():
        device = str(args.device or "cuda:0")
        names = bonesseed_smpl.read_clip_list(args.clips)
        if args.motion_index >= len(names):
            raise IndexError(
                f"motion index {args.motion_index} is outside [0, {len(names)})"
            )
        clips = bonesseed_smpl.load_dir(
            args.motion_dir, names=[names[args.motion_index]]
        )
        motion = MotionLib(
            clips,
            kinematics(args.smpl_mjcf, device),
            device,
            cache_body_state=False,
        )
        initial_xy = motion.root_p[0, :2].clone()
        sim = SimulationContext(sim_cfg)
        scene = instantiate(
            FreeFallSceneCfg(
                num_envs=1, env_spacing=0.0, replicate_physics=True
            )
        )
        sim.reset()
        robot = scene["robot"]
        print("native SMPL free-fall asset:", args.smpl_usd, flush=True)
        print("reference motion:", motion.names[0], flush=True)
        print("joint names:", robot.joint_names, flush=True)
        print("body names:", robot.body_names, flush=True)

        steps_per_fall = max(1, round(args.fall_seconds / CONTROL_DT))
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            reset_character(scene, sim)
            root_start = float(robot.data.root_link_pos_w.torch[0, 2])
            for step in range(steps_per_fall):
                wall_start = time.perf_counter()
                reference_frame = step % int(motion.clip_len[0])
                write_reference(
                    scene, motion, reference_frame, initial_xy
                )
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(CONTROL_DT)
                if step % 25 == 0 or step + 1 == steps_per_fall:
                    root_z = float(robot.data.root_link_pos_w.torch[0, 2])
                    body_min_z = float(
                        robot.data.body_link_pos_w.torch[0, :, 2].min()
                    )
                    print(
                        f"cycle={cycle} t={step * CONTROL_DT:.2f}s "
                        f"root_z={root_z:.4f} min_body_z={body_min_z:.4f}",
                        flush=True,
                    )
                if not args.no_realtime:
                    elapsed = time.perf_counter() - wall_start
                    time.sleep(max(0.0, CONTROL_DT - elapsed))
            root_end = float(robot.data.root_link_pos_w.torch[0, 2])
            print(
                f"cycle={cycle} root_drop={root_start - root_end:.4f}m",
                flush=True,
            )
            cycle += 1


if __name__ == "__main__":
    main()
