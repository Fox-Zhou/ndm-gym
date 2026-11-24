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


import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from ndm_gym.sim.rigid.smpl_mjwarp import CONTROL_DT, SMPL_CFG, make_sim


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


@configclass
class FreeFallSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
    )
    robot = replace(PASSIVE_SMPL_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def reset_character(scene, sim) -> None:
    robot = scene["robot"]
    env_ids = torch.arange(scene.num_envs, device=robot.device)
    pose = robot.data.default_root_state.torch[:, :7].clone()
    pose[:, :3] += scene.env_origins
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


def main() -> None:
    sim_cfg = make_sim(args.device or "cuda:0")
    with launch_simulation(sim_cfg, args), torch.no_grad():
        sim = SimulationContext(sim_cfg)
        scene = instantiate(
            FreeFallSceneCfg(
                num_envs=1, env_spacing=0.0, replicate_physics=True
            )
        )
        sim.reset()
        robot = scene["robot"]
        print("native SMPL free-fall asset:", args.smpl_usd, flush=True)
        print("joint names:", robot.joint_names, flush=True)
        print("body names:", robot.body_names, flush=True)

        steps_per_fall = max(1, round(args.fall_seconds / CONTROL_DT))
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            reset_character(scene, sim)
            root_start = float(robot.data.root_link_pos_w.torch[0, 2])
            for step in range(steps_per_fall):
                wall_start = time.perf_counter()
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
