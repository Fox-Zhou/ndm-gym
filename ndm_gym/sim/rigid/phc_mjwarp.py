"""Minimal batched PHC tracking simulator for Stage-A SuperMimic training."""

from __future__ import annotations

import os

from .. import require_simulator

require_simulator(__name__)

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from ...core.rotations import mat_to_quat, quat_to_mat
from ...robots.phc_smpl import PHC_BODY_NAMES, PHC_SMPL_SPEC


CONTROL_DT = 0.02
DEFAULT_USD_PATH = (
    "/ALG/zhouyangchen/human_template/character/phc_smpl/phc_smpl.usda"
)
PHC_USD_PATH = os.environ.get("PHC_SMPL_USD", DEFAULT_USD_PATH)
FORMAL_SIM_JOINT_NAMES = tuple(
    f"{body}:{axis}" for body in PHC_BODY_NAMES[1:] for axis in range(3)
)


PHC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PHC_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
    ),
    soft_joint_pos_limit_factor=0.9,
    # None keeps the authoritative gains, damping, armature and limits stored
    # in the PHC USD while exposing all DoFs to IsaacLab's target API.
    actuators={
        "phc": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            armature=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
    },
)


@configclass
class PHCTrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
    )
    robot = replace(PHC_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def make_sim(device: str):
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=2000,
            nconmax=256,
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=True,
        ),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics)


def flatten_state(state):
    p, rotation, linear_velocity, angular_velocity, q, qd = state
    return torch.cat(
        (p, mat_to_quat(rotation), linear_velocity, angular_velocity, q, qd), -1
    )


def unflatten_state(value):
    n = PHC_SMPL_SPEC.num_joints
    if value.shape[-1] != 13 + 2 * n:
        raise ValueError("flattened PHC state has the wrong dimension")
    return (
        value[..., :3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13 : 13 + n],
        value[..., 13 + n :],
    )


class PHCTrackingSim:
    """Real Newton/MuJoCo-Warp collection for the 69-DoF PHC character."""

    def __init__(self, sim_cfg, num_envs, motion, mdp, *, gamma=0.99):
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(
            PHCTrackingSceneCfg(
                num_envs=num_envs, env_spacing=0.0, replicate_physics=True
            )
        )
        self.sim.reset()
        self.robot = self.scene["robot"]
        sim_names = tuple(self.robot.joint_names)
        if set(sim_names) != set(FORMAL_SIM_JOINT_NAMES):
            raise ValueError("PHC USD and formal MJCF expose different DoF names")
        # PHC's legacy USD places Neck/Head after the left arm whereas the
        # generated MJCF places them before both arms.  NDM tensors always use
        # the MJCF order; only this simulator boundary reorders them.
        self.formal_to_sim = torch.tensor(
            [sim_names.index(name) for name in FORMAL_SIM_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.sim_to_formal = torch.argsort(self.formal_to_sim)
        if set(self.robot.body_names) != set(PHC_BODY_NAMES):
            raise ValueError("PHC USD and formal MJCF expose different bodies")
        self.num_envs = int(num_envs)
        self.motion, self.mdp, self.gamma = motion, mdp, float(gamma)
        self.default_q = torch.zeros(
            PHC_SMPL_SPEC.num_joints, device=self.device
        )
        self.context = torch.zeros(
            self.num_envs, PHC_SMPL_SPEC.context_dim, device=self.device
        )
        self.frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.previous_action = torch.zeros(
            self.num_envs, PHC_SMPL_SPEC.action_dim, device=self.device
        )
        self.reset(torch.arange(self.num_envs, device=self.device))
        self.refresh()

    def state(self):
        data = self.robot.data
        pose = data.root_link_pose_w.torch
        velocity = data.root_link_vel_w.torch.clone()
        quat_xyzw = pose[:, 3:7]
        rotation = quat_to_mat(
            torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), -1)
        )
        return (
            pose[:, :3] - self.scene.env_origins,
            rotation,
            velocity[:, :3],
            velocity[:, 3:6],
            data.joint_pos.torch[:, self.formal_to_sim].clone(),
            data.joint_vel.torch[:, self.formal_to_sim].clone(),
        )

    def sample_frames(self, count: int):
        clip = torch.randint(
            0, len(self.motion.clip_len), (count,), device=self.device
        )
        start = self.motion.clip_start[clip]
        available = (self.motion.clip_len[clip] - 1).clamp_min(1)
        return start + (torch.rand(count, device=self.device) * available).long()

    def reset(self, env_ids, frames=None):
        if len(env_ids) == 0:
            return
        if frames is None:
            frames = self.sample_frames(len(env_ids))
        p, rotation, v, w, q, qd = (
            value.clone() for value in self.motion.ref_state(frames)
        )
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat_wxyz = mat_to_quat(rotation)
        pose = torch.cat(
            (
                p + self.scene.env_origins[env_ids],
                quat_wxyz[:, 1:4],
                quat_wxyz[:, :1],
            ),
            -1,
        )
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=pose, env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((v, w), -1), env_ids=env_ids
        )
        q_sim = q[:, self.sim_to_formal]
        qd_sim = qd[:, self.sim_to_formal]
        self.robot.write_joint_position_to_sim_index(
            position=q_sim, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=qd_sim, env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(target=q_sim, env_ids=env_ids)
        self.frames[env_ids] = frames
        self.episode_step[env_ids] = 0
        self.previous_action[env_ids] = q
        self.scene.reset(env_ids)

    def refresh(self):
        self.sim.forward()
        self.scene.update(0.0)

    @torch.no_grad()
    def step(self, action):
        state = self.state()
        frames = self.frames.clone()
        previous = self.previous_action.clone()
        action = torch.clamp(action, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.robot.set_joint_position_target_index(
            target=action[:, self.sim_to_formal]
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        next_state = self.state()
        next_frames = frames + 1
        reward, terms = self.mdp.reward(
            next_state, next_frames, action, previous, terms=True
        )
        terminated = self.mdp.termination(next_state, next_frames)
        clip_end = self.motion.clip_end[self.motion.clip_id[frames]]
        timed_out = next_frames >= clip_end
        done = terminated | timed_out
        transition = {
            "s": flatten_state(state),
            "s1": flatten_state(next_state),
            "a": action.clone(),
            "a_prev": previous,
            "f": frames,
            "r": reward,
            "done": done,
            "term": terminated,
            "context": self.context.clone(),
            "terms": terms,
        }
        self.frames.copy_(next_frames)
        self.episode_step.add_(1)
        self.previous_action.copy_(action)
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self.reset(ids)
        return transition
