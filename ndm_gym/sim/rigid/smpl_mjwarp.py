"""Batched SMPL-character simulator for NDM-Gym SuperMimic training."""

from __future__ import annotations

import os

from .. import require_simulator

require_simulator(__name__)

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from mujoco_warp._src.types import OverflowType

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from ...core.rotations import mat_to_quat, quat_to_mat, so3_exp
from ...robots.smpl_character import (
    SMPL_BODY_NAMES,
    SMPL_CHARACTER_SPEC,
)
from .reset_stats import bin_ema, mix_failures, previous_bins


CONTROL_DT = 0.02
DEFAULT_USD_PATH = (
    "/ALG/zhouyangchen/human_template/character/ndm_smpl/usd/ndm_smpl/ndm_smpl.usda"
)
SMPL_USD_PATH = os.environ.get("SMPL_CHARACTER_USD", DEFAULT_USD_PATH)
FORMAL_SIM_JOINT_NAMES = tuple(
    f"{body}_x:{axis}" for body in SMPL_BODY_NAMES[1:] for axis in range(3)
)


def _smpl_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Reproduce the authoritative MJCF's passive joint gains.

    IsaacLab's official MJCF-to-USD importer preserves the geometry, inertias,
    limits and armatures, but does not translate MuJoCo's joint stiffness and
    damping into IsaacLab actuator gains.  Keep those values explicit at the
    simulator boundary instead of editing the generated USD.
    """

    groups = {
        "legs": (
            ("L_Hip", "L_Knee", "L_Ankle", "R_Hip", "R_Knee", "R_Ankle"),
            800.0,
            80.0,
        ),
        "toes_and_upper_limbs": (
            (
                "L_Toe",
                "R_Toe",
                "Neck",
                "Head",
                "L_Thorax",
                "L_Shoulder",
                "L_Elbow",
                "R_Thorax",
                "R_Shoulder",
                "R_Elbow",
            ),
            500.0,
            50.0,
        ),
        "trunk": (("Torso", "Spine", "Chest"), 1000.0, 100.0),
        "wrists_and_hands": (
            ("L_Wrist", "L_Hand", "R_Wrist", "R_Hand"),
            300.0,
            30.0,
        ),
    }
    actuators = {}
    for name, (bodies, stiffness, damping) in groups.items():
        joints = [rf"^{body}_x:[0-2]$" for body in bodies]
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=joints,
            stiffness=stiffness,
            damping=damping,
            armature=0.02,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
    return actuators


SMPL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SMPL_USD_PATH,
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
    actuators=_smpl_actuators(),
)


@configclass
class SMPLTrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
    )
    robot = replace(SMPL_CFG, prim_path="{ENV_REGEX_NS}/Robot")


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
    n = SMPL_CHARACTER_SPEC.num_joints
    if value.shape[-1] != 13 + 2 * n:
        raise ValueError("flattened SMPL-character state has the wrong dimension")
    return (
        value[..., :3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13 : 13 + n],
        value[..., 13 + n :],
    )


class SMPLTrackingSim:
    """Real Newton/MuJoCo-Warp collection for the 69-DoF SMPL character."""

    def __init__(
        self,
        sim_cfg,
        num_envs,
        motion,
        mdp,
        *,
        gamma=0.99,
        max_episode_steps=500,
        reset_noise=True,
        fail_reset=0.0,
        bin_fix=False,
        eval_mode=False,
        usd_path=DEFAULT_USD_PATH,
    ):
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(
            SMPLTrackingSceneCfg(
                num_envs=num_envs,
                env_spacing=0.0,
                replicate_physics=True,
                robot=replace(
                    SMPL_CFG,
                    prim_path="{ENV_REGEX_NS}/Robot",
                    spawn=replace(SMPL_CFG.spawn, usd_path=str(usd_path)),
                ),
            )
        )
        self.sim.reset()
        # Hitting MuJoCo-Warp's Newton/line-search iteration cap is recorded in
        # the per-world overflow bits and the step uses the best iterate, just
        # as native MuJoCo does.  Printing one warning per affected world from
        # a GPU kernel creates tens of gigabytes of duplicate logs in a large
        # training run, so disable only those two print bits.  The limits,
        # solver path and recorded overflow state remain unchanged.
        solver = self.sim.backend_manager._solver
        mjw_model = getattr(solver, "mjw_model", None)
        if mjw_model is not None:
            mjw_model.opt.warn_overflow = int(mjw_model.opt.warn_overflow) & ~int(
                OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
            )
        self.robot = self.scene["robot"]
        sim_names = tuple(self.robot.joint_names)
        if set(sim_names) != set(FORMAL_SIM_JOINT_NAMES):
            raise ValueError("SMPL USD and MJCF expose different DoF names")
        # NDM tensors retain the authoritative MJCF's body and XYZ order.  The
        # importer collapses each body's three hinge joints into one D6 joint
        # named ``<body>_x`` with simulator axes 0, 1 and 2.
        self.formal_to_sim = torch.tensor(
            [sim_names.index(name) for name in FORMAL_SIM_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.sim_to_formal = torch.argsort(self.formal_to_sim)
        if set(self.robot.body_names) != set(SMPL_BODY_NAMES):
            raise ValueError("SMPL USD and MJCF expose different bodies")
        self.num_envs = self.N = int(num_envs)
        self.motion, self.mdp, self.gamma = motion, mdp, float(gamma)
        # Match the standard NDM TrackingSim contract: formal evaluation plays
        # a complete clip, while training retains its configured episode cap.
        self.max_episode_steps = (
            int(max_episode_steps) if not eval_mode else 1_000_000_000
        )
        self.reset_noise = bool(reset_noise)
        self.default_q = torch.zeros(
            SMPL_CHARACTER_SPEC.num_joints, device=self.device
        )
        self.xi = torch.zeros(
            self.num_envs, SMPL_CHARACTER_SPEC.context_dim, device=self.device
        )
        # This backend currently keeps deterministic nominal dynamics.  The
        # context is still explicit so the shared learner and checkpoint have
        # the same contract as the standard SuperMimic path.
        self.xi[:, 0] = 1.0
        self.f = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.ep_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.a_prev = torch.zeros(
            self.num_envs, SMPL_CHARACTER_SPEC.action_dim, device=self.device
        )
        self.start_bin = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.ep_disc_ret = torch.zeros(self.num_envs, device=self.device)
        self.ep_ret = torch.zeros(self.num_envs, device=self.device)
        self.G_max = mdp.max_step_reward / (1 - self.gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=self.device)
        self.bin_w_avg = torch.full(
            (motion.num_bins,), 1.0 / motion.num_bins, device=self.device
        )
        if not 0 <= fail_reset < 1:
            raise ValueError("fail_reset must be in [0, 1)")
        self.fail_reset, self.bin_fix = float(fail_reset), bool(bin_fix)
        if self.fail_reset > 0:
            self.bin_fail = torch.zeros(motion.num_bins, device=self.device)
            self.bin_prev = previous_bins(motion)
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
        """Paper Appendix D.1 return-based reset sampling over one-second bins."""

        motion = self.motion
        # A physics divergence is handled as a terminated episode in step(),
        # but keep the adaptive reset distribution defensive as well: one
        # invalid return must never make multinomial sampling unusable.
        safe_bin_G = torch.nan_to_num(
            self.bin_G, nan=0.0, posinf=self.G_max, neginf=-self.G_max
        )
        soft = torch.softmax(-25.0 * safe_bin_G / self.G_max, 0)
        self.bin_w_avg.copy_(
            torch.nan_to_num(self.bin_w_avg, nan=0.0, posinf=0.0, neginf=0.0)
            .clamp_min_(0.0)
        )
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probability = (
            0.5 / motion.num_bins
            + 0.4 * soft
            + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        )
        if self.fail_reset > 0:
            probability = mix_failures(
                probability, self.bin_fail, self.bin_prev, self.fail_reset
            )
        probability = torch.nan_to_num(
            probability, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min_(0.0)
        total = probability.sum()
        probability = torch.where(
            total > 0,
            probability / total.clamp_min(1e-12),
            torch.full_like(probability, 1.0 / motion.num_bins),
        )
        bins = torch.multinomial(probability, count, replacement=True)
        start = motion.bin_start[bins]
        last = motion.clip_end[motion.bin_clip[bins]]
        end = torch.minimum(start + 50, last)
        frame = start + (
            torch.rand(count, device=self.device) * (end - start).float()
        ).long()
        return torch.minimum(frame, last - 1), bins

    def reset(self, env_ids, frames=None):
        if len(env_ids) == 0:
            return
        if frames is None:
            frames, bins = self.sample_frames(len(env_ids))
        else:
            bins = self.motion.frame_bin[frames]
        p, rotation, v, w, q, qd = (
            value.clone() for value in self.motion.ref_state(frames)
        )
        if self.reset_noise:
            n = len(env_ids)
            p = p + torch.stack(
                (
                    torch.empty(n, device=self.device).uniform_(-0.05, 0.05),
                    torch.empty(n, device=self.device).uniform_(-0.05, 0.05),
                    torch.empty(n, device=self.device).uniform_(-0.01, 0.01),
                ),
                -1,
            )
            rotation = so3_exp(
                torch.stack(
                    (
                        torch.empty(n, device=self.device).uniform_(-0.1, 0.1),
                        torch.empty(n, device=self.device).uniform_(-0.1, 0.1),
                        torch.empty(n, device=self.device).uniform_(-0.2, 0.2),
                    ),
                    -1,
                )
            ) @ rotation
            v = v + torch.empty_like(v).uniform_(-0.2, 0.2)
            w = w + torch.empty_like(w).uniform_(-0.52, 0.52)
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
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
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.scene.reset(env_ids)

    def refresh(self):
        self.sim.forward()
        self.scene.update(0.0)

    @torch.no_grad()
    def step(self, action, state=None):
        state = self.state() if state is None else state
        frames = self.f.clone()
        previous = self.a_prev.clone()
        context = self.xi.clone()
        episode_step = self.ep_step.clone()
        action = torch.clamp(action, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.robot.set_joint_position_target_index(
            target=action[:, self.sim_to_formal]
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        next_state = self.state()
        next_frames = frames + 1
        healthy = self.mdp.healthy_state(next_state)
        # Reward/FK and replay see a safe fallback for diverged worlds.  The
        # guard includes finite-but-exploded states because MuJoCo-Warp can
        # return very large finite values for one step before NaN/Inf appears.
        # corresponding transition is still marked terminated below, reset in
        # this same control step, and contributes zero reward.
        safe_state = tuple(torch.nan_to_num(value) for value in state)
        safe_next_state = tuple(
            torch.where(
                healthy.view((-1,) + (1,) * (value.ndim - 1)),
                value,
                fallback,
            )
            for value, fallback in zip(next_state, safe_state)
        )
        reward, terms = self.mdp.reward(
            safe_next_state, next_frames, action, previous, terms=True
        )
        reward = torch.where(healthy & reward.isfinite(), reward, 0.0)
        terms = {
            name: torch.where(healthy & value.isfinite(), value, 0.0)
            for name, value in terms.items()
        }
        terminated = self.mdp.termination(next_state, next_frames)
        clip_end = self.motion.clip_end[self.motion.clip_id[frames]]
        timed_out = (next_frames >= clip_end) | (
            episode_step + 1 >= self.max_episode_steps
        )
        done = terminated | timed_out
        transition = {
            "s": flatten_state(safe_state),
            "s1": flatten_state(safe_next_state),
            "a": action.clone(),
            "a_prev": previous,
            "f": frames,
            "r": reward,
            "done": done,
            "term": terminated,
            "ep_step": episode_step,
            "xi": context,
            "terms": terms,
        }
        self.ep_disc_ret += (self.gamma ** episode_step.float()) * reward
        self.ep_ret += reward
        self.f.copy_(next_frames)
        self.ep_step.copy_(episode_step + 1)
        self.a_prev.copy_(action)
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            length = self.ep_step[ids].float()
            bootstrap = torch.where(
                terminated[ids],
                0.0,
                (self.gamma**length)
                * (self.ep_ret[ids] / length)
                / (1 - self.gamma),
            )
            returns = self.ep_disc_ret[ids] + bootstrap
            returns = torch.where(returns.isfinite(), returns, 0.0)
            bins = self.start_bin[ids]
            if self.bin_fix:
                self.bin_G.copy_(bin_ema(self.bin_G, bins, returns, 0.8))
            else:
                self.bin_G.index_put_(
                    (bins,), 0.8 * self.bin_G[bins] + 0.2 * returns
                )
            if self.fail_reset > 0:
                failed_bins = self.motion.frame_bin[frames[ids]]
                self.bin_fail.mul_(0.999)
                if self.bin_fix:
                    self.bin_fail.copy_(
                        bin_ema(
                            self.bin_fail,
                            failed_bins,
                            terminated[ids].float(),
                            0.9,
                        )
                    )
                else:
                    self.bin_fail.index_put_(
                        (failed_bins,),
                        0.9 * self.bin_fail[failed_bins]
                        + 0.1 * terminated[ids].float(),
                    )
            transition["ep_len"] = length
            transition["ep_ret_norm"] = self.ep_ret[ids] / (
                length * self.mdp.max_step_reward
            )
            self.reset(ids)
        return transition
