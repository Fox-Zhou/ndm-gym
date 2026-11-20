"""Character-parameterized tracking reward for dynamic reference batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..core.kinematics import ArticulationKinematics
from ..core.rotations import mat_to_6d, rot_angle_sq, yaw_mat
from ..robots.spec import ArticulationSpec
from ..state_codec import ArticulationStateCodec
from ..worldmodels.articulation_increment import ArticulationIncrementWorldModel


class DynamicReference(Protocol):
    root_p: torch.Tensor
    root_R: torch.Tensor
    root_v: torch.Tensor
    root_w: torch.Tensor
    q: torch.Tensor
    qd: torch.Tensor

    def ref_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


@dataclass
class ArticulationRewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


class ArticulationTrackingMDP:
    """The standard NDM reward without G1 dimensions or static MotionLib."""

    def __init__(
        self,
        spec: ArticulationSpec,
        kin: ArticulationKinematics,
        reference: DynamicReference,
        default_q: torch.Tensor,
        *,
        control_dt: float = 0.02,
        cfg: ArticulationRewardCfg | None = None,
    ):
        if default_q.shape != (spec.num_joints,):
            raise ValueError("default_q does not match character spec")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        self.spec, self.kin, self.reference = spec, kin, reference
        self.default_q, self.control_dt = default_q, float(control_dt)
        self.cfg = cfg or ArticulationRewardCfg()
        self.state_codec = ArticulationStateCodec(spec.num_joints)
        self.unflatten_state = self.state_codec.unflatten
        self.normalize_context = self.state_codec.normalize_context
        self.invariant_state_dim = self.state_codec.invariant_dim
        self.context_dim = self.state_codec.context_dim
        self.observation_dim = spec.observation_dim
        self.goal_frame_dim = spec.goal_frame_dim
        self.action_dim = spec.action_dim
        self.critic_extra_dim = spec.critic_extra_dim
        self.critic_feature_dim = self.invariant_state_dim + self.critic_extra_dim
        self.max_step_reward = 5.0 * self.control_dt
        self.obs_noise = torch.cat(
            (
                torch.full((spec.num_joints,), 0.01, device=default_q.device),
                torch.full((spec.num_joints,), 0.5, device=default_q.device),
                torch.full((3,), 0.2, device=default_q.device),
                torch.full((3,), 0.05, device=default_q.device),
                torch.zeros(spec.num_joints, device=default_q.device),
                torch.full((6,), 0.05, device=default_q.device),
            )
        )
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = mid - half * self.cfg.soft_limit_factor
        self.soft_hi = mid + half * self.cfg.soft_limit_factor

    def observation(self, state, previous_action, frames):
        p, R, _, w, q, qd = state
        if previous_action.shape[-1] != self.spec.action_dim:
            raise ValueError("previous action does not match character")
        reference_root_R = self.reference.ref_state(frames)[1]
        relative = mat_to_6d(R.transpose(-1, -2) @ reference_root_R)
        gravity_body = -R[..., 2, :]
        root_w_body = (
            R.transpose(-1, -2) @ w.unsqueeze(-1)
        ).squeeze(-1)
        result = torch.cat(
            (q - self.default_q, qd, root_w_body, gravity_body, previous_action, relative),
            -1,
        )
        if result.shape[-1] != self.spec.observation_dim:
            raise RuntimeError("derived observation dimension is inconsistent")
        return result

    # The native G1 trainer calls these two names.  Keeping the aliases here
    # lets a character-generic collector use the same training semantics.
    def obs(self, state, previous_action, context, frames):
        del context
        return self.observation(state, previous_action, frames)

    def add_noise(self, observation):
        return observation + (2 * torch.rand_like(observation) - 1) * self.obs_noise

    def reward(
        self,
        state,
        frames,
        action,
        previous_action,
        *,
        terms=False,
        contact=None,
    ):
        if contact is not None:
            raise ValueError("contact reward is not configured for this character")
        c, ref = self.cfg, self.reference
        p, R, v, w, q, qd = state
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR, rv, rw, rq, rqd = ref.ref_state(frames)
        rbp, rbR, rbv, rbw = self.kin.forward(rp, rR, rv, rw, rq, rqd)
        # Match NDM's root-anchored body pose reward while preserving world-space
        # root terms. Reference velocities remain in world coordinates.
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            dR.unsqueeze(-3) @ (rbp - rp.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        ref_bR = dR.unsqueeze(-3) @ rbR
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
        e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        tracking = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        action_rate = ((action - previous_action) ** 2).sum(-1)
        joint_limit = (
            torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)
        ).sum(-1)
        reward = (
            tracking
            + c.w_action_rate * action_rate
            + c.w_joint_limit * joint_limit
        ) * self.control_dt
        if not terms:
            return reward
        return reward, {
            "track": tracking * self.control_dt,
            "body_position_error": e_pos,
            "body_orientation_error": e_ori,
            "body_linear_velocity_error": e_lv,
            "body_angular_velocity_error": e_av,
            "root_position_error": e_rp,
            "root_orientation_error": e_ro,
            "action_rate": action_rate,
            "joint_limit": joint_limit,
        }

    def soft_termination(
        self, state, frames, *, tz=0.02, tt=0.04, margin: float = 1.0
    ):
        p, R = state[0], state[1]
        rp, rR = self.reference.ref_state(frames)[:2]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        height = margin * self.cfg.term_height
        tilt = margin * self.cfg.term_tilt
        alive = (1 - torch.sigmoid((e_z - height) / tz)) * (
            1 - torch.sigmoid((e_tilt - tilt) / tt)
        )
        return 1 - alive

    def termination(self, state, frames):
        p, R = state[0], state[1]
        rp, rR = self.reference.ref_state(frames)[:2]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def critic_features(self, state, frames, normalized_context):
        """SuperMimic critic features with dimensions derived from the spec."""

        ref = self.reference
        p, rotation, _, _, q, _ = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        rp, rR, rv, rw, rq, rqd = ref.ref_state(frames)
        del rv, rw, rqd
        body_position = self.kin.forward(*state)[0]
        rbp = self.kin.forward(rp, rR, *ref.ref_state(frames)[2:])[0]
        delta_rotation = yaw_mat(rotation @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        reference_body = anchor.unsqueeze(-2) + (
            delta_rotation.unsqueeze(-3)
            @ (rbp - rp.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            heading_t.unsqueeze(-3)
            @ (reference_body - body_position).unsqueeze(-1)
        ).squeeze(-1).flatten(-2)
        root_error = (heading_t @ (rp - p).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d(rotation.transpose(-1, -2) @ rR)
        result = torch.cat(
            (
                self.state_codec.invariant_features(state),
                rq - q,
                body_error,
                root_error,
                relative_rotation,
                normalized_context,
            ),
            -1,
        )
        if result.shape[-1] != self.critic_feature_dim:
            raise RuntimeError(
                f"critic feature dimension {result.shape[-1]} != "
                f"{self.critic_feature_dim}"
            )
        return result

    def make_world_model(self, args: dict):
        if args.get("wm_backbone", "mlp") != "mlp":
            raise ValueError("SMPL SuperMimic currently supports the MLP world model")
        return ArticulationIncrementWorldModel(
            self.spec,
            hidden=tuple(args["wm_hidden"]),
            act=args["wm_act"],
            layer_norm=not args["wm_no_ln"],
        )

