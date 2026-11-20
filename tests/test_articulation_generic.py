from types import SimpleNamespace

import torch

from ndm_gym.algos.supermimic import SuperMimic
from ndm_gym.policies.tracking_policy import TrackingPolicy
from ndm_gym.robots.spec import ArticulationSpec
from ndm_gym.robots.smpl_character import SMPL_CHARACTER_SPEC
from ndm_gym.tasks.articulation_tracking import ArticulationTrackingMDP
from ndm_gym.worldmodels.articulation_increment import ArticulationIncrementWorldModel


def test_smpl_policy_and_world_model_are_dimensionally_generic():
    spec = SMPL_CHARACTER_SPEC
    policy = TrackingPolicy(
        spec.observation_dim,
        spec.goal_frame_dim * 10,
        hist=10,
        action_dim=spec.action_dim,
    )
    action = policy(
        torch.zeros(2, 10, spec.observation_dim),
        torch.zeros(2, 10 * spec.goal_frame_dim),
    )
    assert action.shape == (2, 69)

    world_model = ArticulationIncrementWorldModel(spec, hidden=(32, 32))
    p = torch.zeros(2, 3)
    R = torch.eye(3).expand(2, -1, -1)
    state = (p, R, p, p, torch.zeros(2, 69), torch.zeros(2, 69))
    next_state = world_model.step(
        state, action.detach(), torch.zeros(2, spec.context_dim)
    )
    assert next_state[0].shape == (2, 3)
    assert next_state[4].shape == (2, 69)


class _Kinematics:
    q_lo = torch.full((3,), -1.0)
    q_hi = torch.full((3,), 1.0)

    def forward(self, p, rotation, v, w, q, qd):
        batch = p.shape[:-1]
        body_p = p.unsqueeze(-2).expand(*batch, 2, 3)
        body_rotation = rotation.unsqueeze(-3).expand(*batch, 2, 3, 3)
        body_v = v.unsqueeze(-2).expand(*batch, 2, 3)
        body_w = w.unsqueeze(-2).expand(*batch, 2, 3)
        return body_p, body_rotation, body_v, body_w


class _Reference:
    goal_frame_dim = 21

    def __init__(self):
        self.root_p = torch.zeros(128, 3)
        self.root_p[:, 2] = 1.0
        self.root_R = torch.eye(3).expand(128, -1, -1).clone()
        self.root_v = torch.zeros(128, 3)
        self.root_w = torch.zeros(128, 3)
        self.q = torch.zeros(128, 3)
        self.qd = torch.zeros(128, 3)
        self.clip_id = torch.zeros(128, dtype=torch.long)
        self.clip_end = torch.tensor([127])

    def ref_state(self, frames):
        return (
            self.root_p[frames],
            self.root_R[frames],
            self.root_v[frames],
            self.root_w[frames],
            self.q[frames],
            self.qd[frames],
        )

    def goal(self, frames, count):
        return torch.zeros(len(frames), count * self.goal_frame_dim)


def _args():
    return SimpleNamespace(
        goal_frames=2,
        hist=2,
        horizon=2,
        wm_hidden=[32, 32],
        wm_act="silu",
        wm_no_ln=False,
        wm_backbone="mlp",
        lr=2e-4,
        no_critic=False,
        critic_cudagraphs=False,
        sm_residual=False,
        contact_reward=0.0,
        ref_contact_fix=False,
        fused_adam=False,
        update_graphs=False,
        no_sync=False,
        pi_ema=0.0,
        wm_mirror=0.0,
        wm_structured=False,
        wm_cudagraphs=False,
        pi_cudagraphs=False,
        wm_deferred_grads=False,
        deferred_bf16=False,
        wm_smooth_w=0.0,
        bptt_decay=1.0,
        pi_rollout_clip=0.0,
        compile_eager=False,
        wm_window=2,
        wm_batch=4,
        pi_batch=4,
        gamma=0.99,
        lam=0.95,
        critic_epochs=1,
        critic_minibatches=1,
        wm_input_noise=0.0,
        wm_sample="uniform",
        wm_prio_alpha=0.6,
        imag_fail_starts=0.0,
        imag_fail_window=8,
        imag_start_perturb=0.0,
        wm_fail_windows=0.0,
        wm_fail_window=8,
        bptt_clip_k=0.0,
        wm_bptt_fp32=False,
        pi_encode_once=False,
        imag_hoist=False,
        pi_deferred_grads=False,
    )


def test_official_supermimic_builds_critic_for_generic_character(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    spec = ArticulationSpec(
        joint_names=("j0", "j1", "j2"),
        tracked_bodies=("root", "foot"),
        foot_bodies=("foot",),
        root_body="root",
    )
    reference = _Reference()
    mdp = ArticulationTrackingMDP(
        spec, _Kinematics(), reference, torch.zeros(3)
    )
    learner = SuperMimic(_args(), mdp.kin, reference, mdp, "cpu")
    assert learner.pi.dec[-1].out_features == spec.action_dim
    assert learner.critic.net[0].in_features == (
        mdp.critic_feature_dim + reference.goal_frame_dim * 2
    )


def test_official_supermimic_runs_full_generic_updates(monkeypatch):
    from ndm_gym.buffer import ReplayBuffer, articulation_fields

    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    spec = ArticulationSpec(
        joint_names=("j0", "j1", "j2"),
        tracked_bodies=("root", "foot"),
        foot_bodies=("foot",),
        root_body="root",
    )
    reference = _Reference()
    mdp = ArticulationTrackingMDP(
        spec, _Kinematics(), reference, torch.zeros(3)
    )
    learner = SuperMimic(_args(), mdp.kin, reference, mdp, "cpu")
    replay = ReplayBuffer(
        8, 4, "cpu", fields=articulation_fields(spec.num_joints), priorities=False
    )
    codec = mdp.state_codec
    context = torch.zeros(4, spec.context_dim)
    context[:, 0] = 0.75
    for step in range(8):
        frames = torch.arange(4) + step
        state = reference.ref_state(frames)
        next_state = reference.ref_state(frames + 1)
        action = torch.zeros(4, spec.action_dim)
        replay.add(
            {
                "s": codec.flatten(state),
                "s1": codec.flatten(next_state),
                "a": action,
                "a_prev": action,
                "xi": context,
                "r": torch.zeros(4),
                "f": frames,
                "ep_step": torch.full((4,), step, dtype=torch.int32),
                "done": torch.zeros(4, dtype=torch.bool),
            }
        )
        learner.update_wm_norms(state, next_state, action, context)
        learner.update_pi_norms(
            mdp.obs(state, action, context, frames),
            reference.goal(frames, learner.K),
        )
    wm_loss, wm_error = learner.wm_update(replay)
    policy_loss, value_loss, reward, grad_norm, survival = learner.policy_update(
        replay
    )
    for value in (
        wm_loss,
        wm_error,
        policy_loss,
        value_loss,
        reward,
        grad_norm,
        survival,
    ):
        assert torch.isfinite(value).all()

