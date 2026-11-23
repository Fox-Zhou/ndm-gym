from types import SimpleNamespace

import torch

from ndm_gym.algos.supermimic import SuperMimic
from ndm_gym.policies.tracking_policy import TrackingPolicy
from ndm_gym.robots.spec import ArticulationSpec
from ndm_gym.robots.smpl_character import SMPL_CHARACTER_SPEC
from ndm_gym.tasks.articulation_tracking import ArticulationTrackingMDP
from ndm_gym.worldmodels.articulation_increment import ArticulationIncrementWorldModel


def test_smpl_viewer_history_uses_training_reset_padding_and_sliding_window():
    from ndm_gym.smpl_viewer import _advance_history

    first = torch.tensor([[1.0, 2.0]])
    history = _advance_history(None, first, 3)
    torch.testing.assert_close(history, first.unsqueeze(1).repeat(1, 3, 1))

    second = torch.tensor([[3.0, 4.0]])
    history = _advance_history(history, second, 3)
    torch.testing.assert_close(
        history,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]]),
    )


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

    def __init__(self):
        self.forward_calls = 0

    def forward(self, p, rotation, v, w, q, qd):
        self.forward_calls += 1
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

    def ref_body_state(self, frames, state=None):
        state = self.ref_state(frames) if state is None else state
        p, rotation, v, w = state[:4]
        batch = p.shape[:-1]
        return (
            p.unsqueeze(-2).expand(*batch, 2, 3),
            rotation.unsqueeze(-3).expand(*batch, 2, 3, 3),
            v.unsqueeze(-2).expand(*batch, 2, 3),
            w.unsqueeze(-2).expand(*batch, 2, 3),
        )

    def ref_body_positions(self, frames, state=None):
        return self.ref_body_state(frames, state)[0]

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


def test_reference_body_cache_is_used_instead_of_recomputed():
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip

    frames = torch.tensor([0, 2, 4])
    root_p = torch.zeros(6, 3)
    root_p[:, 2] = 1.0
    clip = Clip(
        name="reference_cache",
        root_p=root_p,
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(6, -1).clone(),
        joints=torch.zeros(6, 3),
        fps=50.0,
    )

    cached_kin = _Kinematics()
    cached = MotionLib([clip], cached_kin, "cpu", cache_body_state=True)
    cached_kin.forward_calls = 0
    cached_state = cached.ref_body_state(frames)
    assert cached_kin.forward_calls == 0
    assert cached.body_p is not None

    dynamic_kin = _Kinematics()
    dynamic = MotionLib([clip], dynamic_kin, "cpu", cache_body_state=False)
    dynamic_kin.forward_calls = 0
    dynamic_state = dynamic.ref_body_state(frames)
    assert dynamic_kin.forward_calls == 1
    assert dynamic.body_p is None

    for cached_value, dynamic_value in zip(cached_state, dynamic_state):
        torch.testing.assert_close(cached_value, dynamic_value, rtol=0, atol=0)

    spec = ArticulationSpec(
        joint_names=("j0", "j1", "j2"),
        tracked_bodies=("root", "foot"),
        foot_bodies=("foot",),
        root_body="root",
    )
    action = torch.zeros(len(frames), spec.action_dim)
    context = torch.zeros(len(frames), spec.context_dim)
    cached_mdp = ArticulationTrackingMDP(spec, cached_kin, cached, torch.zeros(3))
    dynamic_mdp = ArticulationTrackingMDP(spec, dynamic_kin, dynamic, torch.zeros(3))

    cached_kin.forward_calls = 0
    cached_reward = cached_mdp.reward(cached.ref_state(frames), frames, action, action)
    cached_critic = cached_mdp.critic_features(
        cached.ref_state(frames), frames, context
    )
    assert cached_kin.forward_calls == 2  # Current state only, reward + critic.

    dynamic_kin.forward_calls = 0
    dynamic_reward = dynamic_mdp.reward(
        dynamic.ref_state(frames), frames, action, action
    )
    dynamic_critic = dynamic_mdp.critic_features(
        dynamic.ref_state(frames), frames, context
    )
    assert dynamic_kin.forward_calls == 4  # Current and reference, both calls.
    torch.testing.assert_close(cached_reward, dynamic_reward, rtol=0, atol=0)
    torch.testing.assert_close(cached_critic, dynamic_critic, rtol=0, atol=0)


def test_articulation_termination_catches_nonfinite_and_exploded_physics_state():
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
    frames = torch.tensor([0, 1, 2])
    state = list(reference.ref_state(frames))
    state[4] = state[4].clone()
    state[4][1, 0] = torch.nan
    state[0] = state[0].clone()
    state[0][2, 0] = 2.0e3

    assert mdp.termination(tuple(state), frames).tolist() == [False, True, True]


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


def test_generic_character_residual_setpoint_uses_reference_and_limits(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    spec = ArticulationSpec(
        joint_names=("j0", "j1", "j2"),
        tracked_bodies=("root", "foot"),
        foot_bodies=("foot",),
        root_body="root",
    )
    reference = _Reference()
    reference.q[1] = torch.tensor((0.25, -0.5, 0.75))
    mdp = ArticulationTrackingMDP(
        spec, _Kinematics(), reference, torch.zeros(3)
    )
    args = _args()
    args.sm_residual = True
    learner = SuperMimic(args, mdp.kin, reference, mdp, "cpu")

    assert torch.equal(mdp.action_scale, torch.ones(3))
    frames = torch.tensor((0, 0))
    offsets = torch.tensor(((0.1, 0.2, -0.3), (2.0, -2.0, 2.0)))
    expected = torch.clamp(offsets + reference.q[1], -1.0, 1.0)
    torch.testing.assert_close(learner.setpoint(offsets, frames), expected)


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

