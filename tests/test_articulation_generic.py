import torch

from ndm_gym.policies.tracking_policy import TrackingPolicy
from ndm_gym.robots.phc_smpl import PHC_SMPL_SPEC
from ndm_gym.worldmodels.articulation_increment import ArticulationIncrementWorldModel


def test_phc_policy_and_world_model_are_dimensionally_generic():
    spec = PHC_SMPL_SPEC
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

