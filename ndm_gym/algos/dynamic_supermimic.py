"""Differentiable SuperMimic rollout against a generated dynamic reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DynamicRolloutResult:
    reward: torch.Tensor
    survival: torch.Tensor
    terms: Mapping[str, torch.Tensor]
    final_state: tuple[torch.Tensor, ...]


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze weights while retaining gradients with respect to its inputs."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class DynamicSuperMimicRollout(nn.Module):
    """Frozen tracker + frozen world model, without critic bootstrap.

    This is the Stage-B path used to optimize a generated reference.  Unlike
    native policy training it does not update the policy and deliberately does
    not expose critic gradients to a generator that could exploit value-model
    extrapolation.
    """

    def __init__(
        self,
        policy,
        world_model,
        mdp,
        *,
        goal_frames: int = 10,
        reference_residual: bool = False,
    ):
        super().__init__()
        if goal_frames < 1:
            raise ValueError("goal_frames must be positive")
        self.policy = freeze_module(policy)
        self.world_model = freeze_module(world_model)
        self.mdp = mdp
        self.goal_frames = goal_frames
        self.reference_residual = bool(reference_residual)

    def _setpoint(self, policy_output: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Convert a policy output to the action stored in Stage-A replay.

        A residual SuperMimic policy predicts offsets about the next reference
        pose.  Stage-B references are generated dynamically, so that base must
        be gathered from the differentiable reference on every imagined step;
        it cannot be cached from the Stage-A training MotionLib.
        """

        action = policy_output
        if self.reference_residual:
            reference_q = self.mdp.reference.ref_state(frames + 1)[4]
            action = action + (
                reference_q - self.mdp.default_q
            ) / self.mdp.action_scale
        clamp = getattr(self.mdp, "clamp_action", None)
        return clamp(action) if clamp is not None else action

    def forward(
        self,
        state: tuple[torch.Tensor, ...],
        observation_history: torch.Tensor,
        previous_action: torch.Tensor,
        start_frames: torch.Tensor,
        context: torch.Tensor,
        *,
        horizon: int,
    ) -> DynamicRolloutResult:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        frames = start_frames
        action_prev = previous_action
        rewards, term_values, done_probabilities = [], {}, []
        alive = torch.ones_like(start_frames, dtype=state[0].dtype)
        survivals = []
        for _ in range(horizon):
            goal = self.mdp.reference.goal(frames, self.goal_frames)
            action = self._setpoint(
                self.policy(observation_history, goal), frames
            )
            state = self.world_model.step(state, action, context)
            frames = frames + 1
            reward, terms = self.mdp.reward(
                state, frames, action, action_prev, terms=True
            )
            survivals.append(alive)
            rewards.append(reward)
            for name, value in terms.items():
                term_values.setdefault(name, []).append(value)
            done = self.mdp.soft_termination(state, frames)
            done_probabilities.append(done)
            alive = alive * (1.0 - done)
            observation = self.mdp.observation(state, action, frames)
            observation_history = torch.cat(
                (observation_history[:, 1:], observation.unsqueeze(1)), dim=1
            )
            action_prev = action
        return DynamicRolloutResult(
            reward=torch.stack(rewards, dim=1),
            survival=torch.stack(survivals, dim=1),
            terms={name: torch.stack(values, dim=1) for name, values in term_values.items()},
            final_state=state,
        )

