"""Character-sized state and simulator-context operations.

The original LAFAN/G1 path keeps its module-level constants in
``ndm_gym.state`` for checkpoint compatibility.  Non-G1 tracking tasks use
this codec so the shared SuperMimic learner never needs to guess tensor
boundaries from a robot name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core.rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat


@dataclass(frozen=True)
class ArticulationStateCodec:
    num_joints: int

    def __post_init__(self) -> None:
        if self.num_joints < 1:
            raise ValueError("num_joints must be positive")

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def invariant_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def context_dim(self) -> int:
        return 4 + self.num_joints

    def flatten(self, state) -> torch.Tensor:
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        return torch.cat(
            (p, mat_to_quat(rotation), linear_velocity, angular_velocity, q, qd),
            -1,
        )

    def unflatten(self, value: torch.Tensor):
        if value.shape[-1] != self.state_dim:
            raise ValueError(
                f"flattened state has dimension {value.shape[-1]}, "
                f"expected {self.state_dim}"
            )
        n = self.num_joints
        return (
            value[..., 0:3],
            quat_to_mat(value[..., 3:7]),
            value[..., 7:10],
            value[..., 10:13],
            value[..., 13 : 13 + n],
            value[..., 13 + n : 13 + 2 * n],
        )

    def invariant_features(self, state) -> torch.Tensor:
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                p[..., 2:3],
                mat_to_6d(heading.transpose(-1, -2) @ rotation),
                (rotation_t @ linear_velocity.unsqueeze(-1)).squeeze(-1),
                (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1),
                q,
                qd,
            ),
            -1,
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context has dimension {context.shape[-1]}, "
                f"expected {self.context_dim}"
            )
        n = self.num_joints
        return torch.cat(
            (
                (context[..., :1] - 0.75) / 0.45,
                context[..., 1 : 1 + n] / 0.01,
                context[..., 1 + n :] / 0.05,
            ),
            -1,
        )

