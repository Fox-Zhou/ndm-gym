"""Character-parameterized SuperMimic increment world model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, so3_exp, so3_log, yaw_mat
from ..robots.spec import ArticulationSpec


def invariant_features(state):
    p, R, v, w, q, qd = state
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


class ArticulationIncrementWorldModel(nn.Module):
    """NDM increment model with dimensions derived from an ArticulationSpec."""

    def __init__(
        self,
        spec: ArticulationSpec,
        *,
        hidden=(2048, 2048, 2048),
        act="silu",
        layer_norm=True,
    ):
        super().__init__()
        self.spec = spec
        self.invariant_dim = 13 + 2 * spec.num_joints
        self.output_dim = 12 + 2 * spec.num_joints
        self.input_dim = self.invariant_dim + spec.action_dim + spec.context_dim
        self.net = mlp(
            self.input_dim,
            list(hidden),
            self.output_dim,
            act=ACTIVATIONS[act],
            layer_norm=layer_norm,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.input_dim)
        self.out_norm = RunningNorm(self.output_dim, eps=1.0e-4)
        self.amp = True

    def inputs(self, state, action, normalized_context):
        if action.shape[-1] != self.spec.action_dim:
            raise ValueError("action dimension does not match character spec")
        if normalized_context.shape[-1] != self.spec.context_dim:
            raise ValueError("context dimension does not match character spec")
        return torch.cat(
            (invariant_features(state), action, normalized_context), dim=-1
        )

    @staticmethod
    def targets(state, next_state):
        p, R, v, w, q, qd = state
        p1, R1, v1, w1, q1, qd1 = next_state
        H, Rt = yaw_mat(R), R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(
        self,
        state,
        action,
        normalized_context,
        input_noise: float = 0.0,
        taps=None,
    ):
        if taps is not None:
            raise ValueError(
                "deferred world-model gradients are not enabled for the "
                "character-generic MLP"
            )
        p, R, v, w, q, qd = state
        x = self.in_norm(self.inputs(state, action, normalized_context))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        y = run_mlp(self.net, x, amp=self.amp)
        delta = y * self.out_norm.std() + self.out_norm.mean
        n = self.spec.num_joints
        dp, dtheta, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, n, n), -1)
        H = yaw_mat(R)
        return (
            p + (H @ dp.unsqueeze(-1)).squeeze(-1),
            R @ so3_exp(dtheta),
            v + (R @ dv.unsqueeze(-1)).squeeze(-1),
            w + (R @ dw.unsqueeze(-1)).squeeze(-1),
            q + dq,
            qd + dqd,
        )

    @staticmethod
    def state_error(predicted, target):
        """Frame-local state error with the same layout as ``targets``."""

        p, R, v, w, q, qd = target
        ph, Rh, vh, wh, qh, qdh = predicted
        heading = yaw_mat(R)
        rotation_t = R.transpose(-1, -2)
        return torch.cat(
            (
                (
                    heading.transpose(-1, -2)
                    @ (ph - p).unsqueeze(-1)
                ).squeeze(-1),
                so3_log(rotation_t @ Rh),
                (rotation_t @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )

