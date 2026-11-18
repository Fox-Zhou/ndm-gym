"""Character dimensions shared by non-G1 NDM tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArticulationSpec:
    """Names and derived tensor dimensions of a floating-base character."""

    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    foot_bodies: tuple[str, ...]
    root_body: str

    def __post_init__(self):
        if not self.joint_names:
            raise ValueError("an articulation needs at least one actuated joint")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be unique")
        if not self.tracked_bodies or self.root_body not in self.tracked_bodies:
            raise ValueError("tracked bodies must contain the root body")
        if not set(self.foot_bodies).issubset(self.tracked_bodies):
            raise ValueError("foot bodies must be tracked bodies")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_tracked_bodies(self) -> int:
        return len(self.tracked_bodies)

    @property
    def state_dim(self) -> int:
        # root p, quaternion, linear/angular velocity, q and qd
        return 13 + 2 * self.num_joints

    @property
    def invariant_state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def context_dim(self) -> int:
        # friction, per-joint calibration offset, root-body COM offset
        return 4 + self.num_joints

    @property
    def observation_dim(self) -> int:
        # q, qd, root angular velocity, projected gravity, previous action,
        # reference-relative root rotation (6D)
        return 3 * self.num_joints + 12

    @property
    def goal_frame_dim(self) -> int:
        # relative root p/rotation, root v/w, q and qd
        return 15 + 2 * self.num_joints

    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def critic_extra_dim(self) -> int:
        return (
            self.num_joints
            + 3 * self.num_tracked_bodies
            + 3
            + 6
            + self.context_dim
        )

