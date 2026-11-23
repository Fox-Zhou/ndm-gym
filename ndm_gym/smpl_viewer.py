"""Interactive Viser viewer for the SMPL SuperMimic character.

The viewer never advances physics itself.  :class:`SMPLSuperMimicRuntime`
steps the same ``SMPLTrackingSim`` (Newton with the MuJoCo-Warp solver) used by
training, while this module only converts its states to Viser bone transforms.
The learned world-model rollout and the reference motion are visual overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
import torch

from .algos import ALGOS
from .algos.common import reference_setpoints
from .core.rotations import mat_to_quat
from .policies.tracking_policy import TrackingPolicy


@dataclass(frozen=True)
class StepStats:
    """Last real-simulator transition shown by the viewer."""

    reward: float = 0.0
    tracking_score: float = 0.0
    mpjpe: float = 0.0
    terminated: bool = False
    timed_out: bool = False


def _clone_state(state):
    return tuple(value.clone() for value in state)


def _advance_history(history: torch.Tensor | None, observation: torch.Tensor, length: int):
    if length == 1:
        return observation.unsqueeze(1)
    if history is None:
        return observation.unsqueeze(1).repeat(1, length, 1)
    return torch.cat((history[:, 1:], observation.unsqueeze(1)), 1)


class SMPLSuperMimicRuntime:
    """One-world policy, world-model and real-simulator playback runtime.

    The real character always comes from ``env.step``.  The blue character is
    a closed-loop rollout through the checkpoint's learned world model.  Both
    use the same deployed policy and reference frame sequence.
    """

    def __init__(self, env, kin, motion, mdp, checkpoint: dict, wm_rollout_steps: int = 100):
        from .core import nets

        if checkpoint.get("format") != "ndm-gym-supermimic-smpl-v1":
            raise ValueError("the viewer needs an ndm-gym-supermimic-smpl-v1 checkpoint")
        schema = checkpoint.get("character", {})
        if tuple(schema.get("joint_names", ())) != tuple(mdp.spec.joint_names):
            raise ValueError("checkpoint and runtime SMPL joint order differ")
        if tuple(schema.get("tracked_bodies", ())) != tuple(mdp.spec.tracked_bodies):
            raise ValueError("checkpoint and runtime SMPL body order differ")
        if wm_rollout_steps < 1:
            raise ValueError("wm_rollout_steps must be positive")

        self.env, self.kin, self.motion, self.mdp = env, kin, motion, mdp
        self.device = env.device
        self.args = checkpoint["args"]
        self.K = int(self.args["goal_frames"])
        self.history_length = int(self.args["hist"])
        self.wm_rollout_steps = int(wm_rollout_steps)
        self.residual = ALGOS[self.args["algo"]].is_residual(self.args)

        # run_mlp reads this module-global precision setting in exactly the same
        # way as training/evaluate.py.
        nets.AMP_DTYPE = (
            torch.bfloat16 if self.args.get("amp") == "bf16" else None
        )
        self.policy = TrackingPolicy(
            mdp.observation_dim,
            motion.goal_frame_dim * self.K,
            hist=self.history_length,
            action_dim=mdp.action_dim,
        ).to(self.device)
        self.policy.load_state_dict(checkpoint["pi"])
        self.policy.eval()

        self.world_model = mdp.make_world_model(self.args).to(self.device)
        self.world_model.load_state_dict(checkpoint["wm"])
        self.world_model.eval()
        self.normalized_context = mdp.normalize_context(env.xi)
        self.reference_setpoints = (
            reference_setpoints(motion, mdp.default_q, mdp.action_scale)
            if self.residual
            else None
        )

        self.clip_index = 0
        self.real_history = None
        self.wm_history = None
        self.wm_state = None
        self.wm_previous_action = None
        self.wm_step_count = 0
        self.stats = StepStats()
        self.checkpoint_iteration = int(checkpoint.get("iteration", -1))

    def _setpoint(self, policy_output: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        if self.residual:
            target = policy_output + self.reference_setpoints[frames + 1]
        else:
            target = policy_output
        return self.mdp.clamp_action(target)

    def reset(self, clip_index: int | None = None, frame_offset: int = 0) -> None:
        if clip_index is not None:
            self.clip_index = max(0, min(int(clip_index), len(self.motion.names) - 1))
        length = int(self.motion.clip_len[self.clip_index].item())
        offset = max(0, min(int(frame_offset), max(0, length - 2)))
        frame = self.motion.clip_start[self.clip_index] + offset
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self.env.reset(ids, frame.reshape(1))
        self.env.refresh()
        self.real_history = None
        self.wm_history = None
        self.wm_state = _clone_state(self.env.state())
        self.wm_previous_action = self.env.a_prev.clone()
        self.wm_step_count = 0
        self.stats = StepStats()

    @property
    def frame_offset(self) -> int:
        return int((self.env.f[0] - self.motion.clip_start[self.clip_index]).item())

    @property
    def clip_length(self) -> int:
        return int(self.motion.clip_len[self.clip_index].item())

    def _reset_world_model_from_real(self) -> None:
        self.wm_state = _clone_state(self.env.state())
        self.wm_previous_action = self.env.a_prev.clone()
        self.wm_history = None
        self.wm_step_count = 0

    @torch.no_grad()
    def step(self) -> StepStats:
        frames = self.env.f.clone()
        real_state = self.env.state()
        real_obs = self.mdp.obs(real_state, self.env.a_prev, self.env.xi, frames)
        self.real_history = _advance_history(
            self.real_history, real_obs, self.history_length
        )
        goal = self.motion.goal(frames, self.K)
        real_action = self._setpoint(self.policy(self.real_history, goal), frames)

        if self.wm_state is None:
            self._reset_world_model_from_real()
        wm_obs = self.mdp.obs(
            self.wm_state, self.wm_previous_action, self.env.xi, frames
        )
        self.wm_history = _advance_history(
            self.wm_history, wm_obs, self.history_length
        )
        wm_action = self._setpoint(self.policy(self.wm_history, goal), frames)
        next_wm_state = self.world_model.step(
            self.wm_state, wm_action, self.normalized_context
        )

        transition = self.env.step(real_action, real_state)
        self.wm_state = next_wm_state
        self.wm_previous_action = wm_action
        self.wm_step_count += 1

        next_real_state = self.mdp.unflatten_state(transition["s1"])
        next_frames = transition["f"] + 1
        real_body = self.kin.forward(*next_real_state)[0]
        reference_body = self.motion.ref_body_positions(next_frames)
        mpjpe = (real_body - reference_body).norm(dim=-1).mean(-1)
        tracking_score = transition["terms"]["track"] / self.mdp.max_step_reward
        terminated = bool(transition["term"][0].item())
        done = bool(transition["done"][0].item())
        self.stats = StepStats(
            reward=float(transition["r"][0].item()),
            tracking_score=float(tracking_score[0].item()),
            mpjpe=float(mpjpe[0].item()),
            terminated=terminated,
            timed_out=done and not terminated,
        )

        if done:
            # SMPLTrackingSim performs its standard training reset internally.
            # Replace that random reset before rendering so viewer playback
            # remains on the user-selected clip.
            last_stats = self.stats
            self.reset(self.clip_index, 0)
            self.stats = last_stats
        elif self.wm_step_count >= self.wm_rollout_steps or not bool(
            torch.stack([value.isfinite().all() for value in self.wm_state])
            .all()
            .item()
        ):
            self._reset_world_model_from_real()
        return self.stats

    @torch.no_grad()
    def body_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World-space body positions and quaternions for all three overlays."""

        frames = self.env.f.clone()
        real = self.kin.forward(*self.env.state())[:2]
        reference = self.motion.ref_body_state(frames)[:2]
        predicted = self.kin.forward(*self.wm_state)[:2]

        def numpy_pose(values):
            positions, rotations = values
            quaternions = mat_to_quat(rotations)
            return (
                positions[0].detach().cpu().numpy().astype(np.float32),
                quaternions[0].detach().cpu().numpy().astype(np.float32),
            )

        return {
            "simulation": numpy_pose(real),
            "reference": numpy_pose(reference),
            "world_model": numpy_pose(predicted),
        }


def _primitive_mesh(model, geom_id: int):
    """Create one MuJoCo primitive in its geom-local coordinate system."""

    import mujoco
    import trimesh

    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return trimesh.creation.capsule(
            height=2.0 * float(size[1]), radius=float(size[0]), count=[12, 12]
        )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1]), sections=16
        )
    raise ValueError(
        f"SMPL Viser supports box/capsule/sphere/cylinder geoms, got type {geom_type}"
    )


class SkinnedMjcfCharacter:
    """A whole MJCF character represented by one efficient Viser skinned mesh."""

    def __init__(
        self,
        server,
        path: str,
        body_names: tuple[str, ...],
        *,
        scene_name: str,
        color: tuple[int, int, int],
        opacity: float,
    ):
        import mujoco
        model = mujoco.MjModel.from_xml_path(path)
        body_ids = [int(model.body(name).id) for name in body_names]
        body_slot = {body_id: index for index, body_id in enumerate(body_ids)}
        vertices, faces, vertex_bones = [], [], []
        vertex_offset = 0
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id not in body_slot:
                continue
            primitive = _primitive_mesh(model, geom_id)
            transform = np.eye(4, dtype=np.float64)
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            transform[:3, :3] = rotation.reshape(3, 3)
            transform[:3, 3] = model.geom_pos[geom_id]
            primitive.apply_transform(transform)
            vertices.append(np.asarray(primitive.vertices, dtype=np.float32))
            faces.append(np.asarray(primitive.faces, dtype=np.int32) + vertex_offset)
            vertex_bones.extend([body_slot[body_id]] * len(primitive.vertices))
            vertex_offset += len(primitive.vertices)
        if not vertices:
            raise ValueError(f"no renderable SMPL geoms found in {path}")

        all_vertices = np.concatenate(vertices)
        all_faces = np.concatenate(faces)
        weights = np.zeros((len(all_vertices), len(body_names)), dtype=np.float32)
        weights[np.arange(len(all_vertices)), np.asarray(vertex_bones)] = 1.0
        identity = np.zeros((len(body_names), 4), dtype=np.float32)
        identity[:, 0] = 1.0
        zero = np.zeros((len(body_names), 3), dtype=np.float32)
        self.handle = server.scene.add_mesh_skinned(
            scene_name,
            vertices=all_vertices,
            faces=all_faces,
            bone_wxyzs=identity,
            bone_positions=zero,
            skin_weights=weights,
            color=color,
            opacity=opacity,
            material="standard",
            flat_shading=False,
            side="double",
        )

    @property
    def visible(self) -> bool:
        return bool(self.handle.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.handle.visible = bool(value)

    def update(self, positions: np.ndarray, wxyzs: np.ndarray) -> None:
        if len(self.handle.bones) != len(positions):
            raise ValueError("viewer pose and skinned character have different body counts")
        for bone, position, quaternion in zip(self.handle.bones, positions, wxyzs):
            bone.position = position
            bone.wxyz = quaternion


class SMPLSuperMimicViser:
    """Browser controls and rendering for :class:`SMPLSuperMimicRuntime`."""

    COLORS = {
        "simulation": (205, 145, 85),
        "reference": (75, 200, 125),
        "world_model": (95, 135, 230),
    }

    def __init__(
        self,
        runtime: SMPLSuperMimicRuntime,
        mjcf_path: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        fps: float = 50.0,
        autoplay: bool = False,
    ):
        import viser

        self.runtime = runtime
        self.server = viser.ViserServer(host=host, port=port, label="NDM SMPL SuperMimic")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid(
            "/ground",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=0.5,
            section_size=2.0,
            infinite_grid=False,
            shadow_opacity=0.15,
        )
        names = tuple(runtime.mdp.spec.tracked_bodies)
        self.characters = {
            "simulation": SkinnedMjcfCharacter(
                self.server,
                mjcf_path,
                names,
                scene_name="/characters/simulation",
                color=self.COLORS["simulation"],
                opacity=0.98,
            ),
            "reference": SkinnedMjcfCharacter(
                self.server,
                mjcf_path,
                names,
                scene_name="/characters/reference",
                color=self.COLORS["reference"],
                opacity=0.28,
            ),
            "world_model": SkinnedMjcfCharacter(
                self.server,
                mjcf_path,
                names,
                scene_name="/characters/world_model",
                color=self.COLORS["world_model"],
                opacity=0.48,
            ),
        }
        self.lock = threading.RLock()
        self.pending_reset: tuple[int, int] | None = None
        self.pending_steps = 0
        self._setup_gui(float(fps), autoplay=autoplay)
        self._setup_camera()

    def _setup_gui(self, fps: float, *, autoplay: bool) -> None:
        with self.server.gui.add_folder("Simulation"):
            self.playing = self.server.gui.add_checkbox(
                "Play", initial_value=bool(autoplay)
            )
            self.step_button = self.server.gui.add_button("Step")
            self.reset_button = self.server.gui.add_button("Reset selected motion")
            self.fps = self.server.gui.add_slider(
                "Simulation steps / second",
                min=1,
                max=100,
                step=1,
                initial_value=max(1, min(100, int(round(fps)))),
            )
        with self.server.gui.add_folder("Motion"):
            self.motion_number = self.server.gui.add_number(
                "Motion ID",
                min=0,
                max=len(self.runtime.motion.names) - 1,
                step=1,
                initial_value=self.runtime.clip_index,
            )
            self.frame = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, self.runtime.clip_length - 1),
                step=1,
                initial_value=0,
            )
            self.motion_info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Overlays"):
            self.show_sim = self.server.gui.add_checkbox("Simulation", initial_value=True)
            self.show_ref = self.server.gui.add_checkbox("Reference", initial_value=True)
            self.show_wm = self.server.gui.add_checkbox("World model", initial_value=True)
            self.follow_camera = self.server.gui.add_checkbox("Follow root", initial_value=True)
        with self.server.gui.add_folder("Metrics"):
            self.status = self.server.gui.add_markdown("")

        @self.step_button.on_click
        def _(_) -> None:
            with self.lock:
                self.pending_steps += 1

        @self.reset_button.on_click
        def _(_) -> None:
            with self.lock:
                self.pending_reset = (self.runtime.clip_index, 0)

        @self.motion_number.on_update
        def _(event) -> None:
            if event.client_id is None:
                return
            index = max(
                0,
                min(
                    int(round(float(self.motion_number.value))),
                    len(self.runtime.motion.names) - 1,
                ),
            )
            with self.lock:
                self.pending_reset = (index, 0)

        @self.frame.on_update
        def _(event) -> None:
            if event.client_id is None:
                return
            with self.lock:
                self.pending_reset = (
                    self.runtime.clip_index,
                    int(round(float(self.frame.value))),
                )

        def bind_visible(control, name: str) -> None:
            @control.on_update
            def _(_) -> None:
                self.characters[name].visible = bool(control.value)

        bind_visible(self.show_sim, "simulation")
        bind_visible(self.show_ref, "reference")
        bind_visible(self.show_wm, "world_model")

    def _setup_camera(self) -> None:
        self.camera_roots: dict[Any, np.ndarray] = {}

        @self.server.on_client_connect
        def _(client) -> None:
            root = self.runtime.env.state()[0][0].detach().cpu().numpy()
            client.camera.position = root + np.asarray((-4.0, -4.0, 2.4))
            client.camera.look_at = root + np.asarray((0.0, 0.0, 0.9))
            client.camera.up_direction = (0.0, 0.0, 1.0)
            self.camera_roots[client.client_id] = root.copy()

    def _update_cameras(self, root: np.ndarray) -> None:
        if not bool(self.follow_camera.value):
            self.camera_roots.clear()
            return
        for client_id, client in self.server.get_clients().items():
            previous = self.camera_roots.get(client_id)
            if previous is None:
                self.camera_roots[client_id] = root.copy()
                continue
            delta = root - previous
            if np.any(np.abs(delta) > 1e-8):
                client.camera.position = np.asarray(client.camera.position) + delta
                client.camera.look_at = np.asarray(client.camera.look_at) + delta
            self.camera_roots[client_id] = root.copy()

    def _update_text(self) -> None:
        runtime, stats = self.runtime, self.runtime.stats
        name = runtime.motion.names[runtime.clip_index]
        self.motion_info.content = (
            f"**{runtime.clip_index} / {len(runtime.motion.names) - 1}** · `{name}`  \n"
            f"Frame **{runtime.frame_offset} / {runtime.clip_length - 1}** · "
            f"{runtime.frame_offset / 50.0:.2f} s"
        )
        self.status.content = (
            f"Checkpoint **{runtime.checkpoint_iteration:,}**  \n"
            f"Reward `{stats.reward:.5f}` · tracking `{stats.tracking_score:.4f}`  \n"
            f"MPJPE `{stats.mpjpe:.4f} m` · "
            f"terminated `{'yes' if stats.terminated else 'no'}`"
        )

    def render(self, *, update_text: bool = True) -> None:
        poses = self.runtime.body_poses()
        with self.server.atomic():
            for name, character in self.characters.items():
                character.update(*poses[name])
            self.frame.max = max(1, self.runtime.clip_length - 1)
            self.frame.value = self.runtime.frame_offset
            if update_text:
                self._update_text()
        self._update_cameras(poses["simulation"][0][0])

    def _process_actions(self) -> tuple[bool, bool]:
        with self.lock:
            reset = self.pending_reset
            self.pending_reset = None
            single_step = self.pending_steps > 0
            if single_step:
                self.pending_steps -= 1
        if reset is not None:
            self.runtime.reset(*reset)
            self.motion_number.value = self.runtime.clip_index
            self.render()
        return single_step, reset is not None

    def run(self) -> None:
        self.runtime.reset(self.runtime.clip_index, 0)
        self.render()
        next_tick = time.monotonic()
        next_text = next_tick
        try:
            while True:
                single_step, did_reset = self._process_actions()
                if not bool(self.playing.value) and not single_step:
                    time.sleep(0.01)
                    continue
                interval = 1.0 / max(float(self.fps.value), 1.0)
                now = time.monotonic()
                if did_reset and not single_step:
                    next_tick = now + interval
                    continue
                if not single_step and now < next_tick:
                    time.sleep(min(next_tick - now, 0.01))
                    continue
                self.runtime.step()
                update_text = now >= next_text
                self.render(update_text=update_text)
                if update_text:
                    next_text = now + 0.2
                # Every physics frame is shown in order.  If simulation or
                # rendering is slower than requested, playback slows down
                # instead of skipping states.
                next_tick = max(next_tick + interval, time.monotonic())
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
