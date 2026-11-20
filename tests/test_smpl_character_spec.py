from pathlib import Path
import os

import pytest
import torch

from ndm_gym.robots.smpl_character import (
    SMPL_CHARACTER_SPEC,
    kinematics,
    validate_mjcf,
)


ASSET = Path(
    os.environ.get(
        "SMPL_CHARACTER_MJCF",
        "/ALG/zhouyangchen/human_template/character/phc_smpl/phc_smpl.xml",
    )
)


def test_smpl_dimensions_are_derived_from_the_spec():
    spec = SMPL_CHARACTER_SPEC
    assert spec.num_joints == 69
    assert spec.num_tracked_bodies == 24
    assert spec.state_dim == 151
    assert spec.observation_dim == 219
    assert spec.goal_frame_dim == 153
    assert spec.action_dim == 69


@pytest.mark.skipif(not ASSET.exists(), reason="shared SMPL asset is not mounted")
def test_smpl_mjcf_and_multi_hinge_fk_load():
    validate_mjcf(ASSET)
    kin = kinematics(ASSET, "cpu")
    assert kin.has_multi_hinge_bodies
    assert all(len(indices) == 3 for indices in kin.joint_indices[1:])
    n = SMPL_CHARACTER_SPEC.num_joints
    p = torch.zeros(2, 3)
    R = torch.eye(3).expand(2, -1, -1)
    v = torch.zeros_like(p)
    q = torch.zeros(2, n)
    out = kin.forward(p, R, v, v, q, q, all_bodies=True)
    assert out[0].shape == (2, 24, 3)
    assert out[1].shape == (2, 24, 3, 3)


@pytest.mark.skipif(not ASSET.exists(), reason="shared SMPL asset is not mounted")
def test_multi_hinge_fk_matches_mujoco():
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    joint = np.random.default_rng(7).normal(0.0, 0.25, 69)
    data.qpos[:7] = [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = joint
    mujoco.mj_forward(model, data)

    kin = kinematics(ASSET, "cpu")
    p = torch.tensor([[0.1, -0.2, 1.0]])
    R = torch.eye(3).unsqueeze(0)
    zero = torch.zeros(1, 3)
    q = torch.tensor(joint, dtype=torch.float32).unsqueeze(0)
    body_p, body_R, _, _ = kin.forward(
        p, R, zero, zero, q, torch.zeros_like(q), all_bodies=True
    )
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in kin.body_names
    ]
    expected_p = torch.tensor(data.xpos[body_ids], dtype=torch.float32)
    expected_R = torch.tensor(
        data.xmat[body_ids].reshape(-1, 3, 3), dtype=torch.float32
    )
    torch.testing.assert_close(body_p[0], expected_p, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(body_R[0], expected_R, rtol=1e-5, atol=2e-6)

