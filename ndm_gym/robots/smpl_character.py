"""The fixed 69-DoF SMPL character used by SuperMimic.

The external asset was derived from PHC and records that provenance in its
README.  This module is an NDM-Gym character definition and has no PHC runtime
dependency.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from ..core.kinematics import ArticulationKinematics
from .spec import ArticulationSpec


SMPL_BODY_NAMES = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
    "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow",
    "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow",
    "R_Wrist", "R_Hand",
)
SMPL_JOINT_NAMES = tuple(
    f"{body}_{axis}" for body in SMPL_BODY_NAMES[1:] for axis in "xyz"
)
SMPL_CHARACTER_SPEC = ArticulationSpec(
    joint_names=SMPL_JOINT_NAMES,
    tracked_bodies=SMPL_BODY_NAMES,
    foot_bodies=("L_Toe", "R_Toe"),
    root_body="Pelvis",
)


def validate_mjcf(path: str | Path) -> None:
    root = ET.parse(Path(path)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SMPL character MJCF has no worldbody")
    bodies = tuple(body.get("name") for body in worldbody.iter("body"))
    joints = tuple(
        joint.get("name")
        for body in worldbody.iter("body")
        for joint in body.findall("joint")
        if joint.get("name") is not None
    )
    if bodies != SMPL_BODY_NAMES:
        raise ValueError("SMPL character MJCF body order differs from the spec")
    if joints != SMPL_JOINT_NAMES:
        raise ValueError("SMPL character MJCF hinge order differs from the spec")


def kinematics(mjcf_path: str | Path, device: str) -> ArticulationKinematics:
    validate_mjcf(mjcf_path)
    return ArticulationKinematics(
        str(mjcf_path),
        list(SMPL_JOINT_NAMES),
        list(SMPL_BODY_NAMES),
        device,
    )

