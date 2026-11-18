"""The fixed 69-DoF PHC SMPL character used by SuperMotionFix."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from ..core.kinematics import ArticulationKinematics
from .spec import ArticulationSpec


PHC_BODY_NAMES = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
    "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow",
    "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow",
    "R_Wrist", "R_Hand",
)
PHC_JOINT_NAMES = tuple(
    f"{body}_{axis}" for body in PHC_BODY_NAMES[1:] for axis in "xyz"
)
PHC_SMPL_SPEC = ArticulationSpec(
    joint_names=PHC_JOINT_NAMES,
    tracked_bodies=PHC_BODY_NAMES,
    foot_bodies=("L_Toe", "R_Toe"),
    root_body="Pelvis",
)


def validate_mjcf(path: str | Path) -> None:
    root = ET.parse(Path(path)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("PHC SMPL MJCF has no worldbody")
    bodies = tuple(body.get("name") for body in worldbody.iter("body"))
    joints = tuple(
        joint.get("name")
        for body in worldbody.iter("body")
        for joint in body.findall("joint")
        if joint.get("name") is not None
    )
    if bodies != PHC_BODY_NAMES:
        raise ValueError("PHC SMPL MJCF body order differs from the formal spec")
    if joints != PHC_JOINT_NAMES:
        raise ValueError("PHC SMPL MJCF hinge order differs from the formal spec")


def kinematics(mjcf_path: str | Path, device: str) -> ArticulationKinematics:
    validate_mjcf(mjcf_path)
    return ArticulationKinematics(
        str(mjcf_path),
        list(PHC_JOINT_NAMES),
        list(PHC_BODY_NAMES),
        device,
    )

