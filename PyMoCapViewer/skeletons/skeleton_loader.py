from .vicon_plug_in_gait import vicon_skeleton
from .kinect_v2 import kinect_skeleton
from .azure_kinect import azure_skeleton
from .vnect import vnect_skeleton
from .ghum import ghum_skeleton

from collections import OrderedDict

import pandas as pd
import logging

skeleton_definitions = {
    "vicon": vicon_skeleton,
    "azure": azure_skeleton,
    "kinect_v2": kinect_skeleton,
    "vnect": vnect_skeleton,
    "ghum": ghum_skeleton,
}


def get_skeleton_definition_for_camera(
        df: pd.DataFrame,
        camera_name: str,
        camera_count: int,
):
    """ Get the skeleton definition of current skeleton """
    if camera_name not in skeleton_definitions:
        raise ValueError(f"Unknown camera given: {camera_name}")

    joint_definition = skeleton_definitions[camera_name]
    joint_definition = list(map(lambda x: list(map(str.lower, x)), joint_definition))

    joint_names = get_joints_as_list(df)

    skeleton = []
    for j1, j2 in joint_definition:
        if j1 not in joint_names or j2 not in joint_names:
            logging.warning(f"Camera {camera_count}: Could not find joints: {j1} and/or {j2}.")
            continue

        skeleton.append((joint_names.index(j1), joint_names.index(j2)))

    return skeleton


def get_joints_as_list(df: pd.DataFrame) -> list:
    """ Returns list of joints by removing axis """
    columns = [column.replace(column[column.find(" ("):column.find(")") + 1], "") for column in df.columns]
    joints = list(OrderedDict.fromkeys(columns))
    return list(map(str.lower, joints))
