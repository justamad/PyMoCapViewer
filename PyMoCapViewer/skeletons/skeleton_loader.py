from .vicon_plug_in_gait import vicon_skeleton
from .kinect_v2 import kinect_skeleton
from .azure_kinect import azure_skeleton

from collections import OrderedDict

import pandas as pd

skeleton_definitions = {
    'vicon': vicon_skeleton,
    'azure': azure_skeleton,
    'kinect_v2': kinect_skeleton,
}


def get_skeleton_definition_for_camera(data_frame: pd.DataFrame, camera_name: str):
    if camera_name not in skeleton_definitions:
        raise ValueError(f"Camera {camera_name} is not known.")

    joint_definition = skeleton_definitions[camera_name]
    joint_names = get_joints_as_list(data_frame)

    skeleton = []
    for j1, j2 in joint_definition:
        if j1 not in joint_names or j2 not in joint_names:
            continue
        skeleton.append((joint_names.index(j1), joint_names.index(j2)))

    return skeleton


def get_joints_as_list(df: pd.DataFrame):
    columns = [column.replace(column[column.find(" ("):column.find(")") + 1], "") for column in df.columns]
    return list(OrderedDict.fromkeys(columns))
