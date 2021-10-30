from .vicon_plug_in_gait import skeleton
from .kinect_v2 import skeleton
from .azure_kinect import skeleton

skeleton_definitions = {
    'vicon': skeleton,
    'azure': skeleton,
    'kinect_v2': skeleton,
}


def get_skeleton_definition_for_camera(camera_name: str):
    if camera_name not in skeleton_definitions:
        raise ValueError(f"Camera {camera_name} is not known.")

    return skeleton_definitions[camera_name]
