from scipy.spatial.transform import Rotation as R
from typing import Union

import pandas as pd
import numpy as np


def create_xy_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [x_coord * cell_size - offset, y_coord * cell_size - offset, 0]
    p2 = [(x_coord * cell_size + cell_size) - offset, y_coord * cell_size - offset, 0]
    p3 = [(x_coord * cell_size + cell_size) - offset, (y_coord * cell_size + cell_size) - offset, 0]
    p4 = [x_coord * cell_size - offset, (y_coord * cell_size + cell_size) - offset, 0]
    return p1, p2, p3, p4


def create_yz_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [0, x_coord * cell_size - offset, y_coord * cell_size - offset]
    p2 = [0, (x_coord * cell_size + cell_size) - offset, y_coord * cell_size - offset]
    p3 = [0, (x_coord * cell_size + cell_size) - offset, (y_coord * cell_size + cell_size) - offset]
    p4 = [0, x_coord * cell_size - offset, (y_coord * cell_size + cell_size) - offset]
    return p1, p2, p3, p4


def create_xz_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [x_coord * cell_size - offset, 0, y_coord * cell_size - offset]
    p2 = [(x_coord * cell_size + cell_size) - offset, 0, y_coord * cell_size - offset]
    p3 = [(x_coord * cell_size + cell_size) - offset, 0, (y_coord * cell_size + cell_size) - offset]
    p4 = [x_coord * cell_size - offset, 0, (y_coord * cell_size + cell_size) - offset]
    return p1, p2, p3, p4


def create_orientations_from_quaternions(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if data.shape[1] % 4 != 0:
        raise ValueError("The number of columns should be divisible by 4.")

    rotation_frames = []
    for frame in range(data.shape[0]):
        joint_rotation = []
        for joint in range(data.shape[1] // 4):
            w, x, y, z = np.array(data[frame, joint * 4:joint * 4 + 4])
            rot = R.from_quat([x, y, z, w])
            joint_rotation.append(rot.as_matrix())

        rotation_frames.append(joint_rotation)

    return np.array(rotation_frames).reshape((data.shape[0], data.shape[1] // 4, 3, 3))


def create_orientations_from_euler_angles(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if data.shape[1] % 3 != 0:
        raise ValueError("The number of columns should be divisible by 3.")

    rotation_frames = []
    for frame in range(data.shape[0]):
        joint_rotation = []
        for joint in range(data.shape[1] // 3):
            rotation = R.from_euler("xyz", data[frame, joint * 3: (joint + 1) * 3])
            joint_rotation.append(rotation.as_matrix())

        rotation_frames.append(joint_rotation)

    return np.array(rotation_frames).reshape((data.shape[0], data.shape[1] // 3, 3, 3))
