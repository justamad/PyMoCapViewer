from .skeletons import get_skeleton_definition_for_camera
from .utils import create_xy_points, create_yz_points, create_xz_points
from typing import List, Tuple, Union
from vtk.util.numpy_support import numpy_to_vtk, get_numpy_array_type, numpy_to_vtkIdTypeArray

import pandas as pd
import numpy as np
import open3d as o3d
import vtk
import logging

COLORS = ["red", "green", "blue"]
units = {
    "mm": 1e-3,
    "cm": 1e-2,
    "dm": 1e-1,
    "m": 1.0,
}

planes = {
    "xy": create_xy_points,
    "yz": create_yz_points,
    "xz": create_xz_points,
}

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("my_logger").addHandler(console)


class MoCapViewer(object):

    def __init__(
            self,
            width: int = 1280,
            height: int = 1024,
            sampling_frequency: int = 30,
            sphere_radius: float = 0.03,
            z_min: float = -10,
            z_max: float = 10,
            point_size: float = 3.0,
            grid_axis: str = "xy",
            bg_color: str = "lightslategray"
    ):
        self.__skeleton_objects = []
        self.__point_cloud_objects = []
        self.__max_frames = float("inf")
        self.__cur_frame = 0
        self.__pause = False
        self.__record = False
        self.__trans_vector = np.array([0, 0, 0])
        self.__scale_factor = 1.0
        self.__sphere_radius = sphere_radius
        self.__axis_scale = 0.3
        self.__video_count = 0
        self.__sampling_frequency = sampling_frequency
        self.__grid_dimensions = 11
        self.__grid_cell_size = 0.6
        self.__z_min = z_min
        self.__z_max = z_max
        self.__point_size = point_size

        self.__colors = vtk.vtkNamedColors()
        self.__renderer = vtk.vtkRenderer()
        self.__renderer.SetBackground(self.__colors.GetColor3d(bg_color))
        self.__renderer.ResetCamera()

        self.__render_window = vtk.vtkRenderWindow()
        self.__render_window.SetSize(width, height)
        self.__render_window.AddRenderer(self.__renderer)

        self.__render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.__render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self.__render_window_interactor.SetRenderWindow(self.__render_window)
        self.__render_window_interactor.Initialize()
        self.__timer_id = self.__render_window_interactor.CreateRepeatingTimer(1000 // self.__sampling_frequency)
        self.__render_window_interactor.AddObserver("KeyPressEvent", self.keypress_callback, 1.0)

        self.__draw_coordinate_axes()

        if grid_axis is not None:
            if grid_axis not in planes:
                raise AttributeError(f"Unknown grid axis given: {grid_axis}. Use 'xy', 'yz', or 'xz' or None")

            self.__grid_creator = planes[grid_axis]
            self.__draw_rectilinear_grid()

    def add_skeleton(
            self,
            data: pd.DataFrame,
            skeleton_orientations: np.ndarray = None,
            skeleton_connection: Union[str, List[Tuple[str, str]], List[Tuple[int, int]]] = None,
            color: str = None,
            unit: str = "mm",
    ):
        columns = data.shape[1]
        if columns % 3 != 0:
            raise ValueError(f"Column-number of dataframe should be a multiple of 3, received {columns}.")

        if skeleton_orientations is not None:
            if len(skeleton_orientations.shape) != 4:
                raise AttributeError("Please pass orientation as np.ndarray with (nr_frames, nr_markers, 3, 3)")

            if len(skeleton_orientations) != len(data):
                raise AttributeError(f"Position and orientation data have different lengths:"
                                     f" {len(skeleton_orientations)} vs {len(data)}.")

        n_markers = columns // 3

        if unit not in units:
            raise ValueError(f"Unknown unit given: {unit}.")

        if isinstance(skeleton_connection, str):
            skeleton_connection = get_skeleton_definition_for_camera(
                df=data,
                camera_name=skeleton_connection,
                camera_count=len(self.__skeleton_objects),
            )

        actors_markers = []  # Each marker has an own actor
        lines = []

        if color is not None:
            if color not in self.__colors.GetColorNames():
                raise ValueError(f"Unknown color given: {color}.")
        else:
            color = COLORS[len(self.__skeleton_objects) % len(COLORS)]

        # Create all instances for all markers
        for marker in range(n_markers):
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(100)
            sphere.SetThetaResolution(100)
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(self.__sphere_radius)
            mapper = vtk.vtkPolyDataMapper()
            mapper.AddInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.__colors.GetColor3d(color))
            self.__renderer.AddActor(actor)
            actors_markers.append(actor)

        if skeleton_connection is not None:
            for _ in skeleton_connection:
                line = self.__create_line_vtk_object(color)
                lines.append(line)

        coordinate_axes = []
        if skeleton_orientations is not None:
            for _ in range(n_markers):
                coordinate_axes.append(self.__create_line_vtk_object("red"))
                coordinate_axes.append(self.__create_line_vtk_object("green"))
                coordinate_axes.append(self.__create_line_vtk_object("blue"))

        self.__skeleton_objects.append({
            "data": data.to_numpy() * units[unit],
            "ori_data": skeleton_orientations * 1e-1,
            "skeleton_definition": skeleton_connection,
            "bone_actors": lines,
            "actors_markers": actors_markers,
            "coordinate_axes": coordinate_axes,
        })
        self.__max_frames = min(self.__max_frames, len(data))

    def add_point_cloud_animation(
            self,
            point_cloud_list: List[o3d.geometry.PointCloud],
            unit: str = "mm",
    ):
        vtk_poly_data = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk_poly_data)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(self.__z_min, self.__z_max)
        mapper.SetScalarVisibility(1)
        vtk_actor = vtk.vtkActor()
        vtk_actor.SetMapper(mapper)
        vtk_actor.GetProperty().SetPointSize(self.__point_size)
        self.__renderer.AddActor(vtk_actor)

        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()
        vtk_poly_data.SetPoints(vtk_points)
        vtk_poly_data.SetVerts(vtk_cells)

        f_point_clouds = []
        for pcd in point_cloud_list:
            point_data = np.asarray(pcd.points) * units[unit]
            color_data = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            n_points = len(point_data)

            if n_points > 50000:
                logging.warning(f"Point cloud has large number of points: {n_points}. Visualization might be slow.")

            f_point_clouds.append({
                "data": numpy_to_vtk(point_data),
                "color": numpy_to_vtk(color_data),
                "n_points": n_points,
            })

        self.__point_cloud_objects.append({
            "point_data": f_point_clouds,
            "vtk_poly_data": vtk_poly_data,
        })
        self.__max_frames = min(self.__max_frames, len(point_cloud_list))

    def __draw_coordinate_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(self.__axis_scale, self.__axis_scale, self.__axis_scale)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.__renderer.AddActor(axes)

    def __draw_rectilinear_grid(self):
        for i in range(self.__grid_dimensions):
            for j in range(self.__grid_dimensions):
                actor = self.__create_single_cell(i, j)
                self.__renderer.AddActor(actor)

    def __create_single_cell(self, x_coord: int, y_coord: int):
        offset = self.__grid_dimensions / 2 * self.__grid_cell_size

        points = vtk.vtkPoints()
        p1, p2, p3, p4 = self.__grid_creator(x_coord, y_coord, self.__grid_cell_size, offset)
        points.InsertNextPoint(p1)
        points.InsertNextPoint(p2)
        points.InsertNextPoint(p3)
        points.InsertNextPoint(p4)

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(5)
        polyLine.GetPointIds().SetId(0, 0)
        polyLine.GetPointIds().SetId(1, 1)
        polyLine.GetPointIds().SetId(2, 2)
        polyLine.GetPointIds().SetId(3, 3)
        polyLine.GetPointIds().SetId(4, 0)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    def show_window(self):
        self.__render_window_interactor.AddObserver('TimerEvent', self._update)
        self.__render_window.Render()
        self.__render_window_interactor.Start()

    def _update(self, iren, event):
        if self.__cur_frame >= self.__max_frames:
            self.__cur_frame = 0

        self._draw_new_frame(self.__cur_frame)
        iren.GetRenderWindow().Render()

        if not self.__pause:
            self.__cur_frame += 1

    def _draw_new_frame(self, index: int = 0):
        for pcd_data in self.__point_cloud_objects:
            vtk_poly_data = pcd_data["vtk_poly_data"]
            pcd_info = pcd_data["point_data"][index]
            pc, n_data, colors = pcd_info["data"], pcd_info["n_points"], pcd_info["color"]

            vtk_points = vtk.vtkPoints()
            vtk_cells = vtk.vtkCellArray()
            vtk_poly_data.SetPoints(vtk_points)
            vtk_poly_data.SetVerts(vtk_cells)
            vtk_points.SetNumberOfPoints(n_data)
            vtk_points.SetData(pc)
            vtk_poly_data.GetPointData().SetScalars(colors)

            for n in range(n_data):
                vtk_cells.InsertNextCell(1)
                vtk_cells.InsertCellPoint(n)

            vtk_cells.Modified()
            vtk_points.Modified()

        for skeleton_data in self.__skeleton_objects:
            data = skeleton_data["data"]
            actors_markers = skeleton_data["actors_markers"]
            points = (data[index].reshape(-1, 3) - self.__trans_vector) / self.__scale_factor

            for marker_idx, actor in enumerate(actors_markers):
                x, y, z = points[marker_idx]
                actor.SetPosition(x, y, z)

            # Update bone connections
            joint_connections = skeleton_data["skeleton_definition"]
            lines = skeleton_data["bone_actors"]
            if joint_connections is not None:
                for line, (j1, j2) in zip(lines, joint_connections):
                    line.SetPoint1(points[j1])
                    line.SetPoint2(points[j2])

            # Update local coordinate axes
            coordinate_axes = skeleton_data["coordinate_axes"]
            ori_data = skeleton_data["ori_data"]
            if ori_data is None:
                continue

            ori_data = ori_data[index]
            for marker_idx, marker_pos in enumerate(points):
                cur_point = points[marker_idx]
                cur_ori = ori_data[marker_idx]
                for axis_idx in range(3):
                    x_axis = cur_ori[:, axis_idx]
                    line_obj = coordinate_axes[marker_idx * 3 + axis_idx]
                    line_obj.SetPoint1(cur_point)
                    line_obj.SetPoint2(cur_point + x_axis)

    def keypress_callback(self, obj, ev):
        key = obj.GetKeySym()
        if key == 'space':
            self.__pause = not self.__pause
        elif key == 'Left':
            new_frame = self.__cur_frame - 1
            self.__cur_frame = new_frame if new_frame > 0 else self.__cur_frame
            logging.info(f"Current Frame: {self.__cur_frame}")
        elif key == 'Right':
            new_frame = self.__cur_frame + 1
            self.__cur_frame = new_frame if new_frame < self.__max_frames else self.__cur_frame
            logging.info(f"Current Frame: {self.__cur_frame}")

    def __create_line_vtk_object(self, color) -> vtk.vtkLineSource:
        line = vtk.vtkLineSource()
        line.SetPoint1(0, 0, 0)
        line.SetPoint2(0, 0, 0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.AddInputConnection(line.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.__colors.GetColor3d(color))
        self.__renderer.AddActor(actor)
        return line
