import pandas as pd
import numpy as np
import open3d as o3d
import vtk
import logging

from .skeletons import get_skeleton_definition_for_camera
from .utils import (
    create_xy_points,
    create_yz_points,
    create_xz_points,
    create_orientations_from_euler_angles,
    create_orientations_from_quaternions,
)
from typing import List, Tuple, Union
from vtk.util.numpy_support import numpy_to_vtk, get_numpy_array_type, numpy_to_vtkIdTypeArray
from vtkmodules.vtkIOImage import vtkPNGWriter
from datetime import datetime


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
            bg_color: str = "lightslategray",
            pause: bool = False,
            start_frame: int = 0
    ):
        self.__skeleton_objects = []
        self.__point_cloud_objects = []
        self.__max_frames = float("inf")
        self.__cur_frame = start_frame
        self.__pause = pause
        self.__record = False
        self.__trans_vector = np.array([0, 0, 0])
        self.__scale_factor = 1.0
        self.__sphere_radius = sphere_radius
        self.__axis_scale = 0.3
        self.__video_count = 0
        self.__sampling_frequency = sampling_frequency
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

        # Screen capture utilities
        self.__win_to_img_filter = vtk.vtkWindowToImageFilter()
        self.__win_to_img_filter.SetInput(self.__render_window)
        self.__win_to_img_filter.SetInputBufferTypeToRGB()
        self.__win_to_img_filter.ReadFrontBufferOff()
        self.__win_to_img_filter.Update()

        self.__is_screen_capture = False
        self.__movie_writer = vtk.vtkOggTheoraWriter()
        self.__movie_writer.SetInputConnection(self.__win_to_img_filter.GetOutputPort())

        self.__draw_coordinate_axes()

    def activate_grid(self, grid_axis: str, line_width: float, color: str, dimensions: int = 11):
        if grid_axis not in planes:
            raise AttributeError(f"Unknown grid axis given: {grid_axis}. Use 'xy', 'yz', or 'xz' or None")

        grid_creator = planes[grid_axis]
        color = self.__colors.GetColor3d(color)
        cell_size = 0.6
        offset = dimensions / 2 * cell_size
        for i in range(dimensions):
            for j in range(dimensions):
                p1, p2, p3, p4 = grid_creator(i, j, cell_size, offset)
                points = vtk.vtkPoints()
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
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(line_width)
                self.__renderer.AddActor(actor)


    def add_skeleton(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            skeleton_orientations: Union[np.ndarray, pd.DataFrame] = None,
            skeleton_connection: Union[str, List[Tuple[int, int]]] = None,
            color: str = None,
            orientation: str = "quaternion",
            unit: str = "mm",
            show_labels: bool = False,
            labels_scale: float = 0.01,
    ):
        if len(data.shape) != 2:
            raise AttributeError(f"Data container has wrong dimensions. Given: {data.shape}, Expected: 2")

        if data.shape[1] % 3 != 0:
            raise ValueError(f"Column-number of dataframe should be a multiple of 3, received {data.shape[1]}.")

        column_names = list(data.columns)

        if isinstance(data, pd.DataFrame):
            if isinstance(skeleton_connection, str):
                skeleton_connection = get_skeleton_definition_for_camera(
                    columns=column_names,
                    camera_name=skeleton_connection,
                    camera_count=len(self.__skeleton_objects),
                )

            data = data.to_numpy()

        if skeleton_orientations is not None:
            if len(skeleton_orientations) != len(data):
                raise AttributeError(f"Position and orientation data have different lengths:"
                                     f" {len(skeleton_orientations)} vs {len(data)}.")

            if isinstance(skeleton_orientations, pd.DataFrame):
                skeleton_orientations = skeleton_orientations.to_numpy()

            if orientation == "quaternion":
                skeleton_orientations = create_orientations_from_quaternions(skeleton_orientations)
            elif orientation == "euler":
                skeleton_orientations = create_orientations_from_euler_angles(skeleton_orientations)
            else:
                raise AttributeError(f"Unknown orientation type: {orientation}")

        if unit not in units:
            raise ValueError(f"Unknown unit given: {unit}.")

        n_markers = data.shape[1] // 3
        actors_markers = []  # Each marker has an own actor
        actors_labels = []  # Each marker has an own label
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

            if show_labels:
                # Create the 3D text and the associated mapper and follower (a type of actor) to align with the camera view.
                a_text = vtk.vtkVectorText()
                # Add a little space in front so that the text is not overlapping with the sphere
                a_text.SetText("        " + column_names[marker * 3].replace(" (x)", ""))
                text_mapper = vtk.vtkPolyDataMapper()
                text_mapper.SetInputConnection(a_text.GetOutputPort())
                text_actor = vtk.vtkFollower()
                text_actor.SetMapper(text_mapper)
                text_actor.SetScale(labels_scale, labels_scale, labels_scale)
                text_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                text_actor.SetCamera(self.__renderer.GetActiveCamera())
                actors_labels.append(text_actor)
                self.__renderer.AddActor(text_actor)

        if skeleton_connection is not None:
            for _ in skeleton_connection:
                line = self.__create_line_vtk_object(color)
                lines.append(line)

        coordinate_axes = []
        if skeleton_orientations is not None:
            skeleton_orientations = skeleton_orientations * 1e-1
            for _ in range(n_markers):
                coordinate_axes.append(self.__create_line_vtk_object("red"))
                coordinate_axes.append(self.__create_line_vtk_object("green"))
                coordinate_axes.append(self.__create_line_vtk_object("blue"))

        self.__skeleton_objects.append({
            "data": data * units[unit],
            "ori_data": skeleton_orientations,
            "skeleton_definition": skeleton_connection,
            "bone_actors": lines,
            "actors_markers": actors_markers,
            "actors_labels": actors_labels,
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
            actors_labels = skeleton_data["actors_labels"]
            points = (data[index].reshape(-1, 3) - self.__trans_vector) / self.__scale_factor

            for marker_idx, actor in enumerate(actors_markers):
                x, y, z = points[marker_idx]
                actor.SetPosition(x, y, z)

            for marker_idx, actor in enumerate(actors_labels):
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

        if self.__is_screen_capture:
            self.__win_to_img_filter.Modified()
            self.__movie_writer.Write()

    def keypress_callback(self, obj, ev):
        key = obj.GetKeySym()
        if key == 'space':
            self.__pause = not self.__pause
        elif key == 'Left':
            self.__cur_frame = (self.__cur_frame - 1) % self.__max_frames
            logging.info(f"Current Frame: {self.__cur_frame}")
        elif key == 'Right':
            self.__cur_frame = (self.__cur_frame + 1) % self.__max_frames
            logging.info(f"Current Frame: {self.__cur_frame}")
        elif key == '0':
            self.__cur_frame = 0
            logging.info("Back to frame 0")
        elif key == 'i':
            logging.info(f"Current frame: {self.__cur_frame}")
        elif key == 'q':
            self.__render_window_interactor.ExitCallback()
        elif key == "s":
            self.__save_screenshot()
        elif key == 'Return':
            self.__record_screen_video()

    def __create_line_vtk_object(self, color) -> vtk.vtkLineSource:
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(0, 0, 0)
        line_source.SetPoint2(0, 0, 0)

        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(line_source.GetOutputPort())
        tube_filter.SetRadius(self.__sphere_radius / 4)  # Set the desired tube radius
        tube_filter.SetNumberOfSides(50)  # Set the number of sides of the tube

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.__colors.GetColor3d(color))
        self.__renderer.AddActor(actor)

        return line_source

    def __save_screenshot(self):
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.__render_window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Update()

        file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        writer = vtkPNGWriter()
        writer.SetCompressionLevel(0)
        writer.SetFileName(file_name)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        logging.info(f"Screenshot saved as {file_name}")

    def __record_screen_video(self):
        if not self.__is_screen_capture:
            file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
            logging.info(f"Video capture started: {file_name}")
            self.__movie_writer.SetFileName(file_name)
            self.__movie_writer.Start()
            self.__is_screen_capture = True
        else:
            logging.info("Video capture stopped...")
            self.__movie_writer.End()
            self.__is_screen_capture = False
