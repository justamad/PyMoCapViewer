from .skeletons import get_skeleton_definition_for_camera
from typing import List, Tuple, Union

import vtk
import pandas as pd
import numpy as np
import logging

COLORS = ["red", "green", "blue"]

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)


class MoCapViewer(object):

    def __init__(
            self,
            width: int = 1280,
            height: int = 1024,
            sampling_frequency: int = 30,
            sphere_radius: float = 0.01,
    ):
        self.__skeleton_objects = []
        self.__max_frames = float('inf')
        self.__cur_frame = 0
        self.__pause = False
        self.__record = False
        self.__trans_vector = np.array([0, 0, 0])
        self.__scale_factor = 1.0
        self.__sphere_radius = sphere_radius
        self.__axis_scale = 0.3
        self.__video_count = 0
        self.__sampling_frequency = sampling_frequency

        self.__colors = vtk.vtkNamedColors()
        self.__renderer = vtk.vtkRenderer()
        self.__renderer.SetBackground(0, 0, 0)
        self.__renderer.ResetCamera()

        self.__render_window = vtk.vtkRenderWindow()
        self.__render_window.SetSize(width, height)
        self.__render_window.AddRenderer(self.__renderer)

        self.__render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.__render_window_interactor.SetRenderWindow(self.__render_window)
        self.__render_window_interactor.Initialize()
        self.__timer_id = self.__render_window_interactor.CreateRepeatingTimer(1000 // self.__sampling_frequency)
        self.__render_window_interactor.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)

        self.__draw_coordinate_axes()

    def add_skeleton(
            self,
            data: pd.DataFrame,
            skeleton_connection: Union[str, List[Tuple[str, str]], List[Tuple[int, int]]] = None,
            color: str = None,
    ):
        columns = data.shape[1]
        if columns % 3 != 0:
            raise ValueError(f"Column-number of dataframe should be a multiple of 3, received {columns}.")

        if isinstance(skeleton_connection, str):
            skeleton_connection = get_skeleton_definition_for_camera(
                df=data,
                camera_name=skeleton_connection,
                camera_count=len(self.__skeleton_objects),
            )

        actors_markers = []  # Each marker has an own actor
        actors_bones = []  # Actors for each line segment between two markers
        lines = []

        if color is not None:
            if color not in self.__colors.GetColorNames():
                raise ValueError(f"Unknown color given: {color}.")
        else:
            color = COLORS[len(self.__skeleton_objects) % len(COLORS)]

        # Create all instances for all markers
        n_markers = columns // 3
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
                line = vtk.vtkLineSource()
                line.SetPoint1(0, 0, 0)
                line.SetPoint2(0, 0, 0)
                lines.append(line)

                # Setup actor and mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.AddInputConnection(line.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(self.__colors.GetColor3d(color))
                self.__renderer.AddActor(actor)
                actors_bones.append(actor)

        self.__skeleton_objects.append({
            'data': data.to_numpy(),
            'connections': skeleton_connection,
            'lines': lines,
            'actors_markers': actors_markers,
        })
        self.__max_frames = min(map(lambda x: len(x['data']), self.__skeleton_objects))

    def _calculate_bounding_box(self):
        data = np.concatenate(list(map(lambda x: x['data'].reshape(-1, 3), self.__skeleton_objects)))
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        self.__trans_vector = np.array(
            [(min_values[0] + max_values[0]) / 2,
             min_values[1],
             (min_values[2] + max_values[2]) / 2]
        )
        self.__scale_factor = np.max(data - self.__trans_vector)

    def __draw_coordinate_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(self.__axis_scale, self.__axis_scale, self.__axis_scale)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.__renderer.AddActor(axes)

    def show_window(self):
        self._calculate_bounding_box()
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
        for skeleton_data in self.__skeleton_objects:
            data = skeleton_data['data']
            actors_markers = skeleton_data['actors_markers']
            points = (data[index].reshape(-1, 3) - self.__trans_vector) / self.__scale_factor

            for c_points, actor in enumerate(actors_markers):
                x, y, z = points[c_points]
                actor.SetPosition(x, y, z)

            # Update bone connections
            bones = skeleton_data['connections']
            lines = skeleton_data['lines']
            if bones is None:
                continue

            for line, (j1, j2) in zip(lines, bones):
                line.SetPoint1(points[j1])
                line.SetPoint2(points[j2])

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
