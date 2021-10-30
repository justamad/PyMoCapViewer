from .skeletons import get_skeleton_definition_for_camera

import vtk
import pandas as pd
import numpy as np

COLORS = ["red", "green", "blue"]


class MoCapViewer(object):

    def __init__(
            self,
            sampling_frequency: int,
            width: int = 1280,
            height: int = 1024,
            sphere_radius: float = 0.01
    ):
        self.colors = vtk.vtkNamedColors()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.renderer.ResetCamera()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(width, height)
        self.render_window.AddRenderer(self.renderer)

        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)
        self.render_window_interactor.Initialize()
        self._timer_id = self.render_window_interactor.CreateRepeatingTimer(1000 // sampling_frequency)
        self.render_window_interactor.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)

        self.video_writer = vtk.vtkAVIWriter()
        self.image_filter = vtk.vtkWindowToImageFilter()
        self.image_filter.SetInput(self.render_window)

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

        self._draw_coordinate_axes()

    def add_skeleton(self, data: pd.DataFrame, skeleton_connection=None):
        features = data.shape[1]
        assert features % 3 == 0, f"Markers should have a multiple of 3 columns, received {features}"
        nr_markers = features // 3

        if type(skeleton_connection) == str:
            skeleton_connection = get_skeleton_definition_for_camera(data, skeleton_connection)

        actors_markers = []  # Each marker has an own actor
        actors_bones = []  # Actors for each line segment between two markers
        lines = []

        # Create all instances for all markers
        for marker in range(nr_markers):
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(100)
            sphere.SetThetaResolution(100)
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(self.__sphere_radius)
            mapper = vtk.vtkPolyDataMapper()
            mapper.AddInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.__skeleton_objects)]))
            self.renderer.AddActor(actor)
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
                actor.GetProperty().SetColor(self.colors.GetColor3d(COLORS[len(self.__skeleton_objects)]))
                self.renderer.AddActor(actor)
                actors_bones.append(actor)

        # Invert y-coordinate axis for Azure Kinect data
        # data[data.filter(like='(y)').columns] *= -1

        self.__skeleton_objects.append({
            'data': data.to_numpy(),
            'connections': skeleton_connection,
            'lines': lines,
            'actors_markers': actors_markers,
        })
        self.__max_frames = min(map(lambda x: len(x['data']), self.__skeleton_objects))

    def _calculate_bounding_box(self):
        data = np.concatenate(list(map(lambda x: x['data'].reshape(-1, 3), self.__skeleton_objects)))
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.__trans_vector = np.array([(min_vals[0] + max_vals[0]) / 2, min_vals[1], (min_vals[2] + max_vals[2]) / 2])
        self.__scale_factor = np.max(data - self.__trans_vector)

    def _draw_coordinate_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(self.__axis_scale, self.__axis_scale, self.__axis_scale)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.renderer.AddActor(axes)

    def show_window(self):
        self._calculate_bounding_box()
        self.render_window_interactor.AddObserver('TimerEvent', self._update)
        self.render_window.Render()
        self.render_window_interactor.Start()

    def _update(self, iren, event):
        if self.__cur_frame >= self.__max_frames:
            self.__cur_frame = 0

        self._draw_new_frame(self.__cur_frame)
        iren.GetRenderWindow().Render()

        if self.__record:
            self._write_video()

        if not self.__pause:
            self.__cur_frame += 1

    def _start_video(self):
        self.video_writer.SetFileName(f'video_{self.__video_count}.avi')
        self.video_writer.SetInputConnection(self.image_filter.GetOutputPort())
        self.video_writer.SetRate(30)
        self.video_writer.SetQuality(2)
        self.video_writer.Start()

    def _write_video(self):
        self.image_filter.Modified()
        self.video_writer.Write()

    def _close_video(self):
        self.video_writer.End()
        self.__video_count += 1

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
            print(f"Current Frame: {self.__cur_frame}")
        elif key == 'Right':
            new_frame = self.__cur_frame + 1
            self.__cur_frame = new_frame if new_frame < self.__max_frames else self.__cur_frame
            print(f"Current Frame: {self.__cur_frame}")
        elif key == 'v':
            if self.__record:
                self._close_video()
                self.__record = False
            else:
                self._start_video()
                self.__record = True
