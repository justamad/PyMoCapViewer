# PyMoCapViewer

A simple Python Motion Capture Viewer using the VTK toolkit. It is able to visualize trajectories from 3D markers given as a Pandas Dataframe. It also is able to draw connection between adjacent joints. Until now, it supports pre-defined joint connection for:

- Vicon Plug-in Gait model
- Microsoft Azure Kinect
- Microsoft Kinect v2

## Example
Load the data as a Pandas Dataframe. The columns should contain the joints in the format: J1 (x), J1 (y), J1 (z)

