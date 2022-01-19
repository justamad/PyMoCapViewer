# PyMoCapViewer

![Alt text](docs/title_image.PNG?raw=true "PyMoCap")

A simple Python Motion Capture Viewer using the VTK toolkit. It is able to visualize trajectories from 3D markers given as a Pandas Dataframe. It also is able to draw connection between adjacent joints. Until now, it supports pre-defined joint connection for:

- Vicon Plug-in Gait model
- Microsoft Azure Kinect
- Microsoft Kinect v2
- VNect definition

## Example
Load the kinematic data as a Pandas Dataframe. The columns should contain the joints in the following format: J1 (x), J1 (y), J1 (z), ..., JN (x), JN (y), JN (z).

```python
from PyMoCapViewer import MoCapViewer

import pandas as pd

df = pd.read_csv("file_to_vicon.csv")

render = MoCapViewer(sampling_frequency=100)
render.add_skeleton(df, skeleton_connection="vicon", color="gray")
render.show_window()
```
