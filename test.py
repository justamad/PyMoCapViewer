from PyMoCapViewer import MoCapViewer
from PyMoCapViewer.examples import get_azure_data, get_vicon_data

df_pos, df_ori = get_azure_data()
viewer = MoCapViewer(sphere_radius=0.01, sampling_frequency=30)
viewer.add_skeleton(df_pos, skeleton_connection="azure", skeleton_orientations=df_ori, orientation="quaternion")
viewer.show_window()

df = get_vicon_data()
viewer = MoCapViewer(sphere_radius=0.01, sampling_frequency=100)
viewer.add_skeleton(df, skeleton_connection="vicon", color="red")
viewer.show_window()
