import pandas as pd
import os
import requests

from os.path import exists, join
from typing import Tuple

BASE_URL = "https://raw.githubusercontent.com/justamad/PyMoCapViewer/main/data/"
temp_path = "data"


def get_azure_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_file = download_example_data(join(BASE_URL, "azure_positions.csv"), temp_path)
    ori_file = download_example_data(join(BASE_URL, "azure_orientations.csv"), temp_path)

    pos_df = pd.read_csv(pos_file, index_col=0, sep=";")
    ori_df = pd.read_csv(ori_file, index_col=0, sep=";")
    pos_df = select_main_person(pos_df)
    ori_df = select_main_person(ori_df)
    pos_df = pos_df[[c for c in pos_df.columns if "(c)" not in c]]
    return pos_df, ori_df


def get_vicon_data() -> pd.DataFrame:
    vicon_file = download_example_data(join(BASE_URL, "vicon.csv"), temp_path)
    df = pd.read_csv(vicon_file, index_col=False, sep=";")
    return df


def select_main_person(df: pd.DataFrame) -> pd.DataFrame:
    body_idx_c = df["body_idx"].value_counts()
    df = df[df["body_idx"] == body_idx_c.index[body_idx_c.argmax()]]
    df = df.drop("body_idx", axis=1)
    return df


def download_example_data(url, save_path):
    if not exists(save_path):
        os.makedirs(save_path)

    local_filename = join(save_path, url.split('/')[-1])
    if exists(local_filename):
        return local_filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_filename
