import os
from functools import partial
from pathlib import Path
import shutil

import pandas as pd

# Creat unique ids for each new html div
HTML_IDX = [0]


def path2video(path, local_folder="", collapsible=True, call_nb=HTML_IDX):
    if local_folder:
        local_folder = Path(local_folder) / "video"
        local_folder.mkdir(exist_ok=True, parents=True)

        ext = str(path).split(".")[-1]
        video_name = f"{call_nb[0]:04d}.{ext}"
        dest_img_path = local_folder / video_name
        shutil.copy(path, dest_img_path)
        rel_path = os.path.join("video", video_name)
    else:
        rel_path = path

    # Keep track of count number
    call_nb[0] += 1
    vid_ext = str(rel_path).split(".")[-1]
    vid_str = ('<video controls> <source src="' + str(rel_path) +
               f'" type="video/{vid_ext}"></video>')
    if collapsible:
        vid_str = make_collapsible(vid_str, call_nb[0])
    return vid_str


def drop_redundant_columns(df):
    """
    If dataframe contains multiple lines, drop the ones for which the column
    contains equal values
    """
    if len(df) > 1:
        nunique = df.apply(pd.Series.nunique)
        # Drop columns with all identical values or all None
        cols_to_drop = nunique[nunique <= 1].index
        print(f"Dropping {list(cols_to_drop)}")
        df = df.drop(cols_to_drop, axis=1)
    return df


def make_collapsible(html_str, collapsible_idx=0):
    """
    Create collapsible button to selectively hide large html items such as images
    """
    pref = (
        f'<button data-toggle="collapse" data-target="#demo{collapsible_idx}">'
        "Toggle show image</button>"
        f'<div id="demo{collapsible_idx}" class="collapse">')
    suf = "</div>"
    return pref + html_str + suf


def path2img(
    path,
    local_folder="",
    collapsible=True,
    call_nb=HTML_IDX,
    height=None,
):
    if local_folder:
        local_folder = Path(local_folder) / "imgs"
        local_folder.mkdir(exist_ok=True, parents=True)

        ext = str(path).split(".")[-1]
        img_name = f"{call_nb[0]:04d}.{ext}"
        dest_img_path = local_folder / img_name
        shutil.copy(path, dest_img_path)
        rel_path = os.path.join("imgs", img_name)
    else:
        rel_path = path

    # Keep track of count number
    call_nb[0] += 1
    if height is None:
        end_str = '">'
    else:
        end_str = f'" height="{height}" />'
    img_str = '<img src="' + str(rel_path) + '"/>'
    if collapsible:
        img_str = make_collapsible(img_str, call_nb[0])
    return img_str


def df2html(df, local_folder="", drop_redundant=True, collapsible=True, img_height=None):
    """
    Convert df to html table, getting images for fields which contain 'img_path'
    in their name and videos for fields which contain 'video_path'
    """
    keys = list(df.keys())
    format_dicts = {}
    for key in keys:
        if "img_path" in key:
            format_dicts[key] = partial(path2img,
                                        local_folder=local_folder,
                                        collapsible=collapsible,
                                        height=img_height)
        elif "video_path" in key:
            format_dicts[key] = partial(path2video,
                                        local_folder=local_folder,
                                        collapsible=collapsible)

    if drop_redundant:
        df = drop_redundant_columns(df)

    df_html = df.to_html(escape=False, formatters=format_dicts)

    return df_html
