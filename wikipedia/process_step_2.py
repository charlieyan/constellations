#!/usr/bin/env python

import svgpathtools
from svgpathtools import svg2paths, wsvg
import argparse
from copy import deepcopy
import numpy as np
import sys, os, math
import cairo
import svg
import pickle
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.path import Path
import matplotlib.patches as patches

from common import *
from ego_atheris.interaction.utils import Util

Path = mpath.Path

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def process_svg_step_2_1(data_pickle, do_plt = True):
    directory = os.path.realpath(os.path.dirname(data_pickle))
    data_to_save = pickle.load(open(data_pickle, "rb"))
    points_to_path_map = data_to_save["points_to_path_map"]
    path_to_points_map = data_to_save["path_to_points_map"]
    path_to_p_key_idx_map = data_to_save["path_to_p_key_idx_map"]
    s_p_key_paths = data_to_save["p_key_paths"]
    s_p_key_attrs = data_to_save["p_key_attrs"]
    points = points_to_path_map.keys()
    points = map(lambda x: decode(x), points)
    # get all points, making bounding box
    b = BoundingBox(points)
    center = [
        (b.maxx - b.minx) / 2.0,
        (b.maxx - b.minx) / 2.0]
    closest_point = find_n_closest_point(
        points, center, 1)[0]
    e = encode(closest_point)
    path_of_interest = points_to_path_map[e]
    path_of_interest_indices = map(lambda x: path_to_p_key_idx_map[x],
        path_of_interest)
    indices_file = directory + "/step_2_1.txt"
    with open(indices_file, "w") as text_file:
        text_file.write(encode(
            path_of_interest_indices))
    return True

def process_svg_step_2_2(data_pickle, do_plt = True):
    directory = os.path.realpath(os.path.dirname(data_pickle))
    data_to_save = pickle.load(open(data_pickle, "rb"))
    points_to_path_map = data_to_save["points_to_path_map"]
    path_to_points_map = data_to_save["path_to_points_map"]
    path_to_p_key_idx_map = data_to_save["path_to_p_key_idx_map"]
    s_p_key_paths = data_to_save["p_key_paths"]
    s_p_key_attrs = data_to_save["p_key_attrs"]

    indices_file = directory + "/step_2_1.txt"
    path_of_interest_indices = []
    with open(indices_file, "r") as text_file:
        first_line = text_file.read()
        print first_line
        path_of_interest_indices = map(lambda x: int(x),
            first_line.split(","))
    print "path_of_interest_indices: ", path_of_interest_indices
    fig, ax = plt.subplots()
    ignore_paths = []
    for i in path_of_interest_indices:
        path = s_p_key_paths[i].d()
        ignore_paths.append(path)
        if do_plt:
            transform = data_to_save[
                "path_to_transform_map"][path]
            plot_path(path, transform, 'b')
    for p in path_to_points_map.keys():
        if p not in ignore_paths:
            t = data_to_save["path_to_transform_map"][p]
            plot_path(p, t, 'r')
    plot_png = directory + "/step_2_2.png"
    plt.grid(True)
    ax.set_xlim(0, 600)
    ax.set_ylim(-600, 0)
    ax.set_aspect('equal')
    plt.draw()
    ax.apply_aspect()
    fig.savefig(plot_png)

    data_pickle = directory + "/step_2_2.p"
    data_to_save = {}
    data_to_save["paths"] = path_of_interest_indices
    pickle.dump(data_to_save, open(data_pickle, "wb"))
    print "dumped data_pickle! ", data_pickle
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "get points / paths of interest from constellation svg")
    parser.add_argument('--file',
        type=str, required=True, help='data pickle path')
    args = parser.parse_args()
    res = process_svg_step_2_2(args.file)
