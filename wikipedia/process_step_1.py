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

def process_svg_step_1(svgpath, do_plt):
    result = False

    paths, attributes = svg2paths(svgpath)
    attributes_copy = deepcopy(attributes)
    color_gs = {}
    for i, d in enumerate(attributes):
        if "style" in d.keys():
            if "rgb" in d["style"]:
                d["attrs_idx"] = i
                d["path"] = paths[i]
                d["rgb"] = str(d["style"]).split(
                    "rgb")[1].split(")")[0].replace("(","")
                h = percents_to_hex_code(d["rgb"])
                Util.insert_into_dict_of_arrays(
                    color_gs, h, d)

    if do_plt:
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('hsv')
        colors = cmap(np.linspace(0, 1, 50))

    p_key = "#addf8a"
    points_to_path_map = {}
    path_to_points_map = {}
    path_to_p_key_idx_map = {}
    path_to_transform_map = {}
    connectivity_map = {}
    # assuming the constellation is stored in 1 path
    if p_key not in color_gs:
        return result
    p_key_paths = map(lambda x: x["path"], color_gs[p_key])
    p_key_attrs = map(lambda x: attributes_copy[
        x["attrs_idx"]], color_gs[p_key])

    s_p_key_paths = sorted(p_key_paths, key = lambda k: k.d())
    indices = sorted(range(len(p_key_paths)),
        key=lambda k: p_key_paths[k].d())
    s_p_key_attrs = []
    for i in indices:
        s_p_key_attrs.append(p_key_attrs[i])

    for i, path in enumerate(s_p_key_paths):
        transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if "transform" in s_p_key_attrs[i].keys():
            transform_str = str(s_p_key_attrs[i]["transform"])
            transform = get_transform_matrix(transform_str)
        path_to_transform_map[path.d()] = transform

        path_to_p_key_idx_map[path.d()] = i

        path_points = path_to_points(path.d())

        last_point = []
        for j, p in enumerate(path_points):
            if (p["type"] == "M"):
                first = len(last_point) == 0
                last_point = apply_matrix(transform,
                    p["data"])
                if do_plt and first:
                    font = {'family': 'serif',
                        'color':  colors[i],
                        'weight': 'normal',
                        'size': 15,
                        }
                    plt.text(
                        last_point[0],
                        -last_point[1],
                        str(i), fontdict=font)
            elif (p["type"] == "C"):
                points = p["data"]
                if do_plt:
                    verts = [
                        (last_point[0], -last_point[1]),  # P0
                        (points[0][0], -points[0][1]), # P1
                        (points[1][0], -points[1][1]), # P2
                        (points[2][0], -points[2][1]), # P3
                        ]
                    codes = [Path.MOVETO,
                             Path.CURVE4,
                             Path.CURVE4,
                             Path.CURVE4,
                             ]
                    path = Path(verts, codes)
                    patch = patches.PathPatch(
                        path, facecolor='none', lw=2)
                    ax.add_patch(patch)

                last_point = points[2]
            elif (p["type"] == "L"):
                point = p["data"]
                new_point = apply_matrix(transform, point)
                xs = [last_point[0], new_point[0]]
                ys = [last_point[1], new_point[1]]

                if do_plt:
                    plt.plot(
                        xs,
                        map(lambda x: -1 * x, ys),
                        # ys,
                        linewidth=2, c=colors[i], alpha=0.5)

                c = encode(last_point)
                Util.insert_into_dict_of_arrays(
                    points_to_path_map, c, path.d())
                Util.insert_into_dict_of_arrays(
                    path_to_points_map, path.d(), c)

                d = encode(new_point)
                Util.insert_into_dict_of_arrays(
                    points_to_path_map, d, path.d())
                Util.insert_into_dict_of_arrays(
                    path_to_points_map, path.d(), d)

                Util.insert_into_dict_of_arrays(
                    connectivity_map, c, d)
                # map stores paths from start to end

                last_point = new_point
    for path_id in path_to_points_map.keys():
        path_to_points_map[path_id] = list(
            set(path_to_points_map[path_id]))
    for point in points_to_path_map.keys():
        points_to_path_map[point] = list(
            set(points_to_path_map[point]))

    directory = os.path.realpath(os.path.dirname(svgpath))
    data_pickle = directory + "/step_1.p"
    data_to_save = {}
    data_to_save["color_gs"] = color_gs
    data_to_save["p_key_paths"] = s_p_key_paths
    data_to_save["p_key_attrs"] = s_p_key_attrs
    data_to_save["path_to_p_key_idx_map"] = path_to_p_key_idx_map
    data_to_save["points_to_path_map"] = points_to_path_map
    data_to_save["path_to_points_map"] = path_to_points_map
    data_to_save["path_to_transform_map"] = path_to_transform_map
    data_to_save["connectivity_map"] = connectivity_map
    pickle.dump(data_to_save, open(data_pickle, "wb"))
    print "dumped data_pickle! ", data_pickle
    result = True

    if do_plt:
        plt.grid(True)
        ax.set_xlim(0, 600)
        ax.set_ylim(-600, 0)
        ax.set_aspect('equal')
        plt.draw()
        ax.apply_aspect()

        directory = os.path.realpath(os.path.dirname(svgpath))
        plot_png = directory + "/step_1_plot.png"
        print "plot_png: ", plot_png
        fig.savefig(plot_png)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "get points / paths of interest from constellation svg")
    parser.add_argument('--file', type=str, required=True, help='svg file name')
    parser.add_argument('--plt', type=bool, required=False, default=True)
    args = parser.parse_args()
    process_svg_step_1(args.file, args.plt)
