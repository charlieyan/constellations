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
import math

from common import *
from ego_atheris.interaction.utils import Util

Path = mpath.Path
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def myround(x, base=5):
    return int(base * round(float(x)/base))

def get_distances(last, this, n):
    dist_1 = abs(round(np.linalg.norm(
        np.array(this)-np.array(last)), 3))
    dist_2 = abs(round(np.linalg.norm(
        np.array(n)-np.array(this)), 3))
    return dist_1, dist_2

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    if magA == magB:
        return -1.0
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else:
        return ang_deg
def get_angle(last, this, n):
    line_a = [last, this]
    line_b = [n, this]
    return ang(line_a, line_b)

def process_svg_step_3(constellation_dir, do_plt = True):
    data_pickle = constellation_dir + "/step_1.p"
    directory = os.path.realpath(os.path.dirname(data_pickle))
    data_to_save = pickle.load(open(data_pickle, "rb"))
    points_to_path_map = data_to_save["points_to_path_map"]
    path_to_points_map = data_to_save["path_to_points_map"]
    path_to_p_key_idx_map = data_to_save["path_to_p_key_idx_map"]
    s_p_key_paths = data_to_save["p_key_paths"]
    s_p_key_attrs = data_to_save["p_key_attrs"]
    points = points_to_path_map.keys()
    decoded_points = map(lambda x: decode(x), points)

    # get points in the constellation path and their connectivities
    data_pickle = constellation_dir + "/step_2_2.p"
    step_2_data_pickle = pickle.load(open(data_pickle, "rb"))
    path_of_interest_indices = step_2_data_pickle["paths"]

    all_segments = []
    if do_plt:
        fig, ax = plt.subplots()
    for i in path_of_interest_indices:
        path = s_p_key_paths[i].d()
        points = path_to_points_map[path]
        if do_plt:
            transform = data_to_save["path_to_transform_map"][path]
            segments = plot_path(path, transform, 'b')
            all_segments += segments

    graph_map = {}
    for seg in all_segments:
        last_point = map(lambda x: myround(x, 1), seg[0])
        new_point = map(lambda x: myround(x, 1), seg[1])
        Util.insert_into_dict_of_arrays(graph_map,
            encode(last_point),
            encode(new_point))

    ratio_and_angle_map = {}
    ratio_and_angle_map_2 = {}
    threshold = 15.0
    done = False
    for last_point in graph_map:
        added_something = False
        if done:
            break
        key = last_point
        graph_map[last_point] = list(set(graph_map[last_point]))
        last_point_decoded = decode(last_point)
        last_point_decoded[1] = -1 * last_point_decoded[1]
        points = graph_map[last_point]
        sorted_points, _ =\
            sorted_and_indices(points, cb_for_points)
        for i, point in enumerate(sorted_points):
            if done:
                break
            if point in graph_map:
                key_2 = key + "_" + str(i)
                point_decoded = decode(point)
                point_decoded[1] = -1 * point_decoded[1]
                xs = [point_decoded[0]]
                ys = [point_decoded[1]]
                plt.scatter(xs, ys, s=50, c='r', alpha=0.5)
                next_points = graph_map[point]
                sorted_next_points, _=\
                    sorted_and_indices(next_points, cb_for_points)
                for j, next_point in enumerate(sorted_next_points):
                    if done:
                        break
                    next_point_decoded = decode(
                        next_point)
                    next_point_decoded[1] = -1 * next_point_decoded[1]
                    xs = [last_point_decoded[0], point_decoded[0], next_point_decoded[0]]
                    ys = [last_point_decoded[1], point_decoded[1], next_point_decoded[1]]
                    plt.plot(xs, ys, linewidth=2, c='g', alpha=1.0)
                    dist_1, dist_2 = get_distances(last_point_decoded, point_decoded, next_point_decoded)
                    if dist_1 > threshold and dist_2 > threshold:
                        key_3 = key_2 + "_" + str(j)
                        ratio = min(dist_1, dist_2) / max(dist_1, dist_2)
                        angle = get_angle(
                            last_point_decoded,
                            point_decoded,
                            next_point_decoded)
                        if angle > 0.0:
                            ratio_and_angle_map[key_3] =\
                                {"ratio" : ratio, "angle" : angle}
                            reverse_key = str(round(ratio, 3))\
                                + ":" + str(round(angle, 3))
                            ratio_and_angle_map_2[reverse_key] = key_3
                        else:
                            # print "bad angle"
                            pass
                        # done = True
                        added_something = True
                        # break
                    else:
                        # print "not long enough"
                        pass
        # print "finished something", added_something

    plot_png = directory + "/step_3.png"
    plt.grid(True)
    ax.set_xlim(0, 600)
    ax.set_ylim(-600, 0)
    ax.set_aspect('equal')
    plt.draw()
    ax.apply_aspect()
    fig.savefig(plot_png)

    data_pickle = directory + "/step_3.p"
    data_to_save = {}
    data_to_save["graph_map"] = graph_map
    data_to_save["ratio_and_angle_map"] = ratio_and_angle_map
    data_to_save["ratio_and_angle_map_2"] = ratio_and_angle_map_2
    pickle.dump(data_to_save, open(data_pickle, "wb"))
    print "dumped data_pickle! ", data_pickle

    return graph_map, ratio_and_angle_map, ratio_and_angle_map_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "get points in the constellation with 2 lines attached")
    parser.add_argument('--dir',
        type=str, required=True, help='data pickle path')
    args = parser.parse_args()
    graph, r_and_a, r_and_a_2 = process_svg_step_3(args.dir)
