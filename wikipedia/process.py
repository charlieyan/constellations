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

from ego_atheris.interaction.utils import Util

Path = mpath.Path

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def get_transform_matrix(s):
    numbers = map(lambda x: float(x),
        s.replace("matrix(", "").replace(")", "").split(","))
    return numbers

def clean_up_filename(s):
    return s.replace(".svg", "")

def path_to_points(d_str):
    points = []
    tokens = d_str.split(" ")
    i = 0
    while (i < len(tokens)):
        if tokens[i] == "M":
            coords = map(lambda x: float(x),
                tokens[i + 1].split(","))
            datum = {"type" : "M", "data" : coords}
            points.append(datum)
            i = i + 2
        elif tokens[i] == "C":
            coords_1 = map(lambda x: float(x),
                tokens[i + 1].split(","))
            coords_2 = map(lambda x: float(x),
                tokens[i + 2].split(","))
            coords_3 = map(lambda x: float(x),
                tokens[i + 3].split(","))
            coords = [coords_1, coords_2, coords_3]
            datum = {"type" : "C", "data" : coords}
            points.append(datum)
            i = i + 4
        elif tokens[i] == "L":
            coords = map(lambda x: float(x),
                tokens[i + 1].split(","))
            datum = {"type" : "L", "data" : coords}
            points.append(datum)
            i = i + 2
    return points

def apply_matrix(matrix, point):
    # a pass through matrix is [1, 0, 0, 1, 0, 0]
    x = point[0]
    y = point[1]
    new_x = x * matrix[0] + y * matrix[2] + matrix[4]
    new_y = x * matrix[1] + y * matrix[3] + matrix[5]
    return [new_x, new_y]

def percents_to_hex_code(percent_str):
    tokens = percent_str.split(",")
    rgb = map(lambda x: round(float(x.replace("%", "")) * 255 / 100),
        tokens)
    h = "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])
    return h

class BoundingBox(object):
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)

    def inside(self, point):
        if point[0] >= self.minx:
            if point[0] <= self.maxx:
                if point[1] >= self.miny:
                    if point[1] <= self.maxy:
                        return True
        return False

def decode(e):
    return map(lambda x: float(x), e.split(","))

def encode(d):
    return ",".join(map(lambda x: str(x), d))

def process_svg(svgpath, do_plt):
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

    points_to_path_map = {}
    path_to_points_map = {}
    connectivity_map = {}
    # assuming the constellation is stored in 1 path
    p_key = "#addf8a"
    if p_key not in color_gs:
        return False
    p_key_paths = map(lambda x: x["path"], color_gs[p_key])
    p_key_attrs = map(lambda x: attributes_copy[
        x["attrs_idx"]], color_gs[p_key])
    for i, path in enumerate(p_key_paths):
        transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if "transform" in p_key_attrs[i].keys():
            transform_str = str(p_key_attrs[i]["transform"])
            transform = get_transform_matrix(transform_str)

        path_points = path_to_points(path.d())
        last_point = []
        for j, p in enumerate(path_points):
            if (p["type"] == "M"):
                last_point = apply_matrix(transform,
                    p["data"])
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
                        linewidth=2, c='b', alpha=0.5)

                c = encode(last_point)
                points_to_path_map[c] = i
                Util.insert_into_dict_of_arrays(
                    path_to_points_map, i, c)

                d = encode(new_point)
                points_to_path_map[d] = i
                Util.insert_into_dict_of_arrays(
                    path_to_points_map, i, d)

                Util.insert_into_dict_of_arrays(
                    connectivity_map, c, d) # map stores paths from start to end

                last_point = new_point
    for path_id in path_to_points_map.keys():
        path_to_points_map[path_id] = list(
            set(path_to_points_map[path_id]))

    all_points = []
    tight_box_points = []
    o_points_to_path_map = {}
    o_path_to_points_map = {}
    # approach: get all points, take average to get center point
    # find the closest point in each 'quadrant', that is the tight box
    o_key = "#100f0d"
    if o_key not in color_gs:
        return False
    o_key_paths = map(lambda x: x["path"], color_gs[o_key])
    o_key_attrs = map(lambda x: attributes_copy[
        x["attrs_idx"]], color_gs[o_key])
    for i, path in enumerate(o_key_paths):
        # no transforms ever, just 1 path
        path_points = path_to_points(path.d())
        last_point = []
        for j, p in enumerate(path_points):
            if (p["type"] == "M"):
                last_point = p["data"]
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

                c = ",".join(map(lambda x: str(x), last_point))
                d = ",".join(map(lambda x: str(x), points[2]))
                all_points += [c,d]
                o_points_to_path_map[c] = i
                Util.insert_into_dict_of_arrays(
                    o_path_to_points_map, i, c)
                o_points_to_path_map[d] = i
                Util.insert_into_dict_of_arrays(
                    o_path_to_points_map, i, d)

                last_point = points[2]
            elif (p["type"] == "L"):
                point = p["data"]

                xs = [last_point[0], point[0]]
                ys = [last_point[1], point[1]]

                if do_plt:
                    plt.plot(
                        xs,
                        map(lambda x: -1 * x, ys),
                        # ys,
                        linewidth=2, c='b', alpha=0.5)

                c = ",".join(map(lambda x: str(x), last_point))
                d = ",".join(map(lambda x: str(x), point))
                all_points += [c,d]
                o_points_to_path_map[c] = i
                Util.insert_into_dict_of_arrays(
                    o_path_to_points_map, i, c)
                o_points_to_path_map[d] = i
                Util.insert_into_dict_of_arrays(
                    o_path_to_points_map, i, d)

                last_point = point
    all_points = list(set(all_points))
    all_points_decoded = []
    for p in all_points:
        all_points_decoded.append(
            map(lambda x:
                float(x), p.split(",")))
    xs = map(lambda x: x[0], all_points_decoded)
    ys = map(lambda x: -x[1], all_points_decoded)
    center_point = [(max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0]
    if do_plt:
        plt.scatter(center_point[0], -center_point[1], s=50, c='g', alpha=0.5)
    center_point = np.array(center_point)
    distances = [] # get the four points farthest from the center, these are the outer 'box'
    for i, p in enumerate(all_points_decoded):
        dist = round(np.linalg.norm(
            center_point-np.array(p)), 3)
        distances.append({"dist" : dist, "i" : i})
    distances = sorted(distances, key=lambda k: k["dist"])
    interested_points = map(lambda x: x["i"], distances[:len(distances)-4])
    filtered = []
    for i, p in enumerate(all_points_decoded):
        if i in interested_points:
            filtered.append(p)
    b = BoundingBox(filtered)
    if do_plt:
        plt.scatter([b.minx, b.maxx],
            map(lambda x: -x, [b.miny, b.maxy]),
            s=50, c='g', alpha=0.5)

    # go through points found in constellation, bounds checking them
    points_in_box = []
    paths_found_in_box = {}
    for p in points_to_path_map:
        decoded = map(lambda x: float(x), p.split(","))
        if b.inside(decoded):
            points_in_box.append(decoded)
            path = points_to_path_map[p]
            Util.insert_into_dict_of_arrays(
                paths_found_in_box, path, p)

    # pick the path with the greatest # of points found inside
    max_num_children = 0
    best_path = -1
    for k in paths_found_in_box:
        c = len(paths_found_in_box[k])
        if c > max_num_children:
            max_num_children = c
            best_path = k
    print best_path

    result = False
    if best_path in path_to_points_map:
        print "best path found! ", best_path
        xs = map(lambda x: decode(x)[0], path_to_points_map[best_path])
        ys = map(lambda x: -decode(x)[1], path_to_points_map[best_path])
        if do_plt:
            plt.scatter(xs, ys, s=50, c='r', alpha=0.5)

        directory = os.path.realpath(os.path.dirname(svgpath))
        data_pickle = directory + "/data.p"
        data_to_save = {}
        data_to_save["color_gs"] = color_gs
        data_to_save["p_key_paths"] = p_key_paths
        data_to_save["p_key_attrs"] = p_key_attrs
        data_to_save["points_to_path_map"] = points_to_path_map
        data_to_save["path_to_points_map"] = path_to_points_map
        data_to_save["connectivity_map"] = connectivity_map
        data_to_save["o_key_paths"] = o_key_paths
        data_to_save["o_key_attrs"] = o_key_attrs
        data_to_save["center_point"] = center_point
        pickle.dump(data_to_save, open(data_pickle, "wb"))
        print "dumped data_pickle! ", data_pickle

        result = True
    else:
        print "unable to find a best path!"

    if do_plt:
        # plt.show()
        directory = os.path.realpath(os.path.dirname(svgpath))
        plot_png = directory + "/plot.png"
        fig.savefig(plot_png)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "get points / paths of interest from constellation svg")
    parser.add_argument('--file', type=str, required=True, help='svg file name')
    parser.add_argument('--plt', type=bool, required=False, default=True)
    args = parser.parse_args()
    process_svg(args.file, args.plt)
