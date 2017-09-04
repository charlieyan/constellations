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

def plot_path(d_str, transform, c):
    path_points = path_to_points(d_str)
    last_point = []
    all_segments = []
    for j, p in enumerate(path_points):
        if (p["type"] == "M"):
            last_point = apply_matrix(transform,
                p["data"])
        elif (p["type"] == "C"):
            points = p["data"]
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
            plt.plot(
                xs,
                map(lambda x: -1 * x, ys),
                # ys,
                linewidth=2, c=c, alpha=0.5)
            all_segments.append([last_point, new_point])
            last_point = new_point
    return all_segments

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

def find_n_closest_point(points, point, n):
    distances = []
    for i, p in enumerate(points):
        dist = round(np.linalg.norm(
            point-np.array(p)), 3)
        distances.append({"dist" : dist, "i" : i})
    distances = sorted(distances,
        key=lambda k: k["dist"])
    indices = map(lambda x: x["i"], distances[:n])
    filtered = []
    for i, p in enumerate(points):
        if i in indices:
            filtered.append(p)
    return filtered

def sorted_and_indices(array, cb):
    array = sorted(array, key = lambda k: cb(k))
    indices = sorted(range(len(array)),
        key=lambda k: cb(array[k]))
    return array, indices

def cb_for_points(point):
    return encode(point)

def passthrough(d):
    return d