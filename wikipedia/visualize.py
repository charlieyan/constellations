#!/usr/bin/env python

import argparse
from copy import deepcopy
import numpy as np
import sys, os, math
import cairo
import pickle
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.path import Path
import matplotlib.patches as patches

from common import *

Path = mpath.Path

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def visualize(cname, mirror = False):
    ensure(os.path.realpath("./data/"))
    data_pickle = "./data/all_segments.p"
    data_to_save = pickle.load(open(data_pickle, "rb"))
    all_segments = data_to_save["all_segments"]
    if cname not in all_segments:
        print "can't find cname in all segments, try again"
        for seg in sorted(all_segments.keys()):
            print seg
        return
    fig, ax = plt.subplots()
    segments = all_segments[cname]
    all_points = []
    for seg in segments:
        last_point = seg[0]
        point = seg[1]
        xs = [last_point[0], point[0]]
        ys = [last_point[1], point[1]]
        plt.plot(
            xs,
            map(lambda x: -1 * x, ys),
            linewidth=2, c='b', alpha=1.0)
        all_points.append(last_point)
        all_points.append(point)
    b = BoundingBox(all_points)

    centroid_x, centroid_y = b.centroid()
    plt.scatter(
        centroid_x, -centroid_y)

    plt.grid(True)

    if mirror:
        print "mirroring"
        ax.set_xlim(b.maxx, b.minx)
    else:
        ax.set_xlim(b.minx, b.maxx)
    ax.set_ylim(-b.maxy, -b.miny)
    ax.set_aspect('equal')
    plt.draw()
    ax.apply_aspect()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "plot constellation in 2D!")
    parser.add_argument('--cname', type=str, required=True,
        help='name of constellation, substitute _ for spaces')
    parser.add_argument('--mirror', type=bool, default=False,
        help='mirror the image or not')
    args = parser.parse_args()
    visualize(args.cname, args.mirror)

# examples
# ./visualize.py --cname Aquarius
# ./visualize.py --cname Orion