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

def get_points(cname):
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
    return all_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "plot constellation in 2D!")
    parser.add_argument('--cname', type=str, required=True,
        help='name of constellation, substitute _ for spaces')
    args = parser.parse_args()
    points = get_points(args.cname)
    rounded = map(lambda y: map(lambda x: int(x), y), points)
    rounded_to_originals = {}
    for i, p in enumerate(rounded):
        insert_into_dict_of_arrays(
            rounded_to_originals, encode(p), points[i])
    rounded_to_avgs = {}
    for r in rounded_to_originals:
        points = rounded_to_originals[r]
        xs = map(lambda x: x[0], points)
        ys = map(lambda x: x[1], points)
        avg_x = round(np.mean(xs), 3)
        avg_y = round(np.mean(ys), 3)
        avg = [avg_x, avg_y]
        rounded_to_avgs[r] = avg

# examples
# ./get_points.py --cname Aquarius
# ./get_points.py --cname Orion