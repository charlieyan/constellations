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

data_pickle = "./svgs/Corvus/step_1.p"
directory = os.path.realpath(os.path.dirname(data_pickle))
data_to_save = pickle.load(open(data_pickle, "rb"))
points = data_to_save["points_to_path_map"].keys()
points_to_path = data_to_save["points_to_path_map"]
transforms = data_to_save["path_to_transform_map"]

s_p_key_paths = data_to_save["p_key_paths"]
s_p_key_attrs = data_to_save["p_key_attrs"]

def path1_is_contained_in_path2(path1, path2):
    if path2.intersect(path1):
        return True

    # find a point that's definitely outside path2
    xmin, xmax, ymin, ymax = path2.bbox()
    B = (xmin + 1) + 1j*(ymax + 1)

    A = path1.start  # pick an arbitrary point in path1
    AB_line = Path(Line(A, B))
    number_of_intersections = len(AB_line.intersect(path2))
    if number_of_intersections % 2:  # if number of intersections is odd
        return True
    else:
        return False

path_1 = s_p_key_paths[1]
path_2 = s_p_key_paths[2]
