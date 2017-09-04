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
    points = map(lambda x: decode(x), points)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "get points in the constellation with 2 lines attached")
    parser.add_argument('--dir',
        type=str, required=True, help='data pickle path')
    args = parser.parse_args()
    res = process_svg_step_3(args.dir)
