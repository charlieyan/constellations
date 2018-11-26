#!/usr/bin/env python

import argparse
from copy import deepcopy
import numpy as np
import sys, os, math
import cairo
import pickle
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as patches

from common import *
import copy

from wall_borg.interaction.utils import *
from wall_borg.interaction.sidefinding import *
# from c_01_build_adjacency_graph_dev import *
import networkx as nx

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

if __name__ == "__main__":
    ensure(os.path.realpath("./data/"))

    data_pickle = "./data/all_segments.p"
    data_to_save = pickle.load(open(data_pickle, "rb"))
    all_segments = data_to_save["all_segments"]

    for cname_i, cname in enumerate(sorted(all_segments.keys())):
        if "tes" in cname:
            print cname
            data_to_save["all_segments"]["Bootes"] = data_to_save["all_segments"][cname]
            data_to_save["all_segments"].pop(cname, None)

    data_pickle = "./data/all_segments_bootes.p"
    pickle.dump(data_to_save, open(data_pickle, "wb"))