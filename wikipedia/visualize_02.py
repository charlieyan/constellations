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
import copy

from wall_borg.interaction.utils import *
from wall_borg.interaction.sidefinding import *

import networkx as nx

Path = mpath.Path

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def visualize(cname, mirror = False, debug = False):
    ensure(os.path.realpath("./data/"))
    data_pickle = "./data/all_segments.p"
    data_to_save = pickle.load(open(data_pickle, "rb"))
    all_segments = data_to_save["all_segments"]
    if cname not in all_segments:
        print "can't find cname in all segments, try again"
        for cname in sorted(all_segments.keys()):
            print cname
        return
    fig, ax = plt.subplots()
    segments = all_segments[cname]

    all_points = []
    all_obstacles = []
    obstacles_dedup = []
    for seg in segments:
        last_point = seg[0]
        last_point[1] = -last_point[1]
        point = seg[1]
        # point = [point[0], point[1]]
        # point[1] = -point[1]
        new_point = copy.copy(point) # for some reason need this
        new_point[1] = -new_point[1]

        xs = [last_point[0], point[0]]
        ys = [last_point[1], new_point[1]]
        plt.plot(
            xs, ys,
            linewidth=2, c='b', alpha=1.0)

        new = sorted([map(lambda x: round(x), last_point), map(lambda x: round(x), new_point)])
        dist = math.sqrt((new[0][0] - new[1][0])**2 + (new[0][1] - new[1][1])**2)
        if new not in obstacles_dedup and dist > 1e-1:
            # turns out there is a LOT of duplicate / redundancy
            # of lines and of lines that just go to itself
            all_points.append(last_point)
            all_points.append(new_point)

            all_obstacles.append(RectObstacle(new))
            obstacles_dedup.append(new)

    # print "all_lines", all_obstacles
    adjacency_map = get_nearest_neighbor_map(all_obstacles)
    all_keys = adjacency_map.keys()[:]
    all_keys = sorted(all_keys)
    graph = build_adj_graph_strategy_03(all_obstacles, adjacency_map, all_keys)
    traversals, transitions, results = generate_traversals_01(graph, all_obstacles)
    graph_sides = traversals_to_sides(traversals, transitions,
        results,
        all_obstacles,
        GeoUtil.two_d_make_x_y_theta_hom(-0.1, 0.0, 0.0))

    for side_i, side in enumerate(graph_sides):
        individual_offset = GeoUtil.two_d_make_x_y_theta_hom(-1 * (side_i+1), 0.0, 0.0)
        graph_sides[side_i]["line_world_homs"] = map(lambda h: h.dot(individual_offset),
            graph_sides[side_i]["line_world_homs"])
    # print "len(graph_sides)", len(graph_sides)

    # plotting
    b = BoundingBox(all_points)
    centroid_x, centroid_y = b.centroid()
    plt.scatter(
        centroid_x, centroid_y)

    colors = 'grmyc'
    for k_i, k in enumerate(graph_sides):
        print "k", k_i
        c = colors[k_i % len(colors)]
        # print "c", c

        homs = k["line_world_homs"]
        for h_i, hom in enumerate(homs):
            # if k_i < 2:
            #     continue

            # tools_2d.plot_gnomon(plt,
            #     hom,
            #     2, 1.0, linewidth = 2, c = c)

            corner = k["line_world_lr_corners"][h_i]
            lw = 2.5
            if corner in k["corners_in_rooms"]:
                lw = 6.5
            l_width = line_width(all_obstacles, corner)

            l_w_offset_hom = GeoUtil.two_d_make_x_y_theta_hom(0.0, l_width, 0.0)
            other_hom = hom.dot(l_w_offset_hom)
            plt.plot(
                [hom[0, 2], other_hom[0, 2]],
                [hom[1, 2], other_hom[1, 2]],
                color=c,
                linewidth=lw)

        # if k_i > 0:
        #     break

    plt.grid(True)
    xlim = []
    if mirror:
        print "mirroring"
        xlim = [b.maxx, b.minx]
    else:
        xlim = [b.minx, b.maxx]
    padding = 50
    xlim[0] -= padding
    xlim[1] += padding
    ax.set_xlim(xlim)
    ax.set_ylim(b.miny - padding, b.maxy + padding)
    ax.set_aspect('equal')
    plt.draw()
    ax.apply_aspect()

    if debug:
        plt.figure(2)
        plt.subplot(111)

        g = nx.Graph()
        g.add_nodes_from(graph.nodes.keys())
        for e in graph.edges:
            g.add_edge(e[0], e[1])
        node_colors = ['r'] * len(g.nodes.keys())

        explicit_pos = {}
        for n in graph.nodes.keys():
            vertex_id = graph.nodes[n].data[0]
            vertex_id_tokens = map(lambda x: int(x), vertex_id.split("_"))
            xy = all_obstacles[vertex_id_tokens[0]].corner_pts[vertex_id_tokens[1]]
            explicit_pos[n] = xy
        print "# of vertices", len(adjacency_map.keys())
        print "# of nodes", len(graph.nodes.keys())

        nx.draw(g, explicit_pos, node_size=100, alpha=0.4,
            edge_color='b', font_size=16, with_labels=True,
            node_color=node_colors)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "plot constellation in 2D!")
    parser.add_argument('--cname', type=str, default="",
        help='name of constellation, substitute _ for spaces')
    parser.add_argument('--mirror', type=bool, default=False,
        help='mirror the image or not')
    parser.add_argument('--debug', type=bool, default=False, help="debug info")
    args = parser.parse_args()
    visualize(args.cname, args.mirror, args.debug)

# examples
# ./visualize.py --cname Aquarius
# ./visualize.py --cname Orion