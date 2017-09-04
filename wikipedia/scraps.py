
    o_key = "#100f0d"
    all_points = []
    tight_box_points = []
    o_points_to_path_map = {}
    o_path_to_points_map = {}
    # approach: get all points, take average to get center point
    # find the closest point in each 'quadrant', that is the tight box
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

    if best_path in path_to_points_map:
        print "best path found! ", best_path
        xs = map(lambda x: decode(x)[0], path_to_points_map[best_path])
        ys = map(lambda x: -decode(x)[1], path_to_points_map[best_path])
        if do_plt:
            plt.scatter(xs, ys, s=50, c='r', alpha=0.5)

        directory = os.path.realpath(os.path.dirname(svgpath))
        data_pickle = directory + "/step_1.p"
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

from ego_atheris.interaction.utils import Util

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def get_transform_matrix(s):
    numbers = map(lambda x: float(x),
        s.replace("matrix(", "").replace(")", "").split(","))
    return numbers
    # numbers = np.array(numbers + [0.0, 0.0, 1.0]).reshape(3, 3)
    # return numbers

def clean_up_filename(s):
    return s.replace(".svg", "")

def path_to_points(d_str):
    # my own points
    tokens = d_str.split(" L ")
    tokens = map(lambda x: x.replace("M ", ""), tokens)

    points = []
    for i, t in enumerate(tokens):
        coords = map(lambda x: float(x), t.split(","))
        points.append(coords)
    return points

def svg_path_to_points(path_obj):
    points = []
    flattened = path_obj.flatten()
    for i, f in enumerate(flattened):
        f_class = f.__class__.__name__
        if "MoveTo" in f_class:
            point = [f.dest.x, f.dest.y]
            points.append(point)
        elif "Segment" in f_class:
            point = [f.end.x, f.end.y]
            points.append(point)

    points = list(reversed(points))

    return points

def apply_matrix(matrix, point):
    x = point[0]
    y = point[1]
    new_x = x * matrix[0] + y * matrix[2] + matrix[4]
    new_y = x * matrix[1] + y * matrix[3] + matrix[5]
    return [new_x, new_y]

parser = argparse.ArgumentParser(description = 
    "get points / paths of interest from constellation svg")
parser.add_argument('--file', type=str, required=True, help='svg file name')
args = parser.parse_args()
paths, attributes = svg2paths(args.file)

attributes_copy = deepcopy(attributes)
color_gs = {}
for i, d in enumerate(attributes):
    if "style" in d.keys():
        if "rgb" in d["style"]:
            d["attrs_idx"] = i
            d["path"] = paths[i]
            d["rgb"] = str(d["style"]).split(
                "rgb")[1].split(")")[0].replace("(","")
            Util.insert_into_dict_of_arrays(
                color_gs, d["rgb"], d)

# convert all colors to rgb
key = color_gs.keys()[0]
key_paths = map(lambda x: x["path"], color_gs[key])
key_attrs = map(lambda x: attributes_copy[
    x["attrs_idx"]], color_gs[key])

combined_paths = svgpathtools.path.concatpaths(key_paths)
wsvg(combined_paths, filename="combined.svg")

fnames = []
for i, path in enumerate(key_paths):
    fname = str(i) + ".svg"
    wsvg(path, filename=fname)
    fnames.append(fname)

xs = None
ys = None
xy_one = None
res = None
m = None
transformed_path_objs = []
ms = []

for i, path in enumerate(key_paths):
    transform_str = str(key_attrs[i]["transform"])
    transform = get_transform_matrix(transform_str)

    m = svg.Matrix(transform)
    svg_obj = svg.parse(fnames[i])
    path_obj = svg_obj.items[0].items[0]
    path_obj.transform(m)
    path_obj_points = svg_path_to_points(path_obj)
    xs = map(lambda x: x[0], path_obj_points)
    ys = map(lambda x: x[1], path_obj_points)

    path_points = path_to_points(path.d())
    new_path_points = map(lambda x: apply_matrix(
        transform, x), path_points)
    xs = map(lambda x: x[0], new_path_points)
    ys = map(lambda x: x[1], new_path_points)

#     l = len(xs)
#     o = [1.0] * l
#     xy_one = np.array(xs + ys + o).reshape(3, l)
#     test = np.dot(m, xy_one)
#     xs = test[0,:].tolist()
#     ys = test[1,:].tolist()

    plt.plot(
        xs, map(lambda x: -1 * x, ys),
        linewidth=2, c='r', alpha=0.5)
plt.show()

# xs = map(lambda x: x["coords"][0], star_map)
# ys = map(lambda x: x["coords"][1], star_map)
# areas = map(lambda x: x["coords"][2], star_map)
# colors = np.random.rand(len(xs))
# plt.scatter(xs, ys, s=areas, c=colors, alpha=0.5)

# line_ys = ys[:7]
# line_xs = xs[:7]
# line_areas = [70]*7
# plt.scatter(line_xs, line_ys, s=line_areas, c='r', alpha=0.5)





def svg_path_to_points(path_obj):
    points = []
    flattened = path_obj.flatten()
    for i, f in enumerate(flattened):
        f_class = f.__class__.__name__
        if "MoveTo" in f_class:
            point = [f.dest.x, f.dest.y]
            points.append(point)
        elif "Segment" in f_class:
            point = [f.end.x, f.end.y]
            points.append(point)

    points = list(reversed(points))

    return points


    m = svg.Matrix(transform)
    svg_obj = svg.parse(fnames[i])
    path_obj = svg_obj.items[0].items[0]
    path_obj.transform(m)
    path_obj_points = svg_path_to_points(path_obj)
    xs = map(lambda x: x[0], path_obj_points)
    ys = map(lambda x: x[1], path_obj_points)


for k in color_gs.keys():
    key_paths = map(lambda x: x["path"], color_gs[k])
    wsvg(key_paths, filename=k+".svg")

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
    tokens = d_str.split(" L ")
    tokens = map(lambda x: x.replace("M ", ""), tokens)
    for i, t in enumerate(tokens):
        coords = map(lambda x: float(x), t.split(","))
        points.append(coords)
    return points

def path_to_points_2(d_str):
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

parser = argparse.ArgumentParser(description = 
    "get points / paths of interest from constellation svg")
parser.add_argument('--file', type=str, required=True, help='svg file name')
args = parser.parse_args()
paths, attributes = svg2paths(args.file)

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

for k in color_gs.keys():
    if k not in ["#addf8a", "#100f0d"]:
        continue
    key_paths = map(lambda x: x["path"], color_gs[k])
    wsvg(key_paths, filename=k+".svg")

fig, ax = plt.subplots()

p_key = "#addf8a"
p_key_paths = map(lambda x: x["path"], color_gs[p_key])
p_key_attrs = map(lambda x: attributes_copy[
    x["attrs_idx"]], color_gs[p_key])
for i, path in enumerate(p_key_paths):
    transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if "transform" in p_key_attrs[i].keys():
        transform_str = str(p_key_attrs[i]["transform"])
        transform = get_transform_matrix(transform_str)

    path_points = path_to_points_2(path.d())
    last_point = []
    for i, p in enumerate(path_points):
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
                linewidth=2, c='b', alpha=0.5)
            last_point = new_point

o_key = "#100f0d"
o_key_paths = map(lambda x: x["path"], color_gs[o_key])
o_key_attrs = map(lambda x: attributes_copy[
    x["attrs_idx"]], color_gs[o_key])
for i, path in enumerate(o_key_paths):
    # no transforms ever
    path_points = path_to_points_2(path.d())
    last_point = []
    for i, p in enumerate(path_points):
        if (p["type"] == "M"):
            last_point = p["data"]
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
            xs = [last_point[0], point[0]]
            ys = [last_point[1], point[1]]
            plt.plot(
                xs,
                map(lambda x: -1 * x, ys),
                # ys,
                linewidth=2, c='b', alpha=0.5)
            last_point = point
plt.show()


def path_to_points(d_str):
    points = []
    tokens = d_str.split(" L ")
    tokens = map(lambda x: x.replace("M ", ""), tokens)
    for i, t in enumerate(tokens):
        coords = map(lambda x: float(x), t.split(","))
        points.append(coords)
    return points

colors_of_interest = ["#addf8a", "#100f0d"]
for key in colors_of_interest:
    p_key_paths = map(lambda x: x["path"], color_gs[key])
    p_key_attrs = map(lambda x: attributes_copy[
        x["attrs_idx"]], color_gs[key])
    for i, path in enumerate(p_key_paths):
        transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if "transform" in p_key_attrs[i].keys():
            transform_str = str(p_key_attrs[i]["transform"])
            transform = get_transform_matrix(transform_str)

        path_points = path_to_points(path.d())
        last_point = []
        for i, p in enumerate(path_points):
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
                    linewidth=2, c='b', alpha=0.5)
                last_point = new_point
plt.show()

# plt.scatter(furthest_point[0], -furthest_point[1], s=200, c='r', alpha=0.5)
# path = points_to_path_map[",".join(map(lambda x: str(x), furthest_point))]
# all_rejected_points = path_to_points_map[path]
# print "all_rejected_points: ", all_rejected_points

# not_rejected = []
# for p in all_rejected_points:
#     if p not in all_rejected_points:
#         print "p: ", p
#         decoded = map(lambda x: float(x), p.split(","))
#         not_rejected.append(decoded)
# xs = map(lambda x: x[0], not_rejected)
# ys = map(lambda x: -x[1], not_rejected)
# plt.scatter(xs, ys, s=50, c='r', alpha=0.5)

xs = map(lambda x: x[0], points_in_box)
ys = map(lambda x: -x[1], points_in_box)
plt.scatter(xs, ys, s=50, c='r', alpha=0.5)




original_paths = []
valid_paths = {}
paths_to_invalidate = []
for p in points:
    paths = points_to_path[p]
    l = len(paths)
    if l > 1:
        transforms_for_paths = []
        for p in paths:
            original_paths.append(p)
            paths_to_invalidate.append(p)
            transform = encode(transforms[p])
            transforms_for_paths.append(transform)
        print "transforms_for_paths: ", transforms_for_paths
        p_combined = " ".join(paths)
        valid_paths[p_combined] = {
            "transforms" : transforms_for_paths,
            "paths" : paths
        }
    else:
        transform = encode(transforms[paths[0]])
        valid_paths[paths[0]] = {
            "transforms" : [transform],
            "paths" : paths
        }
        original_paths.append(paths[0])

original_paths = list(set(original_paths))
print "len(original_paths): ", len(original_paths)

output_paths = []
for p in valid_paths:
    if p in paths_to_invalidate:
        continue
    else:
        output_paths.append(p)
output_paths = list(set(output_paths))
print "len(output_paths): ", len(output_paths)

cmap = plt.get_cmap('hsv')
colors = cmap(np.linspace(0, 1, len(output_paths)))

fig, ax = plt.subplots()
for i, p in enumerate(output_paths):
    d = valid_paths[p]
    paths = d["paths"]
    transforms = d["transforms"]
    for j, x in enumerate(paths):
        print type(paths[j])
        temp = decode(transforms[j])
        plot_path(paths[j], temp, colors[i])

plt.show()

# decoded_points = map(lambda x: decode(x), points)