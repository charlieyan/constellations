#!/usr/bin/env python

import os
import json
from pprint import pprint
import unicodedata    
import requests
from lxml import html

import urllib2
from bs4 import BeautifulSoup
import os

import re, urlparse

from process_step_3 import *
import numpy as np
import math

def closest_in_r_and_as(r_and_as, ratio, angle):
    cost = np.inf
    best_idx = 0
    for i, r_and_a in enumerate(r_and_as):
        c = abs(r_and_a[0] - ratio) + abs(r_and_a[1] - angle)
        if c < cost:
            best_idx = i
            cost = c
    return r_and_as[best_idx], cost

data_dir = "./data/"
data_pickle = data_dir + "/all_ratio_and_angle_map.p"
data_to_save = pickle.load(open(data_pickle, "rb"))
r_and_a = data_to_save["all_ratio_and_angle_map"]
r_and_a_2 = data_to_save["all_ratio_and_angle_map_2"]

sorted_keys, _ = sorted_and_indices(r_and_a_2.keys(), passthrough)
r_and_as = []
for k in sorted(sorted_keys):
    r_and_a = map(lambda x: float(x), k.split(":"))
    r_and_as.append(r_and_a)
 # 770 unique ratio / angles found

leg_1_length = 18.1 # mm 'c'
leg_2_length = 9.57 # mm 'a'
leg_3_length = 12.93 # mm 'b'

legs = [leg_1_length, leg_2_length, leg_3_length]
a = legs[1]
b = legs[2]
c = legs[0]
cos_c = (a**2 + b**2 - c**2) / (2 * a * b)
angle_c = math.acos(cos_c)
sin_a = a * math.sin(angle_c) / c
angle_a = math.asin(sin_a)
angle_a = math.degrees(angle_a)
angle_c = math.degrees(angle_c)
angle_b = 180.0 - (angle_a + angle_c)
angle_map = {}
angle_map["1_2"] = angle_c
angle_map["0_1"] = angle_b
angle_map["0_2"] = angle_a
i_selected = []

overall_best_cost = np.inf
best_key = ""
for i, l in enumerate(legs):
    i_selected.append(i)
    for j, k in enumerate(legs):
        if i != j and j not in i_selected:
            ratio = round(min(legs[i], legs[j]) / max(legs[i], legs[j]), 3)
            k = map(lambda x: str(x), sorted([i,j]))
            k = "_".join(k)
            angle = round(angle_map[k], 3)
            # just 3 choose 2 of legs

            key, best_cost = closest_in_r_and_as(
                r_and_as, ratio, angle)
            if best_cost < overall_best_cost:
                overall_best_cost = best_cost
                best_key = key
            print "ratio: ", ratio
            print "angle: ", angle
            print "best key: ", key
            print "best_cost: ", best_cost

best_key = map(lambda x: str(x), best_key)
best_key = ":".join(best_key)
print "best_key: ", best_key
location = r_and_a_2[best_key]
print "where it lives: ", location

svgs_dir = "./svgs/"
cname = location.split(":")[0]
constellation_dir = svgs_dir + cname + "/"
lookup_arg = location.split(":")[1]
lookup_cmd = "./process_step_3.py --dir "\
    + constellation_dir + " --lookup " + lookup_arg
print "after running the pipeline on the specific constellation, run this command: "
print lookup_cmd
# os.system(lookup_cmd)
# expected_png = constellation_dir + "step_3_" + lookup_arg + ".png"
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img=mpimg.imread(expected_png)
# imgplot = plt.imshow(img)
# plt.show()