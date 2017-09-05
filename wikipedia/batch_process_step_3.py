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

def unicode_to_str(u):
    return unicodedata.normalize('NFKD', u).encode('ascii','ignore')

with open('constellations.json') as data_file:    
    data = json.load(data_file)

all_ratio_and_angle_map = {}
all_ratio_and_angle_map_2 = {}
svgs_dir = "./svgs/"
for i, c in enumerate(data):
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")
    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")
    print "3 on ", constellation_dir
    _, r_and_a, r_and_a_2\
        = process_svg_step_3(constellation_dir)
    all_ratio_and_angle_map[c_name] = r_and_a
    for k in r_and_a_2:
        all_ratio_and_angle_map_2[k] =\
            c_name + ":" + r_and_a_2[k]

ensure(os.path.realpath("./data/"))
data_pickle = "./data/all_ratio_and_angle_map.p"
data_to_save = {}
data_to_save["all_ratio_and_angle_map"]\
    = all_ratio_and_angle_map
data_to_save["all_ratio_and_angle_map_2"]\
    = all_ratio_and_angle_map_2
pickle.dump(data_to_save, open(data_pickle, "wb"))
print "dumped data_pickle! ", data_pickle