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
from process_step_2 import *

def unicode_to_str(u):
    return unicodedata.normalize('NFKD', u).encode('ascii','ignore')

with open('constellations.json') as data_file:    
    data = json.load(data_file)

svgs_dir = "./svgs/"
all_segments = {}
for i, c in enumerate(data):
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")
    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")
    data_p = constellation_dir + "/step_1.p"
    data_p = os.path.realpath(data_p)
    if os.path.isfile(data_p):
        print "2_2 on ", constellation_dir
        segments = process_svg_step_2_2(data_p)
        all_segments[c_name] = segments

ensure(os.path.realpath("./data/"))
data_pickle = "./data/all_segments.p"
data_to_save = {}
data_to_save["all_segments"] = all_segments
pickle.dump(data_to_save, open(data_pickle, "wb"))
print "dumped data_pickle! ", data_pickle