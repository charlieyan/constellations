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

from process_step_1 import *

svgs_dir = "./svgs/"
svgs_mkdir_cmd = "mkdir " + svgs_dir
os.system(svgs_mkdir_cmd)

def unicode_to_str(u):
    return unicodedata.normalize('NFKD', u).encode('ascii','ignore')

with open('constellations.json') as data_file:    
    data = json.load(data_file)

results = []
constellation_svgs = []
good_constellations = []
bad_constellations = []

for i, c in enumerate(data):
    # c_name = unicode_to_str(c["name"])
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")

    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")
    constellation_svg = constellation_dir + "source.svg"

    data_p = constellation_dir + "/data.p"
    if os.path.isfile(data_p):
        good_constellations.append(
            constellation_svg)
        continue

    print "processing: ", constellation_svg
    res = process_svg_step_1(constellation_svg, True)
    if not res:
        print "something failed: ", constellation_svg
        bad_constellations.append(constellation_svg)
    else:
        good_constellations.append(constellation_svg)
    results.append(res)
