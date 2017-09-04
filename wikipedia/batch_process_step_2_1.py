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
results = []
good_constellations = []
bad_constellations = []

for i, c in enumerate(data):
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")

    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")

    data_p = constellation_dir + "/step_1.p"
    data_p = os.path.realpath(data_p)

    if os.path.isfile(data_p):
        print "2_1 on ", constellation_dir
        res = process_svg_step_2_1(data_p)
        if not res:
            print "something failed: ", constellation_dir
            bad_constellations.append(constellation_dir)
        else:
            good_constellations.append(constellation_dir)
        results.append(res)