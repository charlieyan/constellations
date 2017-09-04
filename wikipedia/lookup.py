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

parser = argparse.ArgumentParser(description = 
    "get points in the constellation with 2 lines attached")
parser.add_argument('--angle',
    type=float, required=True, help='angle')
parser.add_argument('--ratio',
    type=float, required=True, help='ratio')
args = parser.parse_args()

svgs_dir = "./svgs/"
data_pickle = svgs_dir + "/all_ratio_and_angle_map.p"
data_to_save = pickle.load(open(data_pickle, "rb"))
r_and_a = data_to_save["all_ratio_and_angle_map"]
r_and_a_2 = data_to_save["all_ratio_and_angle_map_2"]
 # 770 unique ratio / angles found
