#!/usr/bin/env python

import json
from pprint import pprint
import unicodedata    
import requests
from lxml import html

import urllib2
from bs4 import BeautifulSoup
import os
from os import listdir
from os.path import isfile, join

import re, urlparse

def urlEncodeNonAscii(b):
    return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)

def unicode_to_str(u):
    return unicodedata.normalize('NFKD', u).encode('ascii','ignore')

with open('constellations.json') as data_file:    
    data = json.load(data_file)

svgs_dir = "./svgs/"
svgs_mkdir_cmd = "mkdir " + svgs_dir

for i, c in enumerate(data):
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")
    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")
    constellation_dir = os.path.realpath(os.path.dirname(
        constellation_dir))

    onlyfiles = [f for f in listdir(constellation_dir) if isfile(
        join(constellation_dir, f))]

    for f in onlyfiles:
        if f != "source.svg":
            f_full_path = constellation_dir + "/" + f
            rm_cmd = "rm -rf " + f_full_path
            os.system(rm_cmd)