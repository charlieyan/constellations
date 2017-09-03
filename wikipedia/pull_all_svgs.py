#!/usr/bin/env python

import json
from pprint import pprint
import unicodedata    
import requests
from lxml import html

import urllib2
from bs4 import BeautifulSoup
import os

import re, urlparse

def urlEncodeNonAscii(b):
    return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)

def iriToUri(iri):
    parts= urlparse.urlparse(iri)
    return urlparse.urlunparse(
        part.encode('idna') if parti==1 else urlEncodeNonAscii(
            part.encode('utf-8'))
        for parti, part in enumerate(parts)
    )

def unicode_to_str(u):
    return unicodedata.normalize('NFKD', u).encode('ascii','ignore')

with open('constellations.json') as data_file:    
    data = json.load(data_file)

svgs_dir = "./svgs/"
svgs_mkdir_cmd = "mkdir " + svgs_dir
os.system(svgs_mkdir_cmd)

for i, c in enumerate(data):
    # c_name = unicode_to_str(c["name"])
    c_name = c["name"]
    c_name = c_name.replace(" ", "_")
    link = iriToUri(
        "https://commons.wikimedia.org/wiki/File:"
        +c_name+"_IAU.svg")

    page = urllib2.urlopen(link)
    soup = BeautifulSoup(page, 'html.parser')
    img = soup.find('img')
    src_str = unicode_to_str(img.attrs["src"])
    src_str = src_str.replace("thumb/", "")

    cut_spot = src_str.index(".svg/")
    src_str = src_str[:cut_spot] + ".svg"

    constellation_dir = unicode_to_str(svgs_dir + c_name + "/")
    mkdir_cmd = "mkdir " + constellation_dir
    os.system(mkdir_cmd)

    constellation_svg = constellation_dir + "source.svg"

    wget_cmd = "wget -O " + constellation_svg + " " + src_str
    os.system(wget_cmd)


