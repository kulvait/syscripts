#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
import statistics
from PIL import Image
from PIL.TiffTags import TAGS
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
from denpy import PETRA
import glob
import numpy as np
from termcolor import colored
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('inputTif', help="Input TIFF file", type=str)
parser.add_argument('outputTif', help="Output TIFF file", type=str)
parser.add_argument("--force", action="store_true")
parser.add_argument("--x-from", type=int, default=0)
parser.add_argument("--x-to", type=int, default=None)
parser.add_argument("--y-from", type=int, default=0)
parser.add_argument("--y-to", type=int, default=None)

#ARG = parser.parse_args([])
ARG = parser.parse_args()

if os.path.exists(ARG.outputTif):
	if force:
		os.remove(ARG.outputTif)
	else:
		raise IOError("File %s exists, add force to overwrite" %
		              (ARG.outputTiff))
im0 = Image.open(ARG.inputTif)
img = np.array(im0)
dimy = img.shape[0]
dimx = img.shape[1]
x_from = ARG.x_from
y_from = ARG.y_from
if ARG.x_to is None:
	x_to = dimx
else:
	x_to = ARG.x_to
if ARG.y_to is None:
	y_to = dimy
else:
	y_to = ARG.y_to
img = img[y_from:y_to, x_from:x_to]
im1 = Image.fromarray(img)
im1.save(ARG.outputTif)
