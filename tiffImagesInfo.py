#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
from bisect import bisect_left
import h5py
import pandas as pd
import matplotlib.pyplot as plt
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

parser = argparse.ArgumentParser()
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("--h5file", type=str, help="H5 file to read data for exposure correction and for current correction.", required=True)
parser.add_argument("--current", action="store_true")
parser.add_argument("--verbose", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()

f = h5py.File(ARG.h5file, 'r')

#exposureData = f["entry/hardware/camera"]

beamCurrent = PETRA.beamCurrentDataset(ARG.h5file)
scanData = PETRA.scanDataset(ARG.h5file)
if ARG.verbose:
	beamCurrent["current"].plot()
	plt.show()
	print(scanData[["image_file", "current"]])

if ARG.current:
	for f in ARG.inputTifFiles:
		fileName = os.path.basename(f)
		print("File %s estimated beam current %0.3fmA."%(fileName, scanData[scanData["image_file"]==fileName]["current"]))
