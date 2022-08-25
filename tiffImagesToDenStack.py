#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
from PIL import Image
from PIL.TiffTags import TAGS
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()


#To write dataframe to den
def writeDenFile(inputTifFiles, denFile, force=False):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	im0 = Image.open(inputTifFiles[0])
	img = np.array(im0)
	dimy = img.shape[0]
	dimx = img.shape[1]
	dtype = img.dtype
	frame = np.zeros([len(inputTifFiles), dimy, dimx], dtype=dtype)
	for i in range(len(inputTifFiles)):
		im = Image.open(inputTifFiles[i])
		img = np.array(im)
		if img.shape[0] != dimx:
			raise IOError("File %s shape %d does not agree with expected %d"%(file, img.shape[0], dimx))
		if img.shape[1] != dimy:
			raise IOError("File %s shape %d does not agree with expected %d"%(file, img.shape[1], dimy))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s"%(file, img.dtype, dtype))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s"%(file, img.dtype, dtype))
		frame[i] = img
	DEN.storeNdarrayAsDEN(denFile, frame)

if ARG.verbose:
	print("Start of the script")
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force)
