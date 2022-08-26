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
parser.add_argument("--float32", help="Output float32 array regardless dtype of input data.",  action="store_true")
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
	if not ARG.float32:
		frame = np.zeros([len(inputTifFiles), dimy, dimx], dtype=dtype)
	if ARG.float32:
		DEN.writeEmptyDEN(denFile, [dimx, dimy, len(inputTifFiles)], force=True)
	for i in range(len(inputTifFiles)):
		f = inputTifFiles[i]
		im = Image.open(f)
		img = np.array(im)
		if img.shape[0] != dimy or img.shape[1] != dimx:
			raise IOError("File %s shape (%d, %d) does not agree with expected (%d, %d) of %s"%(f, img.shape[0], img.shape[0], dimx, dimy, inputTifFiles[0]))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s"%(f, img.dtype, dtype))
		if ARG.float32:
			DEN.writeFrame(denFile, i, img.astype(np.float32), True)
		else:
			frame[i] = img
	if not ARG.float32:
		DEN.storeNdarrayAsDEN(denFile, frame)

if ARG.verbose:
	print("Start of the script")
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force)
