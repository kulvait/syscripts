#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import copy
import imageio.v3 as iio
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("outputDen", type=str)
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()



def handleFileExistence(denFile, force):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))

#To write dataframe to den
def writeDenFile(inputTifFiles, denFile, force=False):
	fileTokens = denFile.rsplit(".", 1)
	realPartTokens = copy.deepcopy(fileTokens)
	imPartTokens = copy.deepcopy(fileTokens)
	realPartTokens[0] = realPartTokens[0] + "_re"
	imPartTokens[0] = imPartTokens[0] + "_im"
	imaFile = ".".join(imPartTokens)
	reaFile = ".".join(realPartTokens);
	print("%s rePart=%s imPart=%s."%(denFile, reaFile, imaFile))
	handleFileExistence(imaFile, force)
	handleFileExistence(reaFile, force)
	img = iio.imread(inputTifFiles[0])
	dimy = img.shape[0]
	dimx = img.shape[1]
	dtype = img.dtype
	realdtype = np.imag(img).dtype
	frameRe = np.zeros([len(inputTifFiles), dimy, dimx], dtype=realdtype)
	frameIm = np.zeros([len(inputTifFiles), dimy, dimx], dtype=realdtype)
	for i in range(len(inputTifFiles)):
		im = iio.imread(inputTifFiles[i])
		img = np.array(im)
		if img.shape[0] != dimx:
			raise IOError("File %s shape %d does not agree with expected %d"%(file, img.shape[0], dimx))
		if img.shape[1] != dimy:
			raise IOError("File %s shape %d does not agree with expected %d"%(file, img.shape[1], dimy))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s"%(file, img.dtype, dtype))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s"%(file, img.dtype, dtype))
		frameRe[i] = np.real(img)
		frameIm[i] = np.imag(img)
	DEN.storeNdarrayAsDEN(reaFile, frameRe)
	DEN.storeNdarrayAsDEN(imaFile, frameIm)

if ARG.verbose:
	print("Start of the script")
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force)
