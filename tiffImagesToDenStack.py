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

parser = argparse.ArgumentParser()
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--float32", help="Output float32 array regardless dtype of input data.",  action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--h5file", type=str, help="H5 file to read data for exposure correction and for current correction.")
parser.add_argument("--current-correction", action="store_true")
parser.add_argument("--target-current-value", type=float, help="Current to correct to [default 100mA]", default=100)
parser.add_argument("--dark-field-correction", type=str, help="Data of the dark field to subtract before current correction", default=None)

#ARG = parser.parse_args([])
ARG = parser.parse_args()


#To write dataframe to den
def writeDenFile(inputTifFiles, denFile, force=False, darkFrame = None, targetCurrentValue = None, scanData = None):
	correctionValuePrinted = False
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
			raise IOError("File %s shape (%d, %d) does not agree with expected (%d, %d) of %s"%(os.path.basename(f), img.shape[0], img.shape[0], dimx, dimy, inputTifFiles[0]))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s of file %s"%(os.paht.basename(f), img.dtype, dtype, os.path.basename(inputTifFiles[0])))
		if darkFrame is not None:
			img = img - darkFrame
		if targetCurrentValue is not None:
		#ARG = parser.parse_args([])
			fileName = os.path.basename(f)
			frameCurrent = scanData[scanData["image_file"]==fileName]["current"].iloc[0]
			factor = targetCurrentValue/frameCurrent
			img = img * factor
		if ARG.float32:
			#Correct so that image has value of one hunderth of the median dark field signal
			correctValue = 0.0
			if darkFrame is not None:
				correctValue = np.median(darkFrame)*0.01
			if not correctionValuePrinted:
				print("Correction value of the minimum signal is %f"%(correctValue))
				correctionValuePrinted=True
			img[img < correctValue] = correctValue
			DEN.writeFrame(denFile, i, img.astype(np.float32), True)
		else:
			img[img < 0] = 0
			frame[i] = img
	if not ARG.float32:
		DEN.storeNdarrayAsDEN(denFile, frame)

if ARG.verbose:
	print("Start of the script")
darkFrame = None
scanData = None
targetCurrentValue = None
if ARG.h5file is not None:
	scanData = PETRA.scanDataset(ARG.h5file)
if ARG.dark_field_correction is not None:
	darkFrame = DEN.getFrame(ARG.dark_field_correction, 0)
if ARG.current_correction:
	targetCurrentValue = ARG.target_current_value
	if ARG.h5file is None:
		print("You have to provice h5file to be albe to perform current correction")
		sys.exit(-1)
	print("Will perform current correction to targetValue %f"%(targetCurrentValue))
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force, darkFrame = darkFrame, targetCurrentValue = targetCurrentValue, scanData = scanData)
