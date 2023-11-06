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
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--float32", help="Output float32 array regardless dtype of input data.",  action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--h5file", type=str, help="H5 file to read data for exposure correction and for current correction.")
parser.add_argument("--current-correction", action="store_true")
parser.add_argument("--target-current-value", type=float, help="Current to correct to [default 100mA]", default=100)
parser.add_argument("--dark-field-correction", type=str, help="Data of the dark field to subtract before current correction", default=None)
parser.add_argument("--dark-frame-inf", type=str, help="Data of the lowest admissible value after dark field correction, data less or equal to max(dark-frame-inf, 1.67 MAD_frame) will be set to  max(dark-frame-inf, 1.67 MAD_frame).", default=None)
parser.add_argument("--export-info", action="store_true")
parser.add_argument("--mean-correction", action="store_true", help="Create resulting mean images to be the same")
parser.add_argument("--median-correction", action="store_true")
parser.add_argument("--gamma", type=float, help="Decode using given gamma value, try 2.2", default=None)

#ARG = parser.parse_args([])
ARG = parser.parse_args()


#To write dataframe to den
def writeDenFile(inputTifFiles, denFile, force = False, exportInfo = False, meanCorrection = False, medianCorrection=False, darkFrame = None, darkFrameInf = None, targetCurrentValue = None, scanData = None, gamma=None, minimumAdmissibleValue=None):
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
	if exportInfo:
		info = np.zeros([3,len(inputTifFiles)], dtype=np.float64)
		# info[0,:] time in the format usual in synchrotron description in ms
		# info[1,:] angle in degrees
		# info[2,:] beam current in mA in the time of acquisition
	#for i in range(len(inputTifFiles)):
	#minimumAdmissibleValue = 0.0
	#Correct by 1.67 MAD
	if darkFrame is not None:
		MAD_frame = np.median(np.absolute(darkFrame - np.median(darkFrame)))
		minimumAdmissibleValue = np.float32(1.67 * MAD_frame)
		if darkFrameInf is not None:
			darkFrameInf[darkFrameInf < minimumAdmissibleValue] = minimumAdmissibleValue
	if ARG.verbose:
		if minimumAdmissibleValue is not None:
			print("Correction value of the minimum signal is %f"%(minimumAdmissibleValue))
	mean0 = 0.0
	for i in range(len(inputTifFiles)):
		start = timer()
		f = inputTifFiles[i]
		if i != 0:
			img = np.array(Image.open(f))
		fileName = os.path.basename(f)
		if img.shape[0] != dimy or img.shape[1] != dimx:
			raise IOError("File %s shape (%d, %d) does not agree with expected (%d, %d) of %s"%(os.path.basename(f), img.shape[0], img.shape[0], dimx, dimy, inputTifFiles[0]))
		if img.dtype != dtype:
			raise IOError("File %s dtype %s does not agree with expected %s of file %s"%(os.path.basename(f), img.dtype, dtype, os.path.basename(inputTifFiles[0])))
		if ARG.float32:
			img = np.float32(img)
		if exportInfo:
			info[0, i] = np.float64(scanData[scanData["image_file"]==fileName].index[0])
			info[1, i] = np.float64(scanData[scanData["image_file"]==fileName]["s_rot"].iloc[0])
			info[2, i] = np.float64(scanData[scanData["image_file"]==fileName]["current"].iloc[0])
		if gamma is not None:
			img = np.power(img, gamma)
		if darkFrame is not None:
			img = img - darkFrame
		if targetCurrentValue is not None:
		#ARG = parser.parse_args([])
			frameCurrent = scanData[scanData["image_file"]==fileName]["current"].iloc[0]
			factor = targetCurrentValue/frameCurrent
			img = img * factor
		if ARG.float32:
			if darkFrameInf is not None:
				img = np.maximum(darkFrameInf, img)
			else:
				if minimumAdmissibleValue is not None:
					img[img < minimumAdmissibleValue] = minimumAdmissibleValue
		else:
			img[img < 0] = 0
			frame[i] = img
		mean = np.mean(img)
		median = np.median(img)
		if meanCorrection:
			if i == 0:
				mean0 = mean
			else:
				img = img * (mean0/mean)
				mean = mean0
				median = median *  (mean0/mean)
		if medianCorrection:
			if i == 0:
				median0 = median
			else:
				img = img * (median0/median)
				mean = mean * (median0/median)
				median = median0
		if ARG.float32:
			DEN.writeFrame(denFile, i, img.astype(np.float32), force=True)
		if ARG.verbose:
			print("Processed %d frame in %0.3fs mean=%f"%(i, timer()-start, mean))
	if not ARG.float32:
		DEN.storeNdarrayAsDEN(denFile, frame, force=force)
	if exportInfo:
		DEN.storeNdarrayAsDEN("%s.info"%(denFile), info, force=force)

if ARG.verbose:
	print("Start of the script")
darkFrame = None
scanData = None
targetCurrentValue = None
darkFrameInf = None
gamma=None
if ARG.gamma is not None:
	gamma = ARG.gamma
if ARG.h5file is not None:
	scanData = PETRA.scanDataset(ARG.h5file)
if ARG.dark_field_correction is not None:
	darkFrame = DEN.getFrame(ARG.dark_field_correction, 0)
if ARG.dark_frame_inf is not None:
	print("Using %s as inf value for correction"%(os.path.basename(ARG.dark_frame_inf)))
	darkFrameInf = DEN.getFrame(ARG.dark_frame_inf, 0)
if ARG.current_correction:
	targetCurrentValue = ARG.target_current_value
	if ARG.h5file is None:
		print("You have to provice h5file to be albe to perform current correction")
		sys.exit(-1)
	print("Will perform current correction to targetValue %f"%(targetCurrentValue))
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force, exportInfo = ARG.export_info, meanCorrection = ARG.mean_correction, medianCorrection=ARG.median_correction, darkFrame = darkFrame, targetCurrentValue = targetCurrentValue, scanData = scanData, darkFrameInf = darkFrameInf, gamma=gamma)
