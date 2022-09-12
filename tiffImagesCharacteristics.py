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
import matplotlib.pyplot as plt
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
parser.add_argument("--mean-correction", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()


#To write dataframe to den
def writeDenFile(inputTifFiles, denFile, force = False, exportInfo = False, meanCorrection = False, darkFrame = None, darkFrameInf = None, targetCurrentValue = None, scanData = None):
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
	info = np.zeros([7,len(inputTifFiles)], dtype=np.float64)
		# info[0,:] time in the format usual in synchrotron description in ms
		# info[1,:] angle in degrees
	#for i in range(len(inputTifFiles)):
	minimumAdmissibleValue = 0.0
	if darkFrame is not None:
		MAD_frame = np.median(np.absolute(darkFrame - np.median(darkFrame)))
		minimumAdmissibleValue = np.float32(1.67 * MAD_frame)
		print("Correction value of the minimum signal is %f"%(minimumAdmissibleValue))
		if darkFrameInf is not None:
			darkFrameInf[darkFrameInf < minimumAdmissibleValue] = minimumAdmissibleValue
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
		#Time, angle, beam current, raw mean, raw median, corrected mean, corrected median
		if exportInfo:
			info[0, i] = np.float64(scanData[scanData["image_file"]==fileName].index[0])
			info[1, i] = np.float64(scanData[scanData["image_file"]==fileName]["s_rot"].iloc[0])
			info[2, i] = np.float64(scanData[scanData["image_file"]==fileName]["current"].iloc[0])
			info[3, i] = np.mean(img)
			info[4, i] = np.median(img)
		if darkFrame is not None:
			img = img - darkFrame
		if targetCurrentValue is not None:
		#ARG = parser.parse_args([])
			frameCurrent = scanData[scanData["image_file"]==fileName]["current"].iloc[0]
			factor = targetCurrentValue/frameCurrent
			img = img * factor
		if ARG.float32:
			#Correct by 1.67 MAD
			if darkFrameInf is not None:
				img = np.maximum(darkFrameInf, img)
			else:
				img[img < minimumAdmissibleValue] = minimumAdmissibleValue
		else:
			img[img < 0] = 0
		if meanCorrection:
			if i == 0:
				mean0 = np.mean(img)
			else:
				mean = np.mean(img)
				img = img * (mean0/mean)
		info[5, i] = np.mean(img)
		info[6, i] = np.median(img, overwrite_input=True)
		print("Processed %d frame in %0.3fs"%(i, timer()-start))
	if exportInfo:
		DEN.storeNdarrayAsDEN("%s.info"%(denFile), info, force=force)
	print("INFO file exported")
	figure, axis = plt.subplots(3, 4)
	#Plot current vs. time, mean intensity vs. time
	#current vs. mean intensity, median intensity vs. mean intensity
	axis[0, 0].plot(info[0,:], info[2,:])
	axis[0, 0].set_title("Current vs time")
	axis[0, 1].plot(info[0,:], info[1,:])
	axis[0, 1].set_title("Angle vs time")
	axis[0, 2].plot(info[0,:], info[4,:])
	axis[0, 2].set_title("Uncorrected median vs time")
	axis[0, 3].plot(info[0,:], info[6,:])
	axis[0, 3].set_title("Corrected median vs time")
	axis[1, 0].plot(info[0,:], info[3,:])
	axis[1, 0].set_title("Uncorrected mean vs time")
	axis[1, 1].plot(info[0,:], info[5,:])
	axis[1, 1].set_title("Corrected mean vs time")
	axis[1, 2].plot(info[2,:], info[4,:])
	axis[1, 2].set_title("Uncorrected median vs current")
	axis[1, 3].plot(info[2,:], info[6,:])
	axis[1, 3].set_title("Corrected median vs current")
	axis[2, 0].plot(info[2,:], info[3,:])
	axis[2, 0].set_title("Uncorrected mean vs current")
	axis[2, 1].plot(info[2,:], info[5,:])
	axis[2, 1].set_title("Corrected mean vs current")
	axis[2, 2].plot(info[6,:], info[5,:])
	axis[2, 2].set_title("Corrected mean vs corrected median")
	axis[2, 3].scatter(info[4,:], info[3,:])
	axis[2, 3].set_title("Uncorrected mean vs uncorrected median")
	plt.show()

if ARG.verbose:
	print("Start of the script")
darkFrame = None
scanData = None
targetCurrentValue = None
darkFrameInf = None
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
writeDenFile(ARG.inputTifFiles, ARG.outputDen, ARG.force, exportInfo = ARG.export_info, meanCorrection = ARG.mean_correction, darkFrame = darkFrame, targetCurrentValue = targetCurrentValue, scanData = scanData, darkFrameInf = darkFrameInf)
