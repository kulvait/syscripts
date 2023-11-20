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
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from bisect import bisect
from bisect import bisect_left


parser = argparse.ArgumentParser()
parser.add_argument('h5file', type=str, help="H5 file to read data for exposure correction and for current correction.")
parser.add_argument("outputInfoDen")
parser.add_argument("--type", default="IMG", const="IMG", nargs="?", choices=["IMG", "REF", "DAR"], help="Process just one category of IMG ... actual images, REF ... flat fields, DAR ... dark fields.")
parser.add_argument("--input-den", default=None, type=str, help="Use this DEN file as the source of data. That is for evaluation of particular processing steps efficacy.")
parser.add_argument("--raw-dir", default=None, type=str, help="Provide raw directory where to find files, by default parrent directory of inputh5.")
parser.add_argument("--target-current-value", type=float, help="Current to correct to [default 100mA]", default=100)
parser.add_argument("--lower-quantile", type=float, help="Computes also mean and median of lower quantile of the data [default 0.9]", default=0.9)
parser.add_argument("--read-info", action="store_true", help="Read info instead of storing it")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--savefig", type=str, help="Save figure as...", default=None)

#ARG = parser.parse_args([])
ARG = parser.parse_args()

#If input-den exists use it and use data read from relevant TIFF instead
def getFrame(ARG, df, inputDir, index):
	if inputDir is None:
		return DEN.getFrame(ARG.input_den, index, dtype=np.float32)
	else:
		fileStr = df["image_file"].iloc[index]
		if isinstance(fileStr, bytes):
			fileStr = fileStr.decode("utf-8")
		tiffFile = os.path.join(inputDir, fileStr.lstrip("/"))
		if not os.path.exists(tiffFile):
			raise IOError("File %s does not exist, inputDir=%s."%(tiffFile, inputDir))
		return np.array(Image.open(tiffFile), dtype=np.float32)



#To write dataframe to den
def createInfoObject(ARG):
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent = True)
	if ARG.type == "REF":
		image_key = 1
	elif ARG.type == "DAR":
		image_key = 2
	else: # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"]==image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	imageCount = len(df)
	inputDir = None
	dimx=0
	dimy=0
	if ARG.input_den is not None:
		if not os.path.exists(ARG.input_den):
			raise IOError("File %s does not exist." % (ARG.input_den))
		else:
			inf = DEN.readHeader(ARG.input_den)
			if len(inf["dimspec"]) != 3:
				raise ValueError("File %s has %d dimensions, shall be 3"%(ARG.input_den, len(inf["dimspec"])))
			if inf["dimspec"][2] != imageCount:
				raise ValueError("File %s contains %d images but the df contains %d descriptions"%(ARG.input_den, inf["dimspec"][2], imageCount))
			dimx = inf["dimspec"][0]
			dimy = inf["dimspec"][1]
			img = getFrame(ARG, df, inputDir, 0)
	else:
		if ARG.raw_dir is not None:
			inputDir = ARG.raw_dir
		else:
			inputDir = os.path.dirname(os.path.realpath(ARG.h5file))
		img = getFrame(ARG, df, inputDir, 0)
		dimy = img.shape[0]
		dimx = img.shape[1]
	zerotime = df["time"].iloc[0]
	info = np.zeros([10,imageCount], dtype=np.float64)
		# info[0,:] time in the format usual in synchrotron description in ms
		# info[1,:] angle in degrees
		# info[2,:] current in mA
		# info[3,:] mean of frame
		# info[4,:] median of frame
		# info[5,:] mean of frame after current correction
		# info[6,:] median of frame after current correction
		# info[7,:] mean of ARG.lower_quantile of the info indexed frame
		# info[8,:] median of ARG.lower_quantile of the frame
		# info[9,:] ARG.lower_quantile
	mean0 = 0.0
	for i in range(imageCount):
		start = timer()
		if i != 0:
			img = getFrame(ARG, df, inputDir, i)
		if img.shape[0] != dimy or img.shape[1] != dimx:
			raise IOError("Frame %d shape (%d, %d) does not agree with expected (%d, %d)"%(i, img.shape[0], img.shape[0], dimx, dimy))
		#Time, angle, beam current, raw mean, raw median, corrected mean, corrected median
		img = img.flatten()
		frameCurrent = np.float32(df["current"].iloc[i])
		mean = np.mean(img)
		median = np.median(img)
		info[0, i] = np.float64((df["time"].iloc[i]-zerotime).to_pytimedelta().total_seconds() )
		info[1, i] = np.float64(df["s_rot"].iloc[i])
		info[2, i] = frameCurrent
		info[3, i] = mean
		info[4, i] = median
		factor = ARG.target_current_value/frameCurrent
		info[5, i] = mean * factor
		info[6, i] = median * factor
		k = int(ARG.lower_quantile * len(img))
		indices = np.argpartition(img, k)
		flat = img[indices[:k]]
		info[7, i] = np.mean(flat)
		info[8, i] = np.median(flat)
		info[9, i] = ARG.lower_quantile
		if ARG.verbose:
			if i % 100 == 1:
				print("Processed %d/%d frame in %0.3fs mean=%f"%(i, imageCount, timer()-start, mean))
	DEN.storeNdarrayAsDEN(ARG.outputInfoDen, info, force=ARG.force)
	return info

if ARG.verbose:
	print("Start tiffImagesInfo.py h5=%s"%(ARG.h5file))

if ARG.raw_dir is not None:
	inputDir = ARG.raw_dir
else:
	inputDir = os.path.dirname(os.path.realpath(ARG.h5file))
scanData = PETRA.scanDataset(ARG.h5file, includeCurrent = True)
if ARG.type == "REF":
	image_key = 1
elif ARG.type == "DAR":
	image_key = 2
else: # IMG
	image_key = 0
df = scanData.loc[scanData["image_key"]==image_key]
df = df.assign(frame_ind=np.arange(len(df)))
zerotime = df["time"].iloc[0]
endtime = df["time"].iloc[-1]
totaltime = (endtime-zerotime).to_pytimedelta().total_seconds()
beamCurrentData = PETRA.beamCurrentDataset(ARG.h5file)
startind = bisect(list(beamCurrentData["time"]), zerotime)
endind = bisect_left(list(beamCurrentData["time"]), endtime)
startind = max(0, startind-1)
endind = min(endind+1, len(beamCurrentData))
beamCurrentData = beamCurrentData.iloc[startind:endind]

if ARG.verbose:
	print("Processing scan %s with %d frames of the length %fs"%(os.path.basename(inputDir), len(df), totaltime))

if ARG.read_info and os.path.exists(ARG.outputInfoDen):
	info = DEN.getNumpyArray(ARG.outputInfoDen)
else:
	info = createInfoObject(ARG)

if ARG.savefig is not None:
	matplotlib.use('Agg')

figure, axis = plt.subplots(3, 4)
#Plot current vs. time, mean intensity vs. time
#current vs. mean intensity, median intensity vs. mean intensity
axis[0, 0].plot(info[0,:], info[2,:])
axis[0, 0].set_title("Current vs time")

beamCurrentTimes = [(x-zerotime).to_pytimedelta().total_seconds() for x in beamCurrentData["time"]]
axis[0, 1].set_xlabel("Time [s]")
axis[0, 1].set_ylabel("Beam current [mA]")
axis[0, 1].scatter(beamCurrentTimes, beamCurrentData["current"], color="red")
axis[0, 1].set_xlim([0.0, totaltime])
axis[0, 1].set_title("Beam current measurements vs time")
ax2 = axis[0, 1].twinx()
ax2.set_ylabel("Uncorrected mean intensity")
ax2.plot(info[0,:], info[3,:])

axis[0, 2].plot(info[0,:], info[1,:])
axis[0, 2].set_title("Angle vs time")
axis[0, 3].scatter(info[6,:], info[5,:])
axis[0, 3].set_title("Corrected mean vs corrected median")

axis[1, 0].plot(info[0,:], info[3,:])
axis[1, 0].set_title("Uncorrected mean vs time")
axis[1, 1].plot(info[0,:], info[4,:])
axis[1, 1].set_title("Uncorrected median vs time")
axis[1, 2].plot(info[0,:], info[5,:])
axis[1, 2].set_title("Corrected mean vs time")
axis[1, 3].plot(info[0,:], info[6,:])
axis[1, 3].set_title("Corrected median vs time")

axis[2, 0].plot(info[0,:], info[7,:])
axis[2, 0].set_title("Uncorrected mean of bottom 90 % vs time")
axis[2, 1].plot(info[0,:], info[8,:])
axis[2, 1].set_title("Uncorrected median of bottom 90% vs time")
axis[2, 2].scatter(info[2,:], info[5,:])
axis[2, 2].set_title("Corrected mean vs current")
axis[2, 3].scatter(info[2,:], info[6,:])
axis[2, 3].set_title("Corrected median vs current")
if ARG.savefig is not None:
	fig = plt.gcf()
	fig.set_size_inches(3*11.75, 3*8.25)
	fig.savefig(ARG.savefig, dpi=50)
else:
	plt.show()

if ARG.verbose:
	print("End tiffImagesInfo.py h5=%s"%(ARG.h5file))
