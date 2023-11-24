#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2023
Current correction is optimal after dark field correction, thus separate script
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
parser.add_argument(
    'h5file',
    type=str,
    help=
    "H5 file to read data for exposure correction and for current correction.")
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument(
    "--type",
    default="IMG",
    const="IMG",
    nargs="?",
    choices=["IMG", "REF"],
    help=
    "Process just one category of IMG ... actual images, REF ... flat fields.")
parser.add_argument("--target-current-value",
                    type=float,
                    help="Current to correct to [default 100mA]",
                    default=100)
parser.add_argument("--saveinfo",
                    type=str,
                    help="Create info object from those data and save it as...",
                    default=None)
parser.add_argument("--savefig",
                    type=str,
                    help="Create figure from those data and save it as...",
                    default=None)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--force", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()


#To write dataframe to den
def currentCorrect(ARG):
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True)
	if ARG.type == "REF":
		image_key = 1
	else:  # IMG
		image_key = 0
	denFile = ARG.outputDen
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	imageCount = len(df)
	inputDir = None
	dimx = 0
	dimy = 0
	if not os.path.exists(ARG.inputDen):
		raise IOError("File %s does not exist." % (ARG.inputDen))
	else:
		inf = DEN.readHeader(ARG.inputDen)
		if len(inf["dimspec"]) != 3:
			raise ValueError("File %s has %d dimensions, shall be 3" %
			                 (ARG.inputDen, len(inf["dimspec"])))
		if inf["dimspec"][2] != imageCount:
			raise ValueError(
			    "File %s contains %d images but the df contains %d descriptions"
			    % (ARG.inputDen, inf["dimspec"][2], imageCount))
		dimx = inf["dimspec"][0]
		dimy = inf["dimspec"][1]
	DEN.writeEmptyDEN(denFile, [dimx, dimy, imageCount],
	                  elementtype=np.dtype("<f4"),
	                  force=True)
	zerotime = df["time"].iloc[0]
	info = np.zeros([10, imageCount], dtype=np.float64)
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
		img = DEN.getFrame(ARG.inputDen, i)
		frameCurrent = np.float32(df["current"].iloc[i])
		factor = ARG.target_current_value / frameCurrent
		DEN.writeFrame(denFile,
		               i,
		               np.array(factor * img, dtype=np.float32),
		               force=True)
		if (ARG.savefig is not None) or (ARG.saveinfo is not None):
			img = img.flatten()
			mean = np.mean(img)
			median = np.median(img)
			info[0, i] = np.float64((df["time"].iloc[i] -
			                         zerotime).to_pytimedelta().total_seconds())
			info[1, i] = np.float64(df["s_rot"].iloc[i])
			info[2, i] = frameCurrent
			info[3, i] = mean
			info[4, i] = median
			info[5, i] = mean * factor
			info[6, i] = median * factor
			k = int(0.9 * len(img))
			indices = np.argpartition(img, k)
			flat = img[indices[:k]]
			info[7, i] = np.mean(flat)
			info[8, i] = np.median(flat)
			info[9, i] = 0.9
		if ARG.verbose:
			if i % 100 == 1:
				print("Processed %d/%d frame in %0.3fs mean=%f" %
				      (i, imageCount, timer() - start, mean))
	if ARG.saveinfo is not None:
		DEN.storeNdarrayAsDEN(ARG.saveinfo, info, force=ARG.force)
	return info


if ARG.verbose:
	print("Start currentCorrect.py h5=%s" % (ARG.h5file))

info = currentCorrect(ARG)

if ARG.savefig is not None:
	matplotlib.use('Agg')
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True)
	if ARG.type == "REF":
		image_key = 1
	else:  # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	zerotime = df["time"].iloc[0]
	endtime = df["time"].iloc[-1]
	totaltime = (endtime - zerotime).to_pytimedelta().total_seconds()
	beamCurrentData = PETRA.beamCurrentDataset(ARG.h5file)
	startind = bisect(list(beamCurrentData["time"]), zerotime)
	endind = bisect_left(list(beamCurrentData["time"]), endtime)
	startind = max(0, startind - 1)
	endind = min(endind + 1, len(beamCurrentData))
	beamCurrentData = beamCurrentData.iloc[startind:endind]
	figure, axis = plt.subplots(3, 4)
	#Plot current vs. time, mean intensity vs. time
	#current vs. mean intensity, median intensity vs. mean intensity
	axis[0, 0].plot(info[0, :], info[2, :])
	axis[0, 0].set_title("Current vs time")
	beamCurrentTimes = [(x - zerotime).to_pytimedelta().total_seconds()
	                    for x in beamCurrentData["time"]]
	axis[0, 1].set_xlabel("Time [s]")
	axis[0, 1].set_ylabel("Beam current [mA]")
	axis[0, 1].scatter(beamCurrentTimes,
	                   beamCurrentData["current"],
	                   color="red")
	axis[0, 1].set_xlim([0.0, totaltime])
	axis[0, 1].set_title("Beam current measurements vs time")
	ax2 = axis[0, 1].twinx()
	ax2.set_ylabel("Uncorrected mean intensity")
	ax2.plot(info[0, :], info[3, :])
	axis[0, 2].plot(info[0, :], info[1, :])
	axis[0, 2].set_title("Angle vs time")
	axis[0, 3].scatter(info[6, :], info[5, :])
	axis[0, 3].set_title("Corrected mean vs corrected median")
	axis[1, 0].plot(info[0, :], info[3, :])
	axis[1, 0].set_title("Uncorrected mean vs time")
	axis[1, 1].plot(info[0, :], info[4, :])
	axis[1, 1].set_title("Uncorrected median vs time")
	axis[1, 2].plot(info[0, :], info[5, :])
	axis[1, 2].set_title("Corrected mean vs time")
	axis[1, 3].plot(info[0, :], info[6, :])
	axis[1, 3].set_title("Corrected median vs time")
	axis[2, 0].plot(info[0, :], info[7, :])
	axis[2, 0].set_title("Uncorrected mean of bottom 90 % vs time")
	axis[2, 1].plot(info[0, :], info[8, :])
	axis[2, 1].set_title("Uncorrected median of bottom 90% vs time")
	axis[2, 2].scatter(info[2, :], info[5, :])
	axis[2, 2].set_title("Corrected mean vs current")
	axis[2, 3].scatter(info[2, :], info[6, :])
	axis[2, 3].set_title("Corrected median vs current")
	fig = plt.gcf()
	fig.set_size_inches(3 * 11.75, 3 * 8.25)
	fig.savefig(ARG.savefig, dpi=50)

if ARG.verbose:
	print("End currentCorrect.py h5=%s" % (ARG.h5file))
