#!/usr/bin/env python
"""
Create sinograms

@author: Vojtech Kulvait
@year: 2023-2024

This script processes data to get volume information from the two consequent volumes that goes next to each other
"""

import argparse
import glob
import os
import errno
import sys
import h5py
import math
from denpy import DEN
from denpy import UTILS
from denpy import PETRA
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file with projected extinctions")
parser.add_argument("--input-h5", default=None, help="H5 file with dataset information")
parser.add_argument("--input-tick", default=None, help="Tick file with dataset information, use input_h5 if None.")
parser.add_argument("--binning-factor", default=None, type=float, help="Binning not considered in pixel shifts.")
parser.add_argument("--inverted-pixshifts", action="store_true")
parser.add_argument(
    "--sample-count",
    default=10,
    type=int,
    help="Number of equaly spaced horizontal lines to select for computation. Zero fo all samples in input file.")
parser.add_argument(
    "--angle-count",
    default=720,
    type=int,
    help="Number of equaly spaced angles to select for computation. Zero for all angles in file without interpolation or ordering.")
parser.add_argument(
    "--override-magnification",
    default=None,
    type=float,
    help="Use this magnification value instead of the one in H5 file")
parser.add_argument("--store-sinograms",
                    default=None,
                    type=str,
                    help="Use to store sinograms to the DEN file.")
parser.add_argument("--center-implementation", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file",
                    default=None,
                    help="Output to log file insted of stdout")
ARG = parser.parse_args()

if not os.path.isfile(ARG.inputFile):
	raise IOError("File %s does not exist" % os.path.abspath(ARG.inputFile))

if ARG.input_tick is None and ARG.input_h5 is None:
	raise IOError("There is no tick file nor h5 file specified!")


if ARG.input_tick is not None:
	if not os.path.exists(ARG.input_tick):
		raise IOError("File %s does not exist" % os.path.abspath(ARG.input_tick))
	print("START createSinograms.py for extinctions %s and tick file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_tick)))

if ARG.input_h5 is not None:
	if not os.path.exists(ARG.input_h5):
		raise IOError("File %s does not exist" % os.path.abspath(ARG.input_h5))
	print("START createSinograms.py for extinctions %s and H5 file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_h5)))

if ARG.log_file:
	sys.stdout.flush()
	sys.stdout = open("%s_tmp" % ARG.log_file, "wt")

def shiftFrame(x, shiftSize):
	if shiftSize == 0:
		return x
	f = np.roll(x, [0, shiftSize])
	if shiftSize > 0:
		f[:, :shiftSize] = 0.0
	else:  #negative
		f[:, shiftSize:] = 0.0
	return f

#Shift by float shift size
def shiftFrameFloat(f, shiftSize):
	intShift = math.floor(shiftSize)
	floatShift = shiftSize - intShift
	intShift = int(intShift)
	f1 = shiftFrame(f, intShift)
	f2 = shiftFrame(f, intShift + 1)
	return f1 * (1.0 - floatShift) + f2 * floatShift

def reduceFrame(f, intShift, xdim_reduced):
	offset  = f.shape[1]-xdim_reduced-intShift
	return f[:,offset:(offset+xdim_reduced)]

#Shift by float shift size
def shiftAndReduceFrameFloat(f, shiftSize, xdim_reduced):
	shiftSizeInt = int(shiftSize + 0.5)
	if abs(shiftSizeInt - shiftSize) < 0.01:
		#Just integer shift
		return reduceFrame(f, shiftSizeInt, xdim_reduced)
	else:
		intShift = math.floor(shiftSize)
		floatShift = shiftSize - intShift
		intShift = int(intShift)
		f1 = reduceFrame(f, intShift, xdim_reduced)
		f2 = reduceFrame(f, intShift + 1, xdim_reduced)
		return (1.0 - floatShift) * f1 + floatShift*f2

def getFrame(inputFile, k, df):
	f = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[k])
	shift = df["pixel_shift"].iloc[k]
#	if k%25 == 0:
#		print("Shifting angle k=%d frame by n=%f pixels"%(k, shift))
	f = shiftFrameFloat(f, df["pixel_shift"].iloc[k])
	return f

def getFrameNew(inputFile, k, df, xdim_reduced):
	f = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[k])
	shift = df["pixel_shift"].iloc[k]
#	if k%25 == 0:
#		print("Shifting angle k=%d frame by n=%f pixels"%(k, shift))
	f = shiftAndReduceFrameFloat(f, df["pixel_shift"].iloc[k], xdim_reduced)
	return f

def getInterpolatedFrameNew(inputFile, theta, df, xdim_reduced):
	exactMatches = np.sum(df["s_rot"] == theta)
	if exactMatches > 0:
		df = df.loc[df["s_rot"] == theta]
		f = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[0])
		f = shiftAndReduceFrameFloat(f, df["pixel_shift"].iloc[0], xdim_reduced)
		if exactMatches > 1:
			for k in range(1, exactMatches):
				g = DEN.getFrame(ARG.inputFile, df["frame_ind"][k])
				g = shiftAndReduceFrameFloat(g, df["pixel_shift"][k], xdim_reduced)
				f = f + g
			f = f / exactMatches
	else:  #I have to interpolate
		df = df.sort_values("s_rot")
		closestHigherInd = np.searchsorted(df["s_rot"], theta, side="right")
		if closestHigherInd == 0:
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd] - 360.000001
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
		elif closestHigherInd == len(df):
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]
			closestHigherInd = 0
			closestHigherAngle = df["s_rot"].iloc[0] + 360.000001
		else:
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
			closestLowerInd = closestHigherInd - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]
		thetaDiff = closestHigherAngle - closestLowerAngle
		lo = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestLowerInd])
		lo = shiftAndReduceFrameFloat(lo, df["pixel_shift"].iloc[closestLowerInd], xdim_reduced)
		hi = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestHigherInd])
		hi = shiftAndReduceFrameFloat(hi, df["pixel_shift"].iloc[closestHigherInd], xdim_reduced)
		lofac = (closestHigherAngle - theta) / thetaDiff
		if thetaDiff > 1.0:
			raise Exception(
			    "Maximal uncoverred rotation theta gap thetadiff=%f > 1.0 can not use this method"
			    % thetaDiff)
			sys.exit()
#		if ARG.verbose:
#			print(
#			    "closestHigherAngle=%f thetaDiff=%f closestLowerAngle=%f lofac=%f theta=%f"
#			    % (closestHigherAngle, thetaDiff, closestLowerAngle, lofac,
#			       theta))
		f = lofac * lo + (1.0 - lofac) * hi
	return f


#There is likely that the dataset does not contain given angle. In that case use interpolation
#Otherwise use matching frames
#Return frame with applied shift
def getInterpolatedFrame(inputFile, df, angle):
	exactMatches = np.sum(df["s_rot"] == angle)
	if exactMatches > 0:
		df = df.loc[df["s_rot"] == angle]
		f = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[0])
		f = shiftFrameFloat(f, df["pixel_shift"].iloc[0])
		if ARG.verbose:
			print("Just shifted frame f by df[pixel_shift].iloc[0]=%f" %
			      (df["pixel_shift"].iloc[0]))
		if exactMatches > 1:
			for k in range(1, exactMatches):
				g = DEN.getFrame(ARG.inputFile, df["frame_ind"][k])
				g = shiftFrameFloat(g, df["pixel_shift"][k])
				f = f + g
			f = f / exactMatches
	else:  #I have to interpolate
		df = df.sort_values("s_rot")
		closestHigherInd = np.searchsorted(df["s_rot"], angle, side="right")
		if closestHigherInd == 0:
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd] - 360.000001
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
		elif closestHigherInd == len(df):
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]
			closestHigherInd = 0
			closestHigherAngle = df["s_rot"].iloc[0] + 360.000001
		else:
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
			closestLowerInd = closestHigherInd - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]
		angleDiff = closestHigherAngle - closestLowerAngle
		lo = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestLowerInd])
		lo = shiftFrameFloat(lo, df["pixel_shift"].iloc[closestLowerInd])
		hi = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestHigherInd])
		hi = shiftFrameFloat(hi, df["pixel_shift"].iloc[closestHigherInd])
		lofac = (closestHigherAngle - angle) / angleDiff
		if angleDiff > 1.0:
			raise Exception(
			    "Maximal uncoverred rotation angle gap anglediff=%f > 1.0 can not use this method"
			    % angleDiff)
			sys.exit()
#		if ARG.verbose:
#			print(
#			    "closestHigherAngle=%f angleDiff=%f closestLowerAngle=%f lofac=%f angle=%f"
#			    % (closestHigherAngle, angleDiff, closestLowerAngle, lofac,
#			       angle))
		f = lofac * lo + (1.0 - lofac) * hi
	return f



header = DEN.readHeader(ARG.inputFile)
if len(header["dimspec"]) != 3:
	raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!" %
	                (arg.inputFile, len(header["dimspec"])))
dimspec = header["dimspec"]
xdim = np.uint32(dimspec[0])
ydim = np.uint32(dimspec[1])
zdim = np.uint32(dimspec[2])

#Equaly spaced Y indices
if ARG.sample_count > ydim or ARG.sample_count <= 0:
	ARG.sample_count = ydim
	rowIndices = np.arange(ydim)
else:
	rowIndices = np.array(np.round(np.linspace(0, ydim, num=ARG.sample_count + 2, endpoint=True))[1:-1],
                     dtype=np.int32)
print("Will produce sinogram with centerImplementation=%s rowIndices=%s"%(ARG.center_implementation, rowIndices))

if ARG.input_tick is not None:
	dat = DEN.getNumpyArray(ARG.input_tick)
	data = {"frame_ind":np.arange(zdim), 
			"s_rot":dat[0],
			"pixel_shift":dat[1]}
	df = pd.DataFrame(data)
else:
	df = PETRA.imageDataset(ARG.input_h5,
                        includePixelShift=True,
                        overrideMagnification=ARG.override_magnification)
#Angles to use in calculation
angleSequence = None

pixShifts = df["pixel_shift"]
if ARG.binning_factor is not None:
	pixShifts = pixShifts/ARG.binning_factor
if ARG.inverted_pixshifts:
	pixShifts = -pixShifts
df["pixel_shift"] = pixShifts
#Now I estimate low and high limits of the shifts and subtract maximum from each side
#Maximum is there in order to keep center at the same position
if ARG.center_implementation:
	maxshift = pixShifts.max()
	minshift = pixShifts.min()
	ldrop = math.ceil(maxshift)
	rdrop = math.floor(minshift)
	drop = np.max([ldrop, -rdrop])
	if ARG.angle_count > 1:
		angleSequence = np.linspace(0, 360, num=ARG.angle_count, endpoint=False)
		sinograms = np.zeros([ARG.sample_count, ARG.angle_count, xdim],
	                     dtype=np.float32)
		for angindex in range(len(angleSequence)):
			ang = angleSequence[angindex]
			frame = getInterpolatedFrame(ARG.inputFile, df, ang)
			for j in range(len(rowIndices)):
				sinograms[j, angindex] = frame[rowIndices[j]]
	else:
		sinograms = np.zeros([ARG.sample_count, zdim, xdim], dtype=np.float32)
		for k in range(zdim):
			frame = getFrame(ARG.inputFile, k, df)
			for j in range(len(rowIndices)):
				sinograms[j, k] = frame[rowIndices[j]]
	#Already applied on stored sinograms
	if drop != 0:
		sinograms = sinograms[:, :, drop:-drop]
	#When we offset pixShifts so that minshift = 0, we then shift array by maxintshift.
	#In case of integer maxintshift that is of the size maxshift-minshift
	#The center of the sequence will be the same as the center of the original sequence with mid shift
	if ARG.verbose:
		print("maxshift=%f, minshift=%f, ldrop=%d, rdrop=%d, drop=%d sinograms.shape=%s" %
		      (maxshift, minshift, ldrop, rdrop, drop, sinograms.shape))
else:
	#New implementation
	pixShifts = pixShifts - pixShifts.min()
	df["pixel_shift"] = pixShifts
	maxshift = pixShifts.max()
	maxintshift = int(maxshift + 0.99)
	additional_center_offset = 0.0 
	xdim_reduced = xdim - maxintshift
	if maxintshift >= xdim:
		raise ValueError("maxintshift >= xdim %d >=%d"%(maxintshift, xdim))
	if abs(maxshift-maxintshift) > 0.01 : 
		additional_center_offset = 0.5*(maxintshift - maxshift)
	if ARG.verbose:
		print("maxshift=%f, maxintshift=%d, additionalCenterOffset=%f" %
          (maxshift, maxintshift, additional_center_offset))
	if ARG.angle_count > 1:
		angleSequence = np.linspace(0, 360, num=ARG.angle_count, endpoint=False)
		sinograms = np.zeros([ARG.sample_count, ARG.angle_count, xdim_reduced],
	                     dtype=np.float32)
		for k in range(len(angleSequence)):
			theta = angleSequence[k]
			frame = getInterpolatedFrameNew(ARG.inputFile, theta, df, xdim_reduced)
			for j in range(len(ySequence)):
				sinograms[j, k] = frame[ySequence[j]]
	else:
		sinograms = np.zeros([ARG.sample_count, zdim, xdim_reduced], dtype=np.float32)
		for k in range(zdim):
			frame = getFrameNew(ARG.inputFile, k, df, xdim_reduced)
			for j in range(len(rowIndices)):
				sinograms[j, k] = frame[rowIndices[j]]

if ARG.store_sinograms is not None:
	DEN.storeNdarrayAsDEN(ARG.store_sinograms, sinograms, force=ARG.force)

if not ARG.center_implementation:
	print("additional_center_offset=%f" % (additional_center_offset))

if ARG.log_file:
	sys.stdout.close()
	sys.stdout = sys.__stdout__
	os.rename("%s_tmp" % ARG.log_file, ARG.log_file)
if ARG.input_tick is not None:
	print("END createSinograms.py for extinctions %s and tick file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_tick)))
if ARG.input_h5 is not None:
	print("END createSinograms.py for extinctions %s and H5 file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_h5)))
