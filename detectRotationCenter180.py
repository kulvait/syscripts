#!/usr/bin/env python
"""
Created 2023

@author: Vojtech Kulvait

Tries to detect rotation center based on input extinction data


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
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file with projected extinctions")
parser.add_argument("--input-h5", default=None, help="H5 file with dataset information")
parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files and use input linspace instead.")
parser.add_argument("--input-tick", default=None, help="Tick file with dataset information, use input-h5 if None.")
parser.add_argument("--input-linspace", action="store_true", help="Use np.linspace(0,180, num, endpoint=False) and zero pixel shifts instead of any specs.")
parser.add_argument("--crop-percent", default=0.0, type=float, help="Crop procent of the sinogram from the sides.")
parser.add_argument("--binning-factor", default=None, type=float, help="Binning not considered in pixel shifts.")
parser.add_argument("--inverted-pixshifts", action="store_true")
parser.add_argument(
	"--sample-count",
	default=10,
	type=int,
	help="Number of equaly spaced horizontal lines to select for computation")
parser.add_argument(
	"--angle-count",
	default=720,
	type=int,
	help="Number of equaly spaced angles to select for computation")
parser.add_argument(
	"--override-magnification",
	default=None,
	type=float,
	help="Use this magnification value instead of the one in H5 file")
parser.add_argument("--store-sinograms",
					default=None,
					type=str,
					help="Use to store sinograms to the DEN file.")
parser.add_argument("--load-sinograms",
					default=None,
					type=str,
					help="Load sinograms from DEN file.")
parser.add_argument(
	"--ord",
	default=2,
	type=int,
	help=
	"Order of norm, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html."
)
parser.add_argument(
	"--sinogram-consistency",
	action="store_true",
	help=
	"Search for the offcet with maximum consistency so that f(x)-f(x+pi) is optimal."
)
parser.add_argument("--max-derivative",
					action="store_true",
					help="Search by the algorithm of extreme derivative.")
parser.add_argument("--svd",
					action="store_true",
					help="Perform SVD analysis to determine center.")
parser.add_argument("--fourier",
					action="store_true",
					help="Perform  find_center_vo from algotom based on https://doi.org/10.1364/OE.22.019078 .")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file",
					default=None,
					help="Output to log file insted of stdout")
parser.add_argument("--center-implementation", action="store_true")
ARG = parser.parse_args()

if ARG.fourier:
	from algotom.prep import calculation

# Check if input file exists and handle errors 
# Check if the path exists at all
if not os.path.exists(ARG.inputFile):
	raise IOError("Path %s does not exist" % os.path.abspath(ARG.inputFile))
# If it exists but is a directory
if os.path.isdir(ARG.inputFile):
	raise IOError("Expected a file but got a directory: %s" % os.path.abspath(ARG.inputFile))
# If it exists and is not a file
if not os.path.isfile(ARG.inputFile):
	raise IOError("Path %d does not point to a regular file" %
				  os.path.abspath(ARG.inputFile))

if ARG.input_tick is None and ARG.input_h5 is None and ARG.input_linspace is False:
	raise IOError("There is no tick file nor h5 file specified!")

if ARG.input_tick is not None:
	if not os.path.exists(ARG.input_tick):
		raise IOError("File %s does not exist" % os.path.abspath(ARG.input_tick))
	print("START detectRotationCenter180.py for extinctions %s and tick file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_tick)))

if ARG.input_h5 is not None:
	if not os.path.exists(ARG.input_h5):
		raise IOError("File %s does not exist" % os.path.abspath(ARG.input_h5))
	print("START detectRotationCenter180.py for extinctions %s and H5 file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_h5)))

if ARG.load_sinograms is not None and not os.path.isfile(ARG.load_sinograms):
	raise IOError("File %s to load sinograms does not exist" % os.path.abspath(ARG.load_sinograms))

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

def reduceFrame(f, intShift, xdim_reduced):
	offset = f.shape[1]-xdim_reduced-intShift
	return f[:,offset:(offset+xdim_reduced)]

#Shift by float shift size
def shiftFrameFloat(f, shiftSize):
	intShift = math.floor(shiftSize)
	floatShift = shiftSize - intShift
	intShift = int(intShift)
	f1 = shiftFrame(f, intShift)
	f2 = shiftFrame(f, intShift + 1)
	return f1 * (1.0 - floatShift) + f2 * floatShift

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
			print("Error: theta=%s thetaDiff=%s closestLowerAngle=%s closestHigherAngle=%s df_min=%f df_max=%f"%(theta, thetaDiff, closestLowerAngle, closestHigherAngle, df["s_rot"].min(), df["s_rot"].max()))
			print(df["s_rot"])
			raise Exception(
				"Maximal uncoverred rotation theta gap thetadiff=%f > 1.0 can not use this method"
				% thetaDiff)
			sys.exit()
#		if ARG.verbose:
#			print(
#				"closestHigherAngle=%f thetaDiff=%f closestLowerAngle=%f lofac=%f theta=%f"
#				% (closestHigherAngle, thetaDiff, closestLowerAngle, lofac,
#				   theta))
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
			for k in range(exactMatches - 1):
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
				"For angle=%f maximal uncoverred rotation angle gap anglediff=%f > 1.0 can not use this method"
				% (angle, angleDiff))
			sys.exit()
		if ARG.verbose:
			print(
				"closestHigherAngle=%f angleDiff=%f closestLowerAngle=%f lofac=%f angle=%f"
				% (closestHigherAngle, angleDiff, closestLowerAngle, lofac,
				   angle))
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
if ARG.sample_count < 1 or ARG.sample_count > ydim:
	raise ValueError("Invalid sample_count=%d" % ARG.sample_count)
ySequence = np.array(np.round(
	np.linspace(0, ydim, num=ARG.sample_count + 2, endpoint=True))[1:-1],
					 dtype=np.int32)
if ARG.sample_count == ydim:
	ySequence = np.arange(ydim)

if ARG.angle_count < 1:
	raise ValueError("Invalid angle_count=%d" % ARG.angle_count)

if ARG.input_tick is not None:
	dat = DEN.getNumpyArray(ARG.input_tick)
	data = {"frame_ind":np.arange(zdim),
			"s_rot":dat[0],
			"pixel_shift":dat[1]}
	df = pd.DataFrame(data)
elif ARG.input_h5 is not None:
	try:
		df = PETRA.imageDataset(ARG.input_h5, includePixelShift=True, overrideMagnification=ARG.override_magnification)
	except Exception as e:
		if ARG.fix_corrupted_h5:
			print("Error in reading H5 file %s, trying to fix it."%(ARG.input_h5))
			data = {"frame_ind":np.arange(zdim),
					"s_rot":np.linspace(0, 180, num=zdim, endpoint=False),
					"pixel_shift":np.zeros(zdim)}
			df = pd.DataFrame(data)
		else:
			raise e
elif ARG.input_linspace:
	data = {"frame_ind":np.arange(zdim),
			"s_rot":np.linspace(0, 180, num=zdim, endpoint=False),
			"pixel_shift":np.zeros(zdim)}
	df = pd.DataFrame(data)

pixShifts = df["pixel_shift"].copy()
rangeShifts = df["pixel_shift"].max()-df["pixel_shift"].min()
if rangeShifts < 0.1:
	for i in range(len(pixShifts)):
		pixShifts.iloc[i] = 0.0
		#df["pixel_shift"].iloc[i] = 0.0
if ARG.binning_factor is not None:
	pixShifts = pixShifts/ARG.binning_factor
if ARG.inverted_pixshifts:
	pixShifts = -pixShifts
df["pixel_shift"] = pixShifts

minAngle = df["s_rot"].min() 
df["s_rot"] = df["s_rot"]-minAngle
minAngle = df["s_rot"].min() 
maxAngle = df["s_rot"].max()
projectionCount=len(df["s_rot"])

maxTestAngle=maxAngle+1
optimalAngleCount = projectionCount + 1
while maxTestAngle > maxAngle and optimalAngleCount > 5:
	optimalAngleCount = optimalAngleCount - 1
	angleSequence = np.linspace(0, 360, num=2*optimalAngleCount, endpoint=False)
	maxTestAngle = angleSequence[optimalAngleCount-1]


#Specific procedure to get range [0, 180) st. distance to next angle after 180 allows for flipping and angle is still in the data
angleSequence = np.linspace(0, 360, num=2*ARG.angle_count, endpoint=False)
angleSequence = angleSequence[:ARG.angle_count]
maxTestAngle=angleSequence[-1]
if ARG.verbose:
	print("Estimated optimal angle_count=%d while projection_count=%d"%(optimalAngleCount, len(angleSequence)))
	print("minAngle=%f maxAngle=%f maxTestedAngle=%f"%(minAngle, maxAngle, maxTestAngle))
if maxAngle < maxTestAngle:
	print("maxAngle=%f shall not be less than %s for the method, setting optimalAngleCount=%d!"%(maxAngle, maxTestAngle, optimalAngleCount))
	if optimalAngleCount <= 5:
		raise ValueError("optimalAngleCount is 5 or less given input data!")
	angleSequence = np.linspace(0, 360, num=2*optimalAngleCount, endpoint=False)
	angleSequence = angleSequence[:optimalAngleCount]
	maxTestAngle=angleSequence[-1]

sinogram_center_offset = 0.0

if ARG.load_sinograms is not None:
	info = DEN.readHeader(ARG.load_sinograms)
	if len(info["dimspec"]) != 3:
		raise TypeError(
			"Dimension of dimspec for file %s shall be 3 but is %d!" %
			(arg.load_sinograms, len(info["dimspec"])))
	if info["dimspec"][1] != ARG.angle_count:
		if ARG.verbose:
			print("Setting angle_count to %d from ARG.angle_count %d" %
				  (info["dimspec"][1], ARG.angle_count))
		ARG.angle_count = info["dimspec"][1]
		angleSequence = np.linspace(0, 180, num=ARG.angle_count, endpoint=False)
	sinograms = DEN.getNumpyArray(ARG.load_sinograms)
	if not ARG.center_implementation:
		pixShifts = pixShifts - pixShifts.min()
		df["pixel_shift"] = pixShifts
		maxshift = pixShifts.max()
		maxintshift = int(maxshift + 0.99)
		xdim_reduced = xdim - maxintshift
		if maxintshift >= xdim:
			raise ValueError("maxintshift >= xdim %d >=%d"%(maxintshift, xdim))
		if abs(maxshift-maxintshift) > 0.01 :
			sinogram_center_offset = 0.5*(maxintshift - maxshift)
elif ARG.center_implementation:
	sinograms = np.zeros([ARG.sample_count, len(angleSequence), xdim_reduced],
					 dtype=np.float32)
	maxshift = pixShifts.max()
	minshift = pixShifts.min()
	ldrop = math.ceil(maxshift)
	rdrop = math.floor(minshift)
	drop = np.max([ldrop, -rdrop])
	if ARG.verbose:
		print("maxshift=%f, minshift=%f, ldrop=%d, rdrop=%d, drop=%d" %
			  (maxshift, minshift, ldrop, rdrop, drop))
	for angindex in range(len(angleSequence)):
		ang = angleSequence[angindex]
		frame = getInterpolatedFrame(ARG.inputFile, df, ang)
		for j in range(len(ySequence)):
			sinograms[j, angindex] = frame[ySequence[j]]
	#Already applied on stored sinograms
	if drop != 0:
		sinograms = sinograms[:, :, drop:-drop]
else:
	#New implementation
	pixShifts = pixShifts - pixShifts.min()
	df["pixel_shift"] = pixShifts
	maxshift = pixShifts.max()
	maxintshift = int(maxshift + 0.99)
	xdim_reduced = xdim - maxintshift
	sinograms = np.zeros([ARG.sample_count, len(angleSequence), xdim_reduced],
					 dtype=np.float32)
	if maxintshift >= xdim:
		raise ValueError("maxintshift >= xdim %d >=%d"%(maxintshift, xdim))
	if abs(maxshift-maxintshift) > 0.01 :
		sinogram_center_offset = 0.5*(maxintshift - maxshift)
	if ARG.verbose:
		print("maxshift=%f, maxintshift=%d, additionalCenterOffset=%f" %
			  (maxshift, maxintshift, sinogram_center_offset))
	for k in range(len(angleSequence)):
		theta = angleSequence[k]
		frame = getInterpolatedFrameNew(ARG.inputFile, theta, df, xdim_reduced)
		for j in range(len(ySequence)):
			sinograms[j, k] = frame[ySequence[j]]

if ARG.store_sinograms is not None:
	DEN.storeNdarrayAsDEN(ARG.store_sinograms, sinograms, force=ARG.force)
	if ARG.verbose:
		print("Sinograms stored %s"%(ARG.store_sinograms))

sinogramPixels = sinograms.shape[2]
print("sinogramPixels=%f"%(sinogramPixels))

def denoiseGaussian2(vec, lim=1):
	maxsigma = len(vec) / 10
	minsigma = 1.0
	cursigma = 1.0
	lim_tol = lim / 100
	gf = gaussian_filter(vec, minsigma)
	pval = scipy.stats.anderson(vec - gf)[0]
	if pval > lim:
		return pval, minsigma, gf
	for i in range(10):
		cursigma = (maxsigma + minsigma) / 2
		gf = gaussian_filter(vec, cursigma)
		pval = scipy.stats.anderson(vec - gf)[0]
		if np.abs(pval - lim) < lim_tol:
			return pval, cursigma, gf
		if pval > lim:
			maxsigma = cursigma
		else:
			minsigma = cursigma
	gf = gaussian_filter(vec, cursigma)
	pval = scipy.stats.anderson(vec - gf)[0]
	return pval, cursigma, gf


def denoiseGaussian(vec):
	maxsigma = len(vec) / 10
	minsigma = 1.0
	cursigma = 1.0
	gf = gaussian_filter(vec, minsigma)
	pval = scipy.stats.normaltest(vec - gf)[1]
	if pval < 1e-10:
		return pval, minsigma, gf
	for i in range(10):
		cursigma = (maxsigma + minsigma) / 2
		gf = gaussian_filter(vec, cursigma)
		pval = scipy.stats.normaltest(vec - gf)[1]
		if pval < 1e-10:
			maxsigma = cursigma
		else:
			minsigma = cursigma
	maxsigma = cursigma
	minsigma = 1.0
	cursigma = 1.0
	gf = gaussian_filter(vec, cursigma)
	pval = scipy.stats.skewtest(vec - gf)[1]
	if pval < 0.1:
		return pval, minsigma, gf
	for i in range(10):
		cursigma = (maxsigma + minsigma) / 2
		gf = gaussian_filter(vec, cursigma)
		pval = scipy.stats.skewtest(vec - gf)[1]
		if np.abs(pval - 0.1) < 0.001:
			return pval, cursigma, gf
		if pval < 0.1:
			maxsigma = cursigma
		else:
			minsigma = cursigma
	return pval, cursigma, gf


def monotonicityChanges(convergingSequence):
	#First check that the converging sequence does not have max and min ARGMIN
	vals = [x[1] for x in convergingSequence]
	if len(vals) < 3:
		return 0
	nondecreasing = True
	if vals[1] < vals[0]:
		nondecreasing = False
	monotonicityChanges = 0
	oldval = vals[1]
	for v in vals[2:]:
		if nondecreasing:
			if v < oldval:
				nondecreasing = False
				monotonicityChanges += 1
		else:
			if v >= oldval:
				noncecreasing = True
				monotonicityChanges += 1
		oldval = v
	return monotonicityChanges


def svdAnalyze(sinogram):
	#	U, S, Vh = np.linalg.svd(sinogram, full_matrices=False)
	U, S, Vh = scipy.sparse.linalg.svds(sinogram, k=20)
	Vh = np.flip(Vh, axis=0)
	S = np.flip(S)
	U = np.flip(U, axis=1)
	significantCount = np.sum(S / S[0] > 0.1)
	if significantCount < 5:
		print("Lot of low quality components")
		return np.nan
	offsets = np.zeros(significantCount)
	print("Spectral ratio is %0.1f" % (S[0] / S[1]))
	Vh = Vh - Vh.mean(axis=1, keepdims=True)
	vlx = Vh.shape[1]
	#	sig = round(vlx/200)
	INDMAX = 100
	i = 0
	while i < significantCount:
		#pval, sigma, Vg = denoiseGaussian(Vh[i])
		pval, sigma, Vg = denoiseGaussian2(Vh[i])
		#Vg = gaussian_filter(Vh[i], sig)
		Vgf = np.flip(Vg)
		Vg_norm = np.linalg.norm(Vg, ord=ARG.ord)
		Vg_plus = np.linalg.norm(Vg + Vgf, ord=ARG.ord)
		Vg_minus = np.linalg.norm(Vg - Vgf, ord=ARG.ord)
		MIN = 0
		MAX = vlx - 1
		if Vg_plus < Vg_minus:
			symmetry = "ANTISYMMETRIC"
			normFit = Vg_plus / Vg_norm
			antisymmetric = 1
		else:
			symmetry = "	SYMMETRIC"
			normFit = Vg_minus / Vg_norm
			antisymmetric = -1
		if ARG.max_derivative:
			if Vg_plus < Vg_minus:
				dx = Vg
			else:
				dx = savgol_filter(Vg, round(len(Vg) / 10), 5, deriv=1)
			MIN = np.min([np.argmin(dx), np.argmax(dx)]) - INDMAX
			MAX = np.max([np.argmin(dx), np.argmax(dx)]) + INDMAX
		IND = np.arange(-INDMAX, INDMAX + 1)
		AMIN = np.max([0, MIN - INDMAX])
		AMAX = np.min([vlx, MAX + 1 + INDMAX])
		centerOffset = 0.5 * AMIN + 0.5 * AMAX - vlx * 0.5
		testVec = Vg[AMIN:AMAX]
		testVecFlipped = antisymmetric * np.flip(testVec)
		J = IND[np.argmin([
			np.linalg.norm(
				(testVec + np.roll(testVecFlipped, i))[INDMAX:-INDMAX],
				ord=ARG.ord) for i in IND
		])]
		while np.abs(J) > 0.9 * INDMAX and INDMAX < 0.25 * vlx:
			INDMAX = 2 * INDMAX
			IND = np.arange(-INDMAX, INDMAX + 1)
			AMIN = np.max([0, MIN - INDMAX])
			AMAX = np.min([vlx, MAX + 1 + INDMAX])
			centerOffset = 0.5 * AMIN + 0.5 * AMAX - vlx * 0.5
			testVec = Vg[AMIN:AMAX]
			testVecFlipped = antisymmetric * np.flip(testVec)
			J = IND[np.argmin([
				np.linalg.norm(
					(testVec + np.roll(testVecFlipped, i))[INDMAX:-INDMAX],
					ord=ARG.ord) for i in IND
			])]
		print(
			"Component %d %s, offset %5.1f pix, position %5.1f pix, Vg_plusminus/Vg_norm=%0.3f lower better, centerOffset=%0.1f, J=%5d, SV=%3.2f, pval=%f, sigma=%f."
			% (i, symmetry, centerOffset + 0.5 * J,
			   vlx * 0.5 - 0.5 + centerOffset + 0.5 * J, normFit, centerOffset,
			   J, S[i], pval, sigma))
		offsets[i] = centerOffset + 0.5 * J
		i = i + 1
	return np.median(offsets)


#When search start is offsetted from the center the tails shall be balanced
#Make zeros at the positions where there is no matching tail
def maskLongerTail(sin, offset):
	sin = sin.copy()
	xdim = sin.shape[1]
	midpoint = 0.5 * xdim
	centerPoint = midpoint + offset
	center_two = round(centerPoint*2, 0)
	betweenPixels = True #If midpoint lies just between the pixels
	if center_two % 2 == 1:
		betweenPixels = False
	centerPoint = center_two / 2
	centerPixel = int(center_two // 2)
	if centerPoint == midpoint:
		return sin
	elif centerPoint < midpoint:#Right tail will be censored
		if betweenPixels: #centerpoint is centerPixel+0.5
			censorIndex = centerPixel * 2
		else:
			censorIndex = 1+centerPixel*2
		sin[:,censorIndex:] = 0
	else:#Left tail will be censored
		if betweenPixels: #centerpoint is centerPixel+0.5
			censorLength = 2*centerPixel - xdim
		else:
			censorLength = 2* centerPixel + 1 - xdim
		sin[:,:censorLength] = 0
	return sin


def estimateTrigonometricFit(sin_orig, shiftEstimate_old=0, iterations=20):
	if shiftEstimate_old == 0:
		sin = sin_orig
	else:
		sin = maskLongerTail(sin_orig, -shiftEstimate_old)
	xdim = sin.shape[1]
	ydim = sin.shape[0]
	sin_mass = sin.sum(axis=1)
	sin_mass = np.maximum(sin_mass, 1e-10)#Avoid zero division
	xcoord = 0.5 + np.arange(xdim)
	coa = sin.dot(xcoord)/sin_mass
	angles = np.linspace(0, 2*np.pi, num=2*ydim, endpoint=False)
	angles = angles[:ydim]
	A = np.zeros([ydim, 3])
	for i in np.arange(ydim):
		angle = angles[i]
		A[i, 0] = 1
		A[i, 1] = np.cos(angle)
		A[i, 2] = np.sin(angle)
	coefs, residuals, RANK, sing = np.linalg.lstsq(A, coa, rcond=None)
	if math.isnan(coefs[0]) or coefs[0] is None:
		print("Error in estimateTrigonometricFit when processing sin_orig with shiftEstimate_old=%d iterations=%s coefs=(%s, %s, %s)"%(shiftEstimate_old, iterations, coefs[0], coefs[1], coefs[2]))
		return None
	shiftEstimate=0.5*xdim-coefs[0]
	#plt.plot(angles, coa, label="Original")
	#plt.plot(angles, A.dot(coefs[0]), label="Fitted")
	#plt.show()
	#print("iterations=%d coefs=%f, %f, %f xdim*0.5=%f shiftEstimate=%f"%(iterations, coefs[0][0], coefs[0][1], coefs[0][2], 0.5*xdim, shiftEstimate))
	if iterations == 1:
		return shiftEstimate
	else:
		return estimateTrigonometricFit(sin_orig, shiftEstimate, iterations-1)


def estimateSmoothFit(sin_orig, shiftEstimate_old=0, iterations=5):
	if shiftEstimate_old == 0:
		sin = sin_orig.copy()
	else:
		sin = maskLongerTail(sin_orig, -shiftEstimate_old)
	sin = shiftFrameFloat(sin, shiftEstimate_old)
	sinogram_second = np.flip(sin, axis=1)
	xdim = sin.shape[1]
	ydim = sin.shape[0]
	sin_mass = sin.sum(axis=1)
	sin_mass = np.maximum(sin_mass, 1e-10)#Avoid zero division
	xcoord = 0.5 + np.arange(xdim)
	angleall = np.linspace(0, 2*np.pi, num=2*ydim, endpoint=False)
	angles = angleall[:ydim]
	N = np.sum(angles < np.pi*0.5)
	smoothfactors = np.arange(10, N, 1)
	xcoord = 0.5 + np.arange(xdim)
	coa_first = sin.dot(xcoord)/sin_mass
	coa_second = sinogram_second.dot(xcoord)/sin_mass
	coa = np.hstack([coa_first, coa_second])
	estimates = np.zeros([9, len(smoothfactors)])
	for i in np.arange(len(smoothfactors)):
		sf = smoothfactors[i]
		coa_firstx = savgol_filter(coa_first, sf, 3)
		coa_secondx = savgol_filter(coa_second, sf, 3)
		coa_savgol = savgol_filter(coa, sf, 3)
		midposition = 0.5*(coa_savgol[ydim-1] + coa_savgol[ydim])
		shiftestimate_left = midposition - coa_firstx[-1]
		shiftestimate_right = coa_secondx[0] - midposition
		shiftestimate = 0.5*(shiftestimate_left + shiftestimate_right)
		estimate_error = np.abs(shiftestimate_left) + np.abs(shiftestimate_right)
		estimates[0, i] = sf
		estimates[1, i] = midposition
		estimates[2, i] = shiftestimate_left
		estimates[3, i] = shiftestimate_right
		estimates[4, i] = shiftestimate
		estimates[5, i] = estimate_error
		estimates[6, i] = shiftestimate_left*shiftestimate_left + shiftestimate_right*shiftestimate_right
		estimates[7, i] = shiftestimate * shiftestimate
		estimates[8, i] = np.abs(shiftestimate_left - shiftestimate_right)
		#if i % 100 == 0:
		#	plt.plot(angleall, np.hstack([coa_firstx, coa_secondx]))
	#plt.show()
	shiftEstimateUpdate = np.median(estimates[4])
	shiftEstimate = shiftEstimate_old + shiftEstimateUpdate
	#plt.plot(estimates[0], estimates[1], label="midposition")
	#plt.title("Mid position")
	#plt.show()
	#plt.plot(estimates[0], estimates[2], label="shiftestimate_left")
	#plt.plot(estimates[0], estimates[3], label="shiftestimate_right")
	#plt.plot(estimates[0], estimates[4], label="shiftestimate")
	#plt.title("Estimates")
	#plt.legend()
	#plt.show()
	#print("iterations=%d shiftEstimate=%f shiftEstimateUpdate=%f"%(iterations, shiftEstimate, shiftEstimateUpdate))
	if iterations == 1:
		return shiftEstimate
	else:
		return estimateSmoothFit(sin_orig, shiftEstimate, iterations-1)


def calculate_entropy(sinogram):
	flattened_sinogram = sinogram.flatten()
	entropy = stats.entropy(np.histogram(flattened_sinogram, bins=256, density=True)[0])
	return entropy

offsets = np.zeros(len(ySequence))
offset_trig = np.zeros(len(ySequence))
offset_smooth = np.zeros(len(ySequence))
subpixeloffsets = np.zeros(len(ySequence))
interpoffsets = np.zeros(len(ySequence))


for j in range(len(ySequence)):
	sinogram = sinograms[j]
	#Subtracting means is not a good operation
	#sinogram = sinogram - sinogram.mean(axis=1, keepdims=True)
	MIN = 0
	width = sinogram.shape[1]
	if ARG.crop_percent is not None and ARG.crop_percent > 0:
		crop = int(width * ARG.crop_percent / 100)
		crop_side = crop // 2
		if width - 2 * crop_side < 2:
			raise ValueError("Too much crop, consider value less than 50%, width=%d crop=%d" % (width, crop))
		sinogram = sinogram[:, crop_side:width - crop_side]
		width = sinogram.shape[1]
	entropy = calculate_entropy(sinogram)
	offset_trig[j] = estimateTrigonometricFit(sinogram)
	offset_smooth[j] = estimateSmoothFit(sinogram)
	if ARG.fourier:
		COD = 0.5*width - 0.5
		cor = calculation.find_center_vo(sinogram, radius=5, step=0.1)
		offsets[j] = cor - COD
		cor = calculation.find_center_vo(sinogram, radius=5, step=0.1, start=0.2*sinogramPixels, stop=0.8*sinogramPixels)
		subpixeloffsets[j] = cor - COD
		cor = calculation.find_center_vo(sinogram, radius=5, step=0.01)
		interpoffsets[j] = cor - COD
		print("j=%d y=%d/%d offsets_default[j]=%0.2f offsets_broad[j]=%0.2f offsets_interp[j]=%0.2f offset_trig[j]=%0.2f offsets_smooth[j]=%0.2f, entropy=%0.2f" % (j, ySequence[j], ydim, offsets[j], subpixeloffsets[j], interpoffsets[j], offset_trig[j], offset_smooth[j], entropy))
	elif ARG.svd:
		print("SVD j=%d y=%d" % (j, ySequence[j]))
		offsets[j] = svdAnalyze(sinogram)
		if not np.isnan(offsets[j]):
			print("For j=%d y=%d the center of symmetry offset %.1f" %
				  (j, ySequence[j], offsets[j]))
			print("")
	elif ARG.sinogram_consistency:
		halfAngleCount = sinogram.shape[0] // 2
		sa = sinogram[0:halfAngleCount]
		sb = -np.flip(sinogram[halfAngleCount:], axis=1)
		#Scan for starting point
		searchMaxShift = width // 3
		searchInitShift = 0
		IND = np.arange(searchInitShift - searchMaxShift,
						searchInitShift + searchMaxShift + 1, 10)
		maskSize = np.max([
			0, searchMaxShift + searchInitShift,
			searchMaxShift - searchInitShift
		])
		normedValues = [
			np.linalg.norm(
				(sa + np.roll(sb, i, axis=1))[:, maskSize:width - maskSize],
				ord=1) for i in IND
		]
		ARGMIN = np.argmin(normedValues)
		searchInitShift = IND[ARGMIN]
		searchMaxShift = 100
		ARGMIN = 0
		IND = [searchInitShift]
		convergingSequence = []
		while len(convergingSequence) < 1 or (
			convergingSequence[-1][0] != searchMaxShift and
			monotonicityChanges(convergingSequence) < 3):
			searchInitShift = IND[ARGMIN]
			IND = np.arange(searchInitShift - searchMaxShift,
							searchInitShift + searchMaxShift + 1)
			maskSize = np.max([
				0, searchMaxShift + searchInitShift,
				searchMaxShift - searchInitShift
			])
			if 2 * maskSize >= width:
				offsets[j] = np.nan
				break
			normedValues = [
				np.linalg.norm(
					(sa + np.roll(sb, i, axis=1))[:, maskSize:width - maskSize],
					ord=1) for i in IND
			]
			ARGMIN = np.argmin(normedValues)
			convergingSequence.append((ARGMIN, IND[ARGMIN]))
			#There shall be masked the same proportion of the sinogram to the left and right from searchInitShift to be ballanced
			#leftMask = np.max([0, searchMaxShift + searchInitShift])
			#rightMask = np.max([0, searchMaxShift - searchInitShift])
			#leftMask = np.max([rightMask, leftMask])
			#rightMask = leftMask
			#if leftMask + rightMask >= MAX:
			#	offsets[j] = np.nan
			#	break
			#normedValues = [np.linalg.norm((sa+np.roll(sb, i, axis=1))[:,leftMask:MAX-rightMask], ord=1) for i in IND]
			#ARGMIN = np.argmin(normedValues)
		else:  #On no break
			J = IND[ARGMIN]
			offsets[j] = 0.5 * J
			print("j=%d convergingSequence=%s offsets[j]=%d" %
				  (j, convergingSequence, offsets[j]))
	else:
		thetaSum = sinogram.sum(axis=0)
		thetaSum = thetaSum - thetaSum.mean()
		p, sigma, thetaSum = denoiseGaussian2(thetaSum)
		#thetaSum = gaussian_filter(thetaSum, len(thetaSum)/1000)
		centerOffset = 0
		MIN = 0
		MAX = len(thetaSum) - 1
		INDMAX = 100
		if ARG.max_derivative:
			dx = scipy.signal.savgol_filter(thetaSum,
											round(len(thetaSum) / 10),
											5,
											deriv=1)
			MIN = np.min([np.argmin(dx), np.argmax(dx)
						 ]) - INDMAX  #Offset for stability
			MAX = np.max([np.argmin(dx), np.argmax(dx)]) + INDMAX
		IND = np.arange(-INDMAX, INDMAX + 1)
		AMIN = np.max([0, MIN - INDMAX])
		AMAX = np.min([len(thetaSum), MAX + 1 + INDMAX])
		centerOffset = AMIN + (AMAX - 1 - AMIN) / 2 - (len(thetaSum) - 1) / 2
		testVec = thetaSum[AMIN:AMAX]
		testVecFlipped = -np.flip(testVec)
		J = IND[np.argmin([
			np.linalg.norm(
				(testVec + np.roll(testVecFlipped, i))[INDMAX:-INDMAX],
				ord=ARG.ord) for i in IND
		])]
		while np.abs(J) > 0.9 * INDMAX and INDMAX < 0.25 * len(thetaSum):
			INDMAX = 2 * INDMAX
			IND = np.arange(-INDMAX, INDMAX + 1)
			AMIN = np.max([0, MIN - INDMAX])
			AMAX = np.min([len(thetaSum), MAX + 1 + INDMAX])
			centerOffset = AMIN + (AMAX - 1 - AMIN) / 2 - (len(thetaSum) -
														   1) / 2
			testVec = thetaSum[AMIN:AMAX]
			testVecFlipped = -np.flip(testVec)
			J = IND[np.argmin([
				np.linalg.norm(
					(testVec + np.roll(testVecFlipped, i))[INDMAX:-INDMAX],
					ord=ARG.ord) for i in IND
			])]
		if sigma > 100:
			offsets[j] = np.nan
		else:
			offsets[j] = centerOffset + 0.5 * J
			print(
				"For j=%d y=%d the center of symmetry offset %.1f or double shift %d corresponds to J=%d  and centerOffset=%.1f"
				% (j, ySequence[j], centerOffset + 0.5 * J,
				   -J - 2 * centerOffset, J, centerOffset))
offset = np.nanmedian(offsets)
#Formatting as string shall give the full precision
print("rotation_center_offset_pix=%s" % (offset))
print("rotation_center_offset_pix_interp=%s" % (np.nanmedian(interpoffsets)))
#Create fit
#First compute pix size
pix_size = 1.0
if ARG.input_tick is None:
	h5 = h5py.File(ARG.input_h5, 'r')
	if "/entry/hardware/camera1" in h5:
		cam = "camera1"
	elif "/entry/hardware/camera" in h5:
		cam = "camera"
	else:
		raise ValueError(
			"There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."
			% info["h5"])
	pix_size_cam = float(h5["entry/hardware/%s/pixelsize" % cam][0])
	if ARG.override_magnification is not None:
		pix_size_mag = ARG.override_magnification
	else:
		pix_size_mag = float(h5["entry/hardware/%s/magnification" % cam][0])
	pix_size = float(pix_size_cam / pix_size_mag)
print("default_pix_size=%s"%(pix_size))

fittable=np.zeros([len(ySequence),5])
for j in range(len(ySequence)):
	fittable[j, 0] = ySequence[j]
	fittable[j, 1] = offsets[j]
	fittable[j, 2] = (offsets[j]+sinogram_center_offset) * pix_size
	fittable[j, 3] = interpoffsets[j]
	fittable[j, 4] = (interpoffsets[j]+sinogram_center_offset) * pix_size

# Calculate the Median Absolute Deviation (MAD)
def mad(data):
	median = np.nanmedian(data)
	mad_value = np.nanmedian(np.abs(data - median))
	return mad_value

#Compute global R2 as a metric of quality of the method before filtering
b, a = np.polyfit(fittable[:,0], fittable[:,3], 1)
predicted_offsets = a + b*fittable[:,0]
mae = np.mean(np.abs(fittable[:,3]-predicted_offsets))
ss_res = np.sum((fittable[:,3]-predicted_offsets)**2)
ss_tot = np.sum((fittable[:,3]-np.mean(fittable[:,3]))**2)
r2 = 1 - ss_res/ss_tot
print("global_mae=%s"%(mae))
print("global_r2=%s"%(r2))


# Compute MAD for interpoffsets
mad_interpoffsets = mad(interpoffsets)
# Use MAD to filter outliers
median_interpoffsets = np.nanmedian(interpoffsets)
# Here we use MAD to define a threshold for outliers
threshold = 2 * mad_interpoffsets
fittable = fittable[np.abs(fittable[:, 3] - median_interpoffsets) < threshold]

num_outliers_rejected = len(ySequence) - len(fittable)

if fittable.shape[0] > 3:
	b, a = np.polyfit(fittable[:,0], fittable[:,1], 1)
	#It is offset = a + b*y
	print("rotation_center_offset_pix_fit_a=%s"%(a))
	print("rotation_center_offset_pix_fit_b=%s"%(b))
	#Fit of interpoffsets
	b, a = np.polyfit(fittable[:,0], fittable[:,3], 1)
	predicted_offsets = a + b*fittable[:,0]
	#Compute MAE and R-squared error
	mae = np.mean(np.abs(fittable[:,3]-predicted_offsets))
	ss_res = np.sum((fittable[:,3]-predicted_offsets)**2)
	ss_tot = np.sum((fittable[:,3]-np.mean(fittable[:,3]))**2)
	r2 = 1 - ss_res/ss_tot
	print("rotation_center_offset_pix_interpfit_a=%s"%(a))
	print("rotation_center_offset_pix_interpfit_b=%s"%(b))
	print("Fit provides rotation_center_offset_pix=a + by = %f + %f y"%(a, b))
	if pix_size != 0:
		b, a = np.polyfit(fittable[:,0], fittable[:,2], 1)
		print("rotation_center_offset_fit_a=%s"%(a))
		print("rotation_center_offset_fit_b=%s"%(b))
		b, a = np.polyfit(fittable[:,0], fittable[:,4], 1)
		print("rotation_center_offset_interpfit_a=%s"%(a))
		print("rotation_center_offset_interpfit_b=%s"%(b))
	print("rotation_center_offset_pix_interp_mae=%s"%(mae))
	print("rotation_center_offset_pix_interp_r2=%s"%(r2))
print("Out %d measurements %d were rejected as outliers"%(len(ySequence), num_outliers_rejected))
print("Included offsets are %s"%(", ".join(["%0.2f"%x for x in fittable[:,3]])))

if ARG.log_file:
	sys.stdout.close()
	sys.stdout = sys.__stdout__
	os.rename("%s_tmp" % ARG.log_file, ARG.log_file)
if ARG.input_tick is not None:
	print("END detectRotationCenter180.py for extinctions %s and tick file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_tick)))
elif ARG.input_h5 is not None:
	print("END detectRotationCenter180.py for extinctions %s and H5 file %s" % (os.path.abspath(ARG.inputFile), os.path.abspath(ARG.input_h5)))
elif ARG.input_linspace:
	print("END detectRotationCenter180.py for extinctions %s and linspace" % (os.path.abspath(ARG.inputFile)))

