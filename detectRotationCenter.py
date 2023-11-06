#!/usr/bin/env python
"""
Created 2022

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
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file with projected extinctions")
parser.add_argument("inputH5", help="H5 file with dataset information")
parser.add_argument("--sample-count", default=10, type=int, help="Number of equaly spaced horizontal lines to select for computation")
parser.add_argument("--angle-count", default=720, type=int, help="Number of equaly spaced angles to select for computation")
parser.add_argument("--override-magnification", default=None, type=float, help="Use this magnification value instead of the one in H5 file")
parser.add_argument("--store-sinograms", default=None, type=str, help="Use to store sinograms to the DEN file.")
parser.add_argument("--load-sinograms", default=None, type=str, help="Load sinograms from DEN file.")
parser.add_argument("--ord", default=2, type=int, help="Order of norm, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html.")
parser.add_argument("--sinogram-consistency", action="store_true", help="Search for the offcet with maximum consistency so that f(x)-f(x+pi) is optimal.")
parser.add_argument("--max-derivative", action="store_true", help="Search by the algorithm of extreme derivative.")
parser.add_argument("--svd", action="store_true", help="Perform SVD analysis to determine center.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
ARG = parser.parse_args()

if not os.path.isfile(ARG.inputFile):
	raise IOError("File %s does not exist"%os.path.abspath(ARG.inputFile))
else:
	print("START 06detectRotationCenter.py for extinctions %s and H5 file %s"%(os.path.abspath(ARG.inputFile), os.path.abspath(ARG.inputH5)))

if not os.path.exists(ARG.inputH5):
	raise IOError("File %s does not exist"%os.path.abspath(ARG.inputH5))

if ARG.load_sinograms is not None and not os.path.isfile(ARG.load_sinograms):
	raise IOError("File %s to load sinograms does not exist"%os.path.abspath(ARG.load_sinograms))

if ARG.log_file:
	sys.stdout.flush()
	sys.stdout = open("%s_tmp"%ARG.log_file, "wt")

if ARG.sinogram_consistency and ARG.angle_count % 2 ==1:
	raise AttributeError("When compuging sinogram consistency ARG.angle_count=%d needs to be divisible by 2"%(ARG.angle_count))

def shiftFrame(x, shiftSize):
	if shiftSize == 0:
		return x
	f = np.roll(x, [0, shiftSize])
	if shiftSize > 0:
		f[:,:shiftSize] = 0.0
	else:#negative
		f[:,shiftSize:] = 0.0
	return f

#Shift by float shift size
def shiftFrameFloat(f, shiftSize):
	intShift = math.floor(shiftSize)
	floatShift = shiftSize - intShift
	intShift = int(intShift)
	f1 = shiftFrame(f, intShift)
	f2 = shiftFrame(f, intShift+1)
	return f1*(1.0-floatShift) + f2*floatShift

#There is likely that the dataset does not contain given angle. In that case use interpolation
#Otherwise use matching frames
#Return frame with applied shift
def getInterpolatedFrame(inputFile, df, angle):
	exactMatches = np.sum(df["s_rot"] == angle)
	if exactMatches > 0:
		df = df.loc[df["s_rot"] == angle]
		f = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[0])
		f = shiftFrameFloat(f, df["pixel_shift"].iloc[0])
		if exactMatches > 1:
			for k in range(exactMatches - 1):
				g = DEN.getFrame(ARG.inputFile, df["frame_ind"][k])
				g = shiftFrameFloat(g, df["pixel_shift"][k])
				f = f + g
			f = f / exactMatches
	else:#I have to interpolate
		df = df.sort_values("s_rot")
		closestHigherInd = np.searchsorted(df["s_rot"], angle, side="right")
		if closestHigherInd == 0:
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]-360.000001
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
		elif closestHigherInd == len(df):
			closestLowerInd = len(df) - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd]
			closestHigherInd = 0
			closestHigherAngle =  df["s_rot"].iloc[0]+360.000001
		else:
			closestHigherAngle = df["s_rot"].iloc[closestHigherInd]
			closestLowerInd = closestHigherInd - 1
			closestLowerAngle = df["s_rot"].iloc[closestLowerInd] 
		angleDiff = closestHigherAngle - closestLowerAngle
		lo = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestLowerInd])
		lo = shiftFrameFloat(lo, df["pixel_shift"].iloc[closestLowerInd])
		hi = DEN.getFrame(ARG.inputFile, df["frame_ind"].iloc[closestHigherInd])
		hi = shiftFrameFloat(hi, df["pixel_shift"].iloc[closestHigherInd])
		lofac = (closestHigherAngle-angle)/angleDiff
		if angleDiff > 1.0:
			raise Exception("Maximal uncoverred rotation angle gap anglediff=%f > 1.0 can not use this method"%angleDiff)
			sys.exit()
		if ARG.verbose:
			print("closestHigherAngle=%f angleDiff=%f closestLowerAngle=%f lofac=%f angle=%f"%(closestHigherAngle, angleDiff, closestLowerAngle, lofac, angle))
		f = lofac*lo + (1.0-lofac)*hi
	return f

header = DEN.readHeader(ARG.inputFile)
if len(header["dimspec"]) != 3:
	raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!"%(arg.inputFile, len(header["dimspec"])))
dimspec = header["dimspec"]
xdim = np.uint32(dimspec[0])
ydim = np.uint32(dimspec[1])
zdim = np.uint32(dimspec[2])

#Equaly spaced Y indices
if ARG.sample_count < 1 or ARG.sample_count > ydim:
	raise ValueError("Invalid sample_count=%d"%ARG.sample_count)
ySequence = np.array(np.round(np.linspace(0, ydim, num=ARG.sample_count + 2, endpoint=True))[1:-1], dtype=np.int32)
if ARG.sample_count == ydim:
	ySequence = np.arange(ydim)


if ARG.angle_count < 1:
	raise ValueError("Invalid angle_count=%d"%ARG.angle_count)

#Angles to use in calculation
angleSequence = np.linspace(0, 360, num=ARG.angle_count, endpoint=False)
sinograms = np.zeros([ARG.sample_count, ARG.angle_count, xdim], dtype=np.float32)
df = PETRA.imageDataset(ARG.inputH5, includePixelShift=True, overrideMagnification=ARG.override_magnification)
#Now I estimate low and high limits of the shifts and subtract maximum from each side
#Maximum is there in order to keep center at the same position
pixShifts = df["pixel_shift"]
maxshift = pixShifts.max()
minshift = pixShifts.min()
ldrop = math.floor(maxshift)
rdrop = math.ceil(minshift)
drop = np.max([ldrop, -rdrop])

if ARG.load_sinograms is not None:
	info = DEN.readHeader(ARG.load_sinograms)
	if len(info["dimspec"]) != 3:
		raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!"%(arg.load_sinograms, len(info["dimspec"])))
	if info["dimspec"][1] != ARG.angle_count:
		print("Setting angle_count to %d from ARG.angle_count %d"%(info["dimspec"][1], ARG.angle_count))
		ARG.angle_count = info["dimspec"][1]
		angleSequence = np.linspace(0, 360, num=ARG.angle_count, endpoint=False)
	sinograms = DEN.getNumpyArray(ARG.load_sinograms)
else:
	for angindex in range(len(angleSequence)):
		ang = angleSequence[angindex]
		frame = getInterpolatedFrame(ARG.inputFile, df, ang)
		for j in range(len(ySequence)):
			sinograms[j,angindex] = frame[ySequence[j]]
	#Already applied on stored sinograms
	sinograms = sinograms[:,:,drop:-drop]

if ARG.store_sinograms is not None:
	DEN.storeNdarrayAsDEN(ARG.store_sinograms, sinograms, force=ARG.force)

def denoiseGaussian2(vec, lim=1):
	maxsigma = len(vec)/10
	minsigma = 1.0
	cursigma = 1.0
	lim_tol = lim/100
	gf = gaussian_filter(vec, minsigma)
	pval = scipy.stats.anderson(vec - gf)[0]
	if pval > lim:
		return pval, minsigma, gf
	for i in range(10):
		cursigma = (maxsigma + minsigma)/2
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
	maxsigma = len(vec)/10
	minsigma = 1.0
	cursigma = 1.0
	gf = gaussian_filter(vec, minsigma)
	pval = scipy.stats.normaltest(vec - gf)[1]
	if pval < 1e-10:
		return pval, minsigma, gf
	for i in range(10):
		cursigma = (maxsigma + minsigma)/2
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
		cursigma = (maxsigma + minsigma)/2
		gf = gaussian_filter(vec, cursigma)
		pval = scipy.stats.skewtest(vec - gf)[1]
		if np.abs(pval-0.1) < 0.001 :
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
	significantCount = np.sum(S/S[0] > 0.1)
	if significantCount < 5:
		print("Lot of low quality components")
		return np.nan
	offsets = np.zeros(significantCount)
	print("Spectral ratio is %0.1f"%(S[0]/S[1]))
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
		Vg_plus  = np.linalg.norm(Vg + Vgf, ord=ARG.ord)
		Vg_minus  = np.linalg.norm(Vg - Vgf, ord=ARG.ord)
		MIN = 0
		MAX = vlx - 1
		if Vg_plus < Vg_minus:
			symmetry = "ANTISYMMETRIC"
			normFit = Vg_plus/Vg_norm
			antisymmetric = 1
		else:
			symmetry = "    SYMMETRIC"
			normFit = Vg_minus/Vg_norm
			antisymmetric = -1
		if ARG.max_derivative:
			if Vg_plus < Vg_minus:
				dx = Vg
			else:
				dx = savgol_filter(Vg, round(len(Vg)/10), 5, deriv=1)
			MIN = np.min([np.argmin(dx), np.argmax(dx)])-INDMAX
			MAX = np.max([np.argmin(dx), np.argmax(dx)])+INDMAX
		IND = np.arange(-INDMAX, INDMAX+1)
		AMIN = np.max([0, MIN-INDMAX])
		AMAX = np.min([vlx, MAX+1+INDMAX])
		centerOffset = 0.5*AMIN + 0.5*AMAX - vlx*0.5
		testVec = Vg[AMIN:AMAX]
		testVecFlipped = antisymmetric * np.flip(testVec)
		J = IND[np.argmin([np.linalg.norm((testVec+np.roll(testVecFlipped, i))[INDMAX:-INDMAX], ord=ARG.ord) for i in IND])]
		while np.abs(J) > 0.9*INDMAX and INDMAX < 0.25*vlx:
			INDMAX = 2*INDMAX
			IND = np.arange(-INDMAX, INDMAX+1)
			AMIN = np.max([0, MIN-INDMAX])
			AMAX = np.min([vlx, MAX+1+INDMAX])
			centerOffset = 0.5*AMIN + 0.5*AMAX - vlx*0.5
			testVec = Vg[AMIN:AMAX]
			testVecFlipped = antisymmetric * np.flip(testVec)
			J = IND[np.argmin([np.linalg.norm((testVec+np.roll(testVecFlipped, i))[INDMAX:-INDMAX], ord=ARG.ord) for i in IND])]
		print("Component %d %s, offset %5.1f pix, position %5.1f pix, Vg_plusminus/Vg_norm=%0.3f lower better, centerOffset=%0.1f, J=%5d, SV=%3.2f, pval=%f, sigma=%f."%(i, symmetry, centerOffset + 0.5*J, vlx*0.5 - 0.5 + centerOffset + 0.5*J, normFit, centerOffset, J, S[i], pval, sigma))
		offsets[i] = centerOffset + 0.5*J
		i = i + 1
	return np.median(offsets)

offsets = np.zeros(len(ySequence))

for j in range(len(ySequence)):
	sinogram = sinograms[j] 
	#Subtracting means is not a good operation
	#sinogram = sinogram - sinogram.mean(axis=1, keepdims=True)
	MIN = 0
	MAX = sinogram.shape[1]
	if ARG.svd:
		print("SVD j=%d y=%d"%(j, ySequence[j]))
		offsets[j] = svdAnalyze(sinogram)
		if not np.isnan(offsets[j]):
			print("For j=%d y=%d the center of symmetry offset %.1f"%(j, ySequence[j], offsets[j]))
			print("")
	elif ARG.sinogram_consistency:
		halfAngleCount =  sinogram.shape[0]//2
		sa = sinogram[0:halfAngleCount]
		sb = -np.flip(sinogram[halfAngleCount:], axis=1)
		#Scan for starting point
		searchMaxShift = MAX//3
		searchInitShift = 0
		IND = np.arange(searchInitShift-searchMaxShift,searchInitShift+searchMaxShift+1, 10)
		maskSize = np.max([0, searchMaxShift + searchInitShift, searchMaxShift - searchInitShift])
		normedValues = [np.linalg.norm((sa+np.roll(sb, i, axis=1))[:, maskSize:MAX-maskSize], ord=1) for i in IND]
		ARGMIN = np.argmin(normedValues)
		searchInitShift = IND[ARGMIN]
		searchMaxShift = 100
		ARGMIN = 0
		IND = [searchInitShift]
		convergingSequence = []
		while len(convergingSequence) < 1 or (convergingSequence[-1][0] != searchMaxShift and monotonicityChanges(convergingSequence) < 3):
			searchInitShift = IND[ARGMIN]
			IND = np.arange(searchInitShift-searchMaxShift,searchInitShift+searchMaxShift+1)
			maskSize = np.max([0, searchMaxShift + searchInitShift, searchMaxShift - searchInitShift])
			if 2*maskSize >= MAX:
				offsets[j] = np.nan
				break
			normedValues = [np.linalg.norm((sa+np.roll(sb, i, axis=1))[:, maskSize:MAX-maskSize], ord=1) for i in IND]
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
		else:#On no break
			J = IND[ARGMIN]
			offsets[j] = 0.5*J
			print("j=%d convergingSequence=%s offsets[j]=%d"%(j, convergingSequence, offsets[j]))
	else:
		thetaSum = sinogram.sum(axis = 0)
		thetaSum = thetaSum - thetaSum.mean()
		p, sigma, thetaSum = denoiseGaussian2(thetaSum)
		#thetaSum = gaussian_filter(thetaSum, len(thetaSum)/1000)
		centerOffset = 0
		MIN = 0
		MAX = len(thetaSum)-1
		INDMAX = 100
		if ARG.max_derivative:
			dx = scipy.signal.savgol_filter( thetaSum, round(len(thetaSum)/10), 5, deriv=1)
			MIN = np.min([np.argmin(dx), np.argmax(dx)])-INDMAX #Offset for stability
			MAX = np.max([np.argmin(dx), np.argmax(dx)])+INDMAX
		IND = np.arange(-INDMAX, INDMAX+1)
		AMIN = np.max([0, MIN-INDMAX])
		AMAX = np.min([len(thetaSum), MAX+1+INDMAX])
		centerOffset = AMIN + (AMAX - 1 - AMIN)/2 - (len(thetaSum)-1)/2
		testVec = thetaSum[AMIN:AMAX]
		testVecFlipped = -np.flip(testVec)
		J = IND[np.argmin([np.linalg.norm((testVec+np.roll(testVecFlipped, i))[INDMAX:-INDMAX], ord=ARG.ord) for i in IND])]
		while np.abs(J) > 0.9*INDMAX and INDMAX < 0.25 * len(thetaSum):
			INDMAX = 2*INDMAX
			IND = np.arange(-INDMAX, INDMAX+1)
			AMIN = np.max([0, MIN-INDMAX])
			AMAX = np.min([len(thetaSum), MAX+1+INDMAX])
			centerOffset = AMIN + (AMAX - 1 - AMIN)/2 - (len(thetaSum)-1)/2
			testVec = thetaSum[AMIN:AMAX]
			testVecFlipped = -np.flip(testVec)
			J = IND[np.argmin([np.linalg.norm((testVec+np.roll(testVecFlipped, i))[INDMAX:-INDMAX], ord=ARG.ord) for i in IND])]
		if sigma > 100:
			offsets[j] = np.nan
		else:
			offsets[j] = centerOffset + 0.5*J
			print("For j=%d y=%d the center of symmetry offset %.1f or double shift %d corresponds to J=%d  and centerOffset=%.1f"%(j, ySequence[j], centerOffset + 0.5*J, -J-2*centerOffset, J, centerOffset))
offset = np.nanmedian(offsets)
h5 = h5py.File(ARG.inputH5, 'r')
if "/entry/hardware/camera1" in h5:
	cam="camera1"
elif "/entry/hardware/camera" in h5:
	cam="camera"
else:
	raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."%info["h5"])
pix_size_cam = float(h5["entry/hardware/%s/pixelsize"%cam][0])
if ARG.override_magnification is not None:
	pix_size_mag = ARG.override_magnification
else:
	pix_size_mag = float(h5["entry/hardware/%s/magnification"%cam][0])
pix_size = float(pix_size_cam/pix_size_mag)
print("rotation_center_offset_pix=%f"%(offset))
print("rotation_center_offset=%f"%(offset*pix_size))
if ARG.log_file:
	sys.stdout.close()
	sys.stdout = sys.__stdout__
	os.rename("%s_tmp"%ARG.log_file, ARG.log_file)
print("END 06detectRotationCenter.py for extinctions %s and H5 file %s"%(os.path.abspath(ARG.inputFile), os.path.abspath(ARG.inputH5)))
