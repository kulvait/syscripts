#!/usr/bin/env python
"""
Created 2024

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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file with fitted coefficients")
parser.add_argument("--input-h5", default=None, help="H5 file with dataset information")
parser.add_argument("--antisymmetric", action="store_true")
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
parser.add_argument(
    "--ord",
    default=1,
    type=int,
    help=
    "Order of norm, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html. Zero is special value to use Pearson corelation."
)
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file",
                    default=None,
                    help="Output to log file insted of stdout")
ARG = parser.parse_args()

if not os.path.isfile(ARG.inputFile):
	raise IOError("File %s does not exist" % os.path.abspath(ARG.inputFile))
else:
	print("START detrotzer.py for extinctions %s" %
	      (os.path.abspath(ARG.inputFile)))

if ARG.input_h5 is not None and not os.path.exists(ARG.input_h5):
	raise IOError("File %s does not exist" % os.path.abspath(ARG.inputH5))

if ARG.log_file:
	sys.stdout.flush()
	sys.stdout = open("%s_tmp" % ARG.log_file, "wt")

header = DEN.readHeader(ARG.inputFile)
if len(header["dimspec"]) != 3:
	raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!" %
	                (arg.inputFile, len(header["dimspec"])))
dimspec = header["dimspec"]
dimx = np.uint32(dimspec[0])
dimy = np.uint32(dimspec[1])
dimz = np.uint32(dimspec[2])

if dimx % 2 == 1:
	print("dimx=%d is odd"%(dimx))


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





if ARG.input_h5 is not None:
	df = PETRA.imageDataset(ARG.input_h5,
                        includePixelShift=True,
                        overrideMagnification=ARG.override_magnification)
#Now I estimate low and high limits of the shifts and subtract maximum from each side
#Maximum is there in order to keep center at the same position
	pixShifts = df["pixel_shift"]
	maxshift = pixShifts.max()
	minshift = pixShifts.min()
	ldrop = math.ceil(maxshift)
	rdrop = math.floor(minshift)
	drop = np.max([ldrop, -rdrop])
	if ARG.verbose:
		print("maxshift=%f, minshift=%f, ldrop=%d, rdrop=%d, drop=%d" % (maxshift, minshift, ldrop, rdrop, drop))

sinograms = DEN.getNumpyArray(ARG.inputFile)


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
			symmetry = "    SYMMETRIC"
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


offsets = np.zeros(dimz)

for j in range(dimz):
	sinogram = sinograms[j]
	#Subtracting means is not a good operation
	#sinogram = sinogram - sinogram.mean(axis=1, keepdims=True)
	MIN = 0
	MAX = dimy
	if ARG.svd:
		print("SVD j=%d y=%d" % (j, ySequence[j]))
		offsets[j] = svdAnalyze(sinogram)
		if not np.isnan(offsets[j]):
			print("For j=%d y=%d the center of symmetry offset %.1f" %
			      (j, ySequence[j], offsets[j]))
			print("")
	elif ARG.sinogram_consistency:
		halfAngleCount = sinogram.shape[0] // 2
		sa = sinogram
		sb = -np.flip(sinogram, axis=1)
		if j != 0:
			sb = -sb
		if ARG.antisymmetric:
			sb = -sb
		#Scan for starting point
		searchMaxShift = MAX // 3
		searchInitShift = 0
		IND = np.arange(searchInitShift - searchMaxShift,
		                searchInitShift + searchMaxShift + 1, 10)
		maskSize = np.max([
		    0, searchMaxShift + searchInitShift,
		    searchMaxShift - searchInitShift
		])
		if ARG.verbose:
			print("Searching [%d,%d] maskSize=%d"%(IND[0], IND[-1], maskSize))
		normedValues = np.zeros(len(IND))
		x1 = sa[:, maskSize:(MAX - maskSize)]
		x1 = x1.flatten()
		for i in range(len(IND)):
			x2 = np.roll(sb, IND[i], axis=1)[:, maskSize:(MAX - maskSize)]
			x2 = x2.flatten()#For corrcoef computation
			if ARG.ord == 0:
				normedValues[i] = np.corrcoef(x1, x2)[0,1]
			else:
				normedValues[i] = np.linalg.norm(x1 + x2, ord=ARG.ord)
		if ARG.verbose:
			plt.title("Init estimate")
			plt.plot(IND, normedValues)
			plt.show()
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
			if 2 * maskSize >= MAX:
				offsets[j] = np.nan
				break
			normedValues = np.zeros(len(IND))
			x1 = sa[:, maskSize:(MAX - maskSize)]
			x1 = x1.flatten()
			for i in range(len(IND)):
				x2 = np.roll(sb, IND[i], axis=1)[:, maskSize:(MAX - maskSize)]
				x2 = x2.flatten()#For corrcoef computation
				if ARG.ord == 0:
					normedValues[i] = np.corrcoef(x1, x2)[0,1]
				else:
					normedValues[i] = np.linalg.norm(x1 + x2, ord=ARG.ord)
			ARGMIN = np.argmin(normedValues)
			convergingSequence.append((ARGMIN, IND[ARGMIN]))
		else:  #On no break
			J = IND[ARGMIN]
			offsets[j] = 0.5 * J
			print("j=%d convergingSequence=%s offsets[j]=%f" %
			      (j, convergingSequence, offsets[j]))
		if ARG.verbose:
			plt.title("Error norm for sinogram consistendy")
			plt.plot(IND, normedValues)
			plt.show()
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
			    "For j=%d the center of symmetry offset %.1f or double shift %d corresponds to J=%d  and centerOffset=%.1f"
			    % (j, centerOffset + 0.5 * J,
			       -J - 2 * centerOffset, J, centerOffset))
offset = np.nanmedian(offsets)
if ARG.log_file:
	sys.stdout.close()
	sys.stdout = sys.__stdout__
	os.rename("%s_tmp" % ARG.log_file, ARG.log_file)
print("END detectRotationCenter.py for extinctions %s" %
      (os.path.abspath(ARG.inputFile)))
