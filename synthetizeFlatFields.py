#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

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
parser.add_argument("inputSVD", help="DEN file A to create matrix of scalar products")
parser.add_argument("inputFlatCoefficients", help="DEN file B to create matrix of scalar products")
parser.add_argument("predictedFlatField", help="Output DEN with scalar products")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
ARG = parser.parse_args()

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

infV = DEN.readHeader(ARG.inputSVD)
infM = DEN.readHeader(ARG.inputFlatCoefficients)

if infM["dimcount"] != 2 : 
	if infM["dimcount"] != 3 and infM["dimspec"][2] != 1:
		raise ValueError("infM[\"dimcount\"] = %d "%(infM["dimcount"]))


if infV["dimcount"] != 3:
	raise ValueError("infV[\"dimcount\"] = %d but it shall be 3", infV["dimcount"])

projCount = infM["dimspec"][0]
vCount = infV["dimspec"][2]
vUsedCount = infM["dimspec"][1]
dimx = infV["dimspec"][0]
dimy = infV["dimspec"][1]

if vUsedCount > vCount:
	raise ValueError("Number of used vectors %d is greater than vectors in SVD %d"%(vUsedCount, vCount))


if vUsedCount < 101:
	V = np.zeros(shape=(vUsedCount, dimy, dimx), dtype=np.float32)
	for z in range(vUsedCount):
		V[z] = DEN.getFrame(ARG.inputSVD, z)
else:
	V = None

A = DEN.getFrame(ARG.inputFlatCoefficients, 0 )

dimspec = [dimx, dimy, projCount]
DEN.writeEmptyDEN(ARG.predictedFlatField, dimspec, force=ARG.force)
for k in range(projCount):
	ff = np.zeros(shape=(dimy, dimx), dtype=np.float32)
	for i in range(vUsedCount):
		if V is not None:
			SV = V[i]
		else:
			SV = DEN.getFrame(ARG.inputSVD, i)
		ff = ff + SV * A[i, k]
	DEN.writeFrame(ARG.predictedFlatField, k, ff, force=True)
