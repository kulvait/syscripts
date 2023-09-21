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
parser.add_argument("inputFileA", help="DEN file A to create matrix of scalar products")
parser.add_argument("inputFileB", help="DEN file B to create matrix of scalar products")
parser.add_argument("productMatrix", help="Output DEN with scalar products")
parser.add_argument("--inverse", default=None, type=str, help="Store also inverse of the matrix")
parser.add_argument("--count-left", default=None, type=int, help="Number of projections from A to consider equals number of rows.")
parser.add_argument("--count-right", default=None, type=int, help="Number of projections from B to consider equals number of columns.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
ARG = parser.parse_args()

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

equalInputs = False
if ARG.inputFileA == ARG.inputFileB:
	equalInputs = True

infA = DEN.readHeader(ARG.inputFileA)
infB = DEN.readHeader(ARG.inputFileB)

if infA["dimcount"] != 3 or infB["dimcount"] != 3:
	raise ValueError("infA[\"dimcount\"] = %d and  infB[\"dimcount\"] =%d but it shall be 3", infA["dimcount"], infB["dimcount"])


rowCount = infA["dimspec"][2]
colCount = infB["dimspec"][2]

if ARG.count_left is not None:
	if ARG.count_left <= rowCount:
		rowCount = ARG.count_left
	else:
		raise IOError("ARG.count_left=%d is higher than number of frames in %s=%d"%(ARG.count_left, ARG.inputFileA, infA["dimspec"][2]))

if ARG.count_right is not None:
	if ARG.count_right <= colCount:
		colCount = ARG.count_right
	else:
		raise IOError("ARG.count_right=%d is higher than number of frames in %s=%d"%(ARG.count_right, ARG.inputFileB, infB["dimspec"][2]))

if ARG.inverse is not None and colCount != rowCount:
	raise IOError("To compute inverse rowcount=%d shall be equal to colCount=%d."%(rowCount, colCount))

F = np.zeros(shape=(rowCount, colCount),dtype=np.float32)
for i in range(rowCount):
	A = DEN.getFrame(ARG.inputFileA, i)
	for j in range(colCount):
		if equalInputs and j < i and i < rowCount:
			F[i, j] = F[j, i]
		else:
			B = DEN.getFrame(ARG.inputFileB, j)
			F[i,j] = A.ravel().dot(B.ravel()) #See also https://stackoverflow.com/questions/44763910/multiply-2d-numpy-arrays-element-wise-and-sum

if ARG.inverse is not None:
	I = np.linalg.inv(F)
	DEN.storeNdarrayAsDEN(ARG.inverse, I, force=ARG.force)

DEN.storeNdarrayAsDEN(ARG.productMatrix, F, force=ARG.force)
