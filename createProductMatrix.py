#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

@author: Vojtech Kulvait

Frame product. Faster implementation in dentk-frameproduct, where however inverse is not computed and matrix is transposed with respect to this output.
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
import multiprocessing as mp
import itertools

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
parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")
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

frameSize = infA["dimspec"][0]*infA["dimspec"][1]
frameSizeB = infB["dimspec"][0]*infB["dimspec"][1]
if frameSize != frameSizeB:
	raise ValueError("Objects A and B have different framesizes.")

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

F = np.zeros(shape=(rowCount, colCount), dtype=np.float32)

if ARG.threads == -1:
	ARG.threads = mp.cpu_count()
	if ARG.verbose:
		print("Will be computing products of A=%s and B=%s data with ARG.threads=mp.cpu_count()=%d."%(ARG.inputFileA, ARG.inputFileB, ARG.threads))

#See also https://stackoverflow.com/questions/44763910/multiply-2d-numpy-arrays-element-wise-and-sum
def tenzorProduct(A_ravel, B_ravel):
	return A_ravel.dot(B_ravel)

if ARG.threads != 0:
	H = np.zeros(shape=(rowCount, colCount), dtype=np.object_)
	print("Creating pool with %d threads"%ARG.threads)
	pool = mp.Pool(ARG.threads)
	if equalInputs:
		for i in range(rowCount):
			A = DEN.getFrame(ARG.inputFileA, i)
			A = A.ravel()
			for j in range(colCount):
				if j >= i or i >= rowCount:
					B = DEN.getFrame(ARG.inputFileB, j)
					H[i, j] = pool.apply_async(tenzorProduct, args=(A, B.ravel()))
		pool.close()
		pool.join()
		for i in range(rowCount):
			for j in range(colCount):
				if j >= i or i >= rowCount:
					F[i,j] = H[i, j].get()
				else:
					F[i,j] = F[j,i]
	else:
		if rowCount < 500:
			A_matrices = np.zeros(shape=(rowCount, frameSize), dtype=np.float32)
			for i in range(rowCount):
				A_matrices[i] = DEN.getFrame(ARG.inputFileA, i).ravel()
			for j in range(colCount):
				print("Just processing %010d/%010d files"%(j, colCount))
				B = DEN.getFrame(ARG.inputFileB, j).ravel()
				F[:,j] = pool.map(B.dot, A_matrices)
				#for i in range(rowCount):
					#F[i, j] = A_matrices[i].dot(B)
				#	H[i, j] = pool.apply_async(tenzorProduct, args=(A_matrices[i], B))
				#for i in range(rowCount):
				#	H[i,j].wait()
				#	F[i, j] = H[i,j].get()
			pool.close()
			pool.join()
#			for i in range(rowCount):
#				for j in range(colCount):
#					F[i, j] = H[i,j].get()
		else:
			for i in range(rowCount):
				A = DEN.getFrame(ARG.inputFileA, i)
				A = A.ravel()
				print("Just read %010d/%010d files"%(i, rowCount))
				for j in range(colCount):
					B = DEN.getFrame(ARG.inputFileB, j)
					H[i, j] = pool.apply_async(tenzorProduct, args=(A, B.ravel()))
			pool.close()
			pool.join()
			for i in range(rowCount):
				for j in range(colCount):
					F[i, j] = H[i,j].get()
else:
	for i in range(rowCount):
		A = DEN.getFrame(ARG.inputFileA, i)
		A = A.ravel()
		for j in range(colCount):
			if equalInputs and j < i and i < rowCount:
				F[i, j] = F[j, i]
			else:
				B = DEN.getFrame(ARG.inputFileB, j).ravel()
				F[i,j] = tenzorProduct(A, B)

if ARG.inverse is not None:
	I = np.linalg.inv(F)
	DEN.storeNdarrayAsDEN(ARG.inverse, I, force=ARG.force)

DEN.storeNdarrayAsDEN(ARG.productMatrix, F, force=ARG.force)
