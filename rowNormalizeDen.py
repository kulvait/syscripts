#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
from denpy import DEN
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "16"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16"  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import sys
import time
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--value", type=float, default=1.0, help="Final row mean.")
parser.add_argument("--force", action="store_true")

ARG = parser.parse_args()
#First read dimensions of the DEN file
header = DEN.readHeader(ARG.inputDen)
if not header["dimcount"] in [2, 3]:
	print("File %s shall have dimension two or three" %
	      (os.path.basename(ARG.inputDen)))
	sys.exit(-1)
if header["dimcount"] == 2:
	zdim = 1
else:
	zdim = header["dimspec"][2]
xdim = header["dimspec"][0]
ydim = header["dimspec"][1]


DEN.writeEmptyDEN(ARG.outputDen, [xdim, ydim, zdim],
                  header["type"],
                  force=ARG.force)
#binshape = (ydim_bin, xdim_bin)
#newshape = (arraySize_bin, boxsize)

#numba
#newshape = (arraySize_bin, boxsize)
#g = np.zeros((arraySize_bin), dtype=np.float32)



for k in range(zdim):
	print("Start for %d" % k)
	start = time.time()
	f = DEN.getFrame(ARG.inputDen, k)
	f_mean = f.mean(axis=1)
	f_norm = f/f_mean[:,np.newaxis]
	f_norm *= ARG.value
	DEN.writeFrame(ARG.outputDen, k, f_norm, force=ARG.force)
	print("Elapsed %0.2fs" % (time.time() - start))
