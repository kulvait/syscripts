#!/usr/bin/env python3
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
parser.add_argument("--bin-x",
                    type=int,
                    default=1,
                    help="X dimension of binning box.")
parser.add_argument("--bin-y",
                    type=int,
                    default=1,
                    help="Y dimension of binning box.")
parser.add_argument("--force", action="store_true")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--average', action='store_true')
group.add_argument('--median', action='store_true')

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
bin_x = ARG.bin_x
bin_y = ARG.bin_y
xdim_red = xdim - xdim % ARG.bin_x
ydim_red = ydim - ydim % ARG.bin_y
xdim_out = xdim_red // ARG.bin_x
ydim_out = ydim_red // ARG.bin_y
if xdim_red < 1 or ydim_red < 1 or xdim_out < 1 or ydim_out < 1:
	print("Dimensions are zero after reduction!")
	sys.exit(-1)
if xdim % ARG.bin_x != 0:
	print("Xdim %d has to be recuced to %d" % (xdim, xdim_red))
if ydim % ARG.bin_y != 0:
	print("Ydim %d has to be recuced to %d" % (ydim, ydim_red))

DEN.writeEmptyDEN(ARG.outputDen, [xdim_out, ydim_out, zdim],
                  header["type"],
                  force=ARG.force)
print("Created file [dimx, dimy, dimz] = [%d, %d, %d]" %
      (xdim_out, ydim_out, zdim))

boxsize = bin_x * bin_y
arraySize_red = ydim_red * xdim_red
arraySize_bin = arraySize_red // boxsize
ydim_bin = ydim_red // bin_y
xdim_bin = xdim_red // bin_x
newshape = (ydim_bin, xdim_bin, boxsize)
g = np.zeros((ydim_bin, xdim_bin), dtype=np.float32)
#binshape = (ydim_bin, xdim_bin)
#newshape = (arraySize_bin, boxsize)

#numba
#newshape = (arraySize_bin, boxsize)
#g = np.zeros((arraySize_bin), dtype=np.float32)


def medianx(f, g):
	for ind in range(g.shape[0]):
		g[ind] = statistics.median(f[ind])


for k in range(zdim):
	print("Start for %d" % k)
	start = time.time()
	f = DEN.getFrame(ARG.inputDen, k, row_to=ydim_red, col_to=xdim_red)
	f.shape = (ydim_bin, bin_y, xdim_bin, bin_x)
	f_reshaped = f.swapaxes(1, 2)
	f_reshaped = f_reshaped.reshape(newshape)
	if ARG.median:
		np.median(f_reshaped, axis=-1, overwrite_input=True, out=g)
	if ARG.average:
		g = np.mean(f_reshaped, axis=-1)
	DEN.writeFrame(ARG.outputDen, k, g, force=ARG.force)
	print("Elapsed %0.2fs" % (time.time() - start))
