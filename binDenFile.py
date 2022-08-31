#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
from denpy import DEN
import numpy as np
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--bin-x", type=int, default=1, help="X dimension of binning box.")
parser.add_argument("--bin-y", type=int, default=1, help="Y dimension of binning box.")
parser.add_argument("--force", action="store_true")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--average', action='store_true')
group.add_argument('--median',	action='store_true')


ARG = parser.parse_args()
#First read dimensions of the DEN file
header = DEN.readHeader(ARG.inputDen)
if not header["dimcount"] in [2,3]:
	print("File %s shall have dimension two or three"%(os.path.basename(ARG.inputDen)))
	sys.exit(-1)
if header["dimcount"] == 2:
	zdim = 1
else:
	zdim = header["dimspec"][2]
xdim = header["dimspec"][0]
ydim = header["dimspec"][1]
xdim_red = xdim - xdim % ARG.bin_x
ydim_red = ydim - ydim % ARG.bin_y
xdim_out = xdim_red // ARG.bin_x
ydim_out = ydim_red // ARG.bin_y
if xdim_red < 1 or ydim_red <1 or xdim_out <1 or ydim_out<1:
	print("Dimensions are zero after reduction!")
	sys.exit(-1)
if xdim % ARG.bin_x != 0:
	print("Xdim %d has to be recuced to %d"%(xdim, xdim_red))
if ydim % ARG.bin_y != 0:
	print("Ydim %d has to be recuced to %d"%(ydim, ydim_red))

DEN.writeEmptyDEN(ARG.outputDen, [xdim_out, ydim_out, zdim], header["type"], force=ARG.force)
print("Created file [dimx, dimy, dimz] = [%d, %d, %d]"%(xdim_out, ydim_out, zdim))

def reduce(ARG, box):
	if ARG.median:
		return np.median(box)
	if ARG.average:
		return np.mean(box)

def binning(ARG, f, bin_x, bin_y):
	m,n = np.shape(f)
	bin_m = m//bin_y
	bin_n = n//bin_x
	strides_spec = f.itemsize*np.array([(m // bin_m) * n, (n // bin_n), n, 1], dtype=np.int32)
	print("Spec [%d %d %d %d] dtype=%s"%(m/bin_m*n, n/bin_n, n, 1, strides_spec.dtype))
	strided_reshape = np.lib.stride_tricks.as_strided(f,shape=(bin_m,bin_n,m//bin_m,n//bin_n),strides = strides_spec)
	return np.array([reduce(ARG, col) for row in strided_reshape for col in row]).reshape(bin_m,bin_n)


for k in range(zdim):
	f = DEN.getFrame(ARG.inputDen, k, row_to = ydim_red, col_to = xdim_red)
	#According to https://stackoverflow.com/questions/40097213/how-do-i-median-bin-a-2d-image-in-python
	g = binning(ARG, f, ARG.bin_x, ARG.bin_y)
	print(g.shape)
	DEN.writeFrame(ARG.outputDen, k, g, force=ARG.force)

