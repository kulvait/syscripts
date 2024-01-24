#!/usr/bin/env python
"""
Created 2024

@author: Vojtech Kulvait

This script produces camera matrices for given dataset based on H5 file and provided dimensions of pixels

"""

import argparse
import scipy.io
import numpy as np
import os
from denpy import DEN


parser = argparse.ArgumentParser()
parser.add_argument("anglesMat", help="Mat file with angle sequence")
parser.add_argument("shiftsMat", help="Mat file with shift sequence")
parser.add_argument("outputTickFile", help="Output series of tuples in the form (angle, pixel_shift)")
parser.add_argument("--binning-factor", default=None, type=float, help="Binning not considered in pixel shifts.")
parser.add_argument("--inverted-pixshifts", action="store_true")

parser.add_argument(
    "--zero-offsets", action="store_true",
    help="Write zero offsets regardless of shiftsMat file.")
parser.add_argument("--force", action="store_true")
ARG = parser.parse_args()

angles = scipy.io.loadmat(ARG.anglesMat)
angles = angles["angles"].flatten()
angles = np.array(angles, dtype=np.float32)
minAngle = angles.min() 
angles = angles-minAngle
if angles.max() <= 2.0 * np.pi:
	angles = angles*180.0/np.pi
if ARG.zero_offsets:
	shifts = np.zeros(len(angles), dtype=np.float32)
	print("MAT to %s min(angles)=%f max(angles)=%f zero shifts"%(os.path.abspath(ARG.outputTickFile), np.min(angles), np.max(angles)))
else:
	shifts = scipy.io.loadmat(ARG.shiftsMat)
	shifts = shifts["offset_shift"].flatten()
	shifts = np.array(shifts, dtype=np.float32)
	shiftOffset = 0.5*(np.max(shifts) + np.min(shifts))
	shifts = shifts - shiftOffset
	if ARG.binning_factor is not None:
		shifts = shifts / ARG.binning_factor
	if ARG.inverted_pixshifts:
		shifts = -shifts
	print("MAT to %s min(angles)=%f max(angles)=%f min(shifts)=%f max(shifts)=%f"%(os.path.abspath(ARG.outputTickFile), np.min(angles), np.max(angles), np.min(shifts), np.max(shifts)))

ticks = np.zeros((2, len(angles)), dtype=np.float32)#Angle, pixel_shift
ticks[0,:] = angles#angles
ticks[1,:] = shifts#pixshifts
DEN.storeNdarrayAsDEN(ARG.outputTickFile, ticks, force=ARG.force)
