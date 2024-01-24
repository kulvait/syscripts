#!/usr/bin/env python
"""
Created 2024

@author: Vojtech Kulvait

This script produces camera matrices for given dataset based on H5 file and provided dimensions of pixels

"""

import argparse
import glob
import os
import errno
import sys
import h5py
import math
import re
from denpy import DEN
from denpy import UTILS
from denpy import PETRA
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("inputH5", help="H5 file with dataset information")
parser.add_argument("outputTickFile", help="Output series of tuples in the form (angle, pixel_shift)")
parser.add_argument(
    "--override-magnification",
    default=None,
    type=float,
    help="Use this magnification value instead of the one in H5 file")
parser.add_argument("--binning-factor", default=None, type=float, help="Binning not considered in pixel shifts.")
parser.add_argument("--inverted-pixshifts", action="store_true")
parser.add_argument("--force", action="store_true")
ARG = parser.parse_args()

df = PETRA.imageDataset(ARG.inputH5, includePixelShift=True, overrideMagnification=ARG.override_magnification)

angles = df["s_rot"].to_numpy()
shifts = df["pixel_shift"].to_numpy()
angles = np.array(angles, dtype=np.float32)
shifts = np.array(shifts, dtype=np.float32)
minAngle = angles.min() 
angles = angles-minAngle
shiftOffset = 0.5*(np.max(shifts) + np.min(shifts))
shifts = shifts - shiftOffset
if ARG.binning_factor is not None:
	shifts = shifts / ARG.binning_factor
if ARG.inverted_pixshifts:
	shifts = -shifts
print("H5 to %s min(angles)=%f max(angles)=%f min(shifts)=%f max(shifts)=%f"%(os.path.abspath(ARG.outputTickFile), np.min(angles), np.max(angles), np.min(shifts), np.max(shifts)))

ticks = np.zeros((2, len(angles)), dtype=np.float32)#Angle, pixel_shift
ticks[0,:] = angles
ticks[1,:] = shifts
DEN.storeNdarrayAsDEN(ARG.outputTickFile, ticks, force=ARG.force)
