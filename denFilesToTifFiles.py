#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
from PIL import Image
import sys
import os
import argparse
from denpy import DEN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDir")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument('--suffix', type=str, default="")

ARG = parser.parse_args()


def writeSlice(img, fileName, force=None):
	if force is None:
		force = False
	if os.path.exists(fileName) and not force:
		raise IOError("File %s already exist!" % fileName)
	im = Image.fromarray(img, mode='F')  # float32
	im.save(fileName, "TIFF")


header = DEN.readHeader(ARG.inputDen)
if header["dimcount"] < 2:
	print(
	    "File %s is %d-dimensional, too few dimensions to convert it to TIFF.")
	sys.exit(1)
xdim = header["dimspec"][0]
ydim = header["dimspec"][1]
if not os.path.exists(ARG.outputDir):
	os.makedirs(ARG.outputDir)

if header["dimcount"] == 2:
	writeSlice(DEN.getFrame(ARG.inputDen, 0),
	           "%s/%s.tif" % (ARG.outputDir, ARG.suffix), ARG.force)
else:
	frameCount = np.prod(header["dimspec"][2:])
	significantDigits = int(np.log10(frameCount))+1
	if significantDigits <= 3:
		formatString = "%03d"
	else:
		formatString = "%%0%dd"%(significantDigits)
	for index in np.ndindex(header["dimspec"][2:]):
		if len(index) == 1:
			index = index[0]
			writeSlice(
			    DEN.getFrame(ARG.inputDen, index),
			    "%s/%s%s.tif" % (ARG.outputDir, formatString % (index), ARG.suffix),
			    ARG.force)
		else:
			writeSlice(
			    DEN.getFrame(ARG.inputDen, index),
			    "%s/%s%s.tif" % (ARG.outputDir, "_".join(
			        formatString % (e) for e in index), ARG.suffix), ARG.force)
