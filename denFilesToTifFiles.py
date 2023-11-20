#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
from PIL import Image
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDir")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument('--suffix', type=str, default="")

#ARG = parser.parse_args([
#    "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03/",
#    "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_trans03.prj50", "--slab-width", "100",
#    "--verbose"#, "--store-sinogram", "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_trans03.sin"
#	,"--slab-width", "50"
#])

ARG = parser.parse_args()

def writeSlice(img, fileName, force=None):
	if force is None:
		force=False
	if os.path.exists(fileName) and not force:
		raise IOError("File %s already exist!"%fileName)
	im = Image.fromarray(img, mode='F')  # float32
	im.save(fileName, "TIFF")

header = DEN.readHeader(ARG.inputDen)
if header["dimcount"] < 2:
	print("File %s is %d-dimensional, too few dimensions to convert it to TIFF.")
	sys.exit(1)
xdim = header["dimspec"][0]
ydim = header["dimspec"][1]
if not os.path.exists(ARG.outputDir):
	os.makedirs(ARG.outputDir)

if header["dimcount"] == 2:
	writeSlice(DEN.getFrame(ARG.inputDen, 0), "%s/%s.tif"%(ARG.outputDir, ARG.suffix), ARG.force)
else:
	for index in np.ndindex(header["dimspec"][2:]):
		if len(index)==1:
			index = index[0]
			writeSlice(DEN.getFrame(ARG.inputDen, index), "%s/%s%s.tif"%(ARG.outputDir,"%03d"%(index), ARG.suffix), ARG.force)
		else:
			writeSlice(DEN.getFrame(ARG.inputDen, index), "%s/%s%s.tif"%(ARG.outputDir,"_".join("%03d"%(e) for e in index), ARG.suffix), ARG.force)
