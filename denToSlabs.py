#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
from libtiff import TIFF
from PIL import Image
from PIL.TiffTags import TAGS
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument('--first-index', type=int, default=None)
parser.add_argument('--last-index', type=int, default=None)
parser.add_argument('--slab-width', type=int, default=None)
parser.add_argument('--suffix', type=str, default="")


ARG = parser.parse_args()

#To create dataframe with given columns
def insertToDf(df, dat, name):
	time = dat["%s/time" % (name)]
	value = dat["%s/value" % (name)]
	for i in range(len(value)):
		t = time[i]
		v = value[i]
		df.loc[t][name] = v

def writeSlabs(ARG, images, outputDen, force=None):
	if force is None:
		force=False
	if ARG.verbose:
		print("Preparing file %s"%outputDen)
	row_count = images.shape[1]
	col_count = images.shape[1]
	angles_count = images.shape[0]
	if os.path.exists(outputDen) and not ARG.force:
		print("File %s exist, add --force to overwrite." % ARG.outputDen)
		return
	if ARG.slab_width is not None:
		for i in range(0, row_count, ARG.slab_width):
			fileName = slabFileName(ARG.outputDen, i, i+ARG.slab_width)
			DEN.storeNdarrayAsDEN(fileName, images[:,i:i+ARG.slab_width,:], force=force)
	else:
		DEN.storeNdarrayAsDEN(outputDen, images, force=force)

def slabFileName(origName, fromID, toID):
	f = origName.rsplit('.', 1)
	fileName = "%sfrom_%05d_to_%05d"%(f[0], fromID, toID);
	if len(f) == 2:
		fileName = "%s%s.%s"%(fileName, ARG.suffix, f[1])
	return fileName

den = DEN.getNumpyArray(ARG.inputDen)
if ARG.first_index is not None:
	den = den[ARG.first_index:ARG.last_index]
writeSlabs(ARG, den, ARG.outputDen, ARG.force)
