#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
from libtiff import TIFF
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inputDir")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--neglog", action="store_true")
parser.add_argument("--store-sinogram", type=str, default=None)
parser.add_argument('--first-index', type=int, default=None)
parser.add_argument('--last-index', type=int, default=None)
parser.add_argument('--slab-width', type=int, default=None)

ARG = parser.parse_args([
    "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03/",
    "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_trans03.prj50",
    "--slab-width",
    "100",
    "--verbose"  #, "--store-sinogram", "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_trans03.sin"
    ,
    "--slab-width",
    "50"
])

ARG = parser.parse_args()


#To create dataframe with given columns
def insertToDf(df, dat, name):
	time = dat["%s/time" % (name)]
	value = dat["%s/value" % (name)]
	for i in range(len(value)):
		t = time[i]
		v = value[i]
		df.loc[t][name] = v


#To write dataframe to den
def writeDenFile(df, inputDir, denFile, force=False):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	df = df.sort_values("time", ascending=True)
	fileStr = df["image_file"].iloc[
	    0]  #In some versions this is string but in some bytes
	if isinstance(fileStr, bytes):
		fileStr = fileStr.decode("utf-8")
	file = os.path.join(inputDir, fileStr)
	img = TIFF.open(file)
	img = img.read_image()
	dimy = img.shape[0]
	dimx = img.shape[1]
	dimz = len(df)
	print("Creating %s %dx%dx%d" % (denFile, dimx, dimy, dimz))
	DEN.writeEmptyDEN(denFile, dimx, dimy, dimz, True)
	for i in range(len(df)):
		fileStr = df["image_file"].iloc[
		    i]  #In some versions this is string but in some bytes
		if isinstance(fileStr, bytes):
			fileStr = fileStr.decode("utf-8")
		img = TIFF.open(os.path.join(inputDir, fileStr))
		img = img.read_image()
		print("Writing %d-th file %s of shape %d,%d into %s" %
		      (i, fileStr, img.shape[0], img.shape[1], denFile))
		DEN.writeFrame(denFile, i, img, True)


def writeSlab(ARG, tifFiles, outputDen, force=None, outputSinogram=None):
	if force is None:
		force = False
	if ARG.verbose:
		print("Preparing file %s" % outputDen)
	tif = TIFF.open(tifFiles[0])
	sin = tif.read_image()
	angles_count = sin.shape[0]
	col_count = sin.shape[1]
	row_count = len(tifFiles)
	if ARG.verbose:
		print(
		    "The file %s has dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."
		    % (tifFiles[0], sin.shape[0], sin.shape[1], sin.dtype, sin.min(),
		       sin.max(), sin.mean()))
	sinograms = np.zeros(shape=(0, angles_count, col_count), dtype=np.float32)
	for i in range(len(tifFiles)):
		f = tifFiles[i]
		img = TIFF.open(f)
		img = img.read_image()
		img.shape = (1, angles_count, col_count)
		sinograms = np.concatenate((sinograms, img))
		if ARG.verbose and i % 10 == 0:
			print("Read file %d of %d" % (i + 1, len(tifFiles)))
	if os.path.exists(outputDen) and not ARG.force:
		print("File %s exist, add --force to overwrite." % ARG.outputDEN)
		return
	if ARG.store_sinogram is not None:
		if os.path.exists(outputSinogram) and not force:
			print("File %s exist, add --force to overwrite." % outputSinogram)
		else:
			DEN.storeNdarrayAsDEN(outputSinogram, sinograms, force)
	sinograms = np.transpose(sinograms,
	                         (1, 0, 2))  #Transpose to standard projections
	if ARG.neglog:
		sinograms = np.log(np.reciprocal(sinograms))
	DEN.storeNdarrayAsDEN(outputDen, sinograms, force)


def slabFileName(origName, fromID, toID):
	f = origName.rsplit('.', 1)
	fileName = "%sfrom_%05d_to_%05d" % (f[0], i, i + ARG.slab_width)
	if len(f) == 2:
		fileName = "%s.%s" % (fileName, f[1])
	return fileName


if ARG.verbose:
	print("Start of the script")
tifFiles = glob.glob(os.path.join(ARG.inputDir, "*.tif"))
tifFiles.sort()
if ARG.first_index is not None:
	tifFiles = tifFiles[ARG.first_index:ARG.last_index]
if ARG.slab_width is not None:
	for i in range(0, len(tifFiles), ARG.slab_width):
		fileName = slabFileName(ARG.outputDen, i, i + ARG.slab_width)
		if ARG.store_sinogram is not None:
			sinogramName = slabFileName(ARG.store_sinogram, i,
			                            i + ARG.slab_width)
		else:
			sinogramName = None
		writeSlab(ARG, tifFiles[i:i + ARG.slab_width], fileName, ARG.force,
		          sinogramName)
else:
	writeSlab(ARG, tifFiles, ARG.outputDen, ARG.force, ARG.store_sinogram)
