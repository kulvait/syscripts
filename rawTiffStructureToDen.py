#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
#from libtiff import TIFF
#pd.set_option('display.max_columns', 100) to display untruncated columns
from PIL import Image
from PIL.TiffTags import TAGS
import numpy as np
import sys
import os
import argparse
from denpy import DEN
from denpy import PETRA
import io
from contextlib import redirect_stdout
from contextlib import redirect_stderr

parser = argparse.ArgumentParser()
parser.add_argument("inputh5")
parser.add_argument("outputDir")
parser.add_argument(
	"--raw-dir",
	default=None,
	type=str,
	help=
	"Provide raw directory where to find files, by default parrent directory of inputh5."
)
parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files.")
parser.add_argument("--force", action="store_true")

try:
	_out = io.StringIO()
	_err = io.StringIO()
	with redirect_stdout(_out):
		with redirect_stderr(_err):
			ARG = parser.parse_args(sys.argv[1:] or ['--help'])
except SystemExit as err:
	print("Program to produce dar.den ref.den and img.den files.")
	#possibly parser.print_help()
	sys.stderr.write(_err.getvalue())
	sys.stdout.write(_out.getvalue())
	sys.exit(err.code)
#ARG = parser.parse_args()
#ARG = parser.parse_args(["/home/user/desy_example_data/syn0101_17L_Ti_12w_000_nexus.h5", "/tmp/desy", "--force"])


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
	if df.empty:
		raise ValueError("Dataframe is empty, can not write file %s" % (os.path.realpath(denFile)))
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	if "time" in df.columns:
		df = df.sort_values("time", ascending=True)
	fileStr = df["image_file"].iloc[0]#In some versions this is string but in some bytes
	if isinstance(fileStr, bytes):
		fileStr = fileStr.decode("utf-8")
	file = os.path.join(inputDir, fileStr.lstrip("/"))
	im0 = Image.open(file)
	img = np.array(im0)
	dimy = img.shape[0]
	dimx = img.shape[1]
	dimz = len(df)
	dtype = img.dtype
	#	 img=TIFF.open(file)
	#	 img = img.read_image()
	#	 dimy=img.shape[0]
	#	 dimx=img.shape[1]
	print("Creating %s %dx%dx%d" % (denFile, dimx, dimy, dimz))
	DEN.writeEmptyDEN(denFile, [dimx, dimy, dimz], force=True)
	for i in range(len(df)):
		fileStr = df["image_file"].iloc[i]#In some versions this is string but in some bytes
		if isinstance(fileStr, bytes):
			fileStr = fileStr.decode("utf-8")
		img = np.array(Image.open(os.path.join(inputDir, fileStr.lstrip("/"))),
					   dtype=np.float32)
		print("Writing %d-th file %s of shape %d,%d into %s" %
			  (i, fileStr, img.shape[0], img.shape[1], denFile))
		DEN.writeFrame(denFile, i, img, force=True)

# Function to scan directory for TIFF files and create a DataFrame
def scanForTiffFiles(directory, exclude_files):
	tiff_files = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
				filepath = os.path.join(root, file)
				relative_path = os.path.relpath(filepath, directory)
				if relative_path not in exclude_files:
					tiff_files.append(relative_path)
	tiff_files.sort()
	return pd.DataFrame({'image_file': tiff_files})

#Parse input data
if ARG.raw_dir is not None:
	inputDir = ARG.raw_dir
else:
	inputDir = os.path.dirname(os.path.realpath(ARG.inputh5))

if not os.path.exists(ARG.outputDir):
	print("Create dir %s" % (ARG.outputDir))
	os.mkdir(ARG.outputDir)

df = PETRA.scanDataset(ARG.inputh5)

dark = df.loc[df["image_key"] == 2]
white = df.loc[df["image_key"] == 1]
scan = df.loc[df["image_key"] == 0]

writeDenFile(dark, inputDir, os.path.join(ARG.outputDir, "dar.den"), ARG.force)
writeDenFile(white, inputDir, os.path.join(ARG.outputDir, "ref.den"), ARG.force)

if ARG.fix_corrupted_h5 and scan.empty:
	print("HDF5 file is corrupted. Scanning directory for TIFF files to create img.den...")
	dark_files = set(dark["image_file"])
	white_files = set(white["image_file"])
	exclude_files = dark_files.union(white_files)
	scan = scanForTiffFiles(inputDir, exclude_files)

writeDenFile(scan, inputDir, os.path.join(ARG.outputDir, "img.den"), ARG.force)
