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
import time
import argparse
from multiprocessing.dummy import Pool  # threads
try:
	import tifffile  # about twice as fast as PIL and releases the GIL, so the reading threads scale
except ImportError:
	tifffile = None
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
parser.add_argument("-j", "--threads", default=-1, type=int, dest="j",
	help="Threads reading the TIFF files; the DEN files are written by one thread in frame order. "
		 "[defaults to -1: a quarter of the CPUs available to the process (in a SLURM job its allocation), at least 16 and at most 32, but not more than available; 0 without threading]")
parser.add_argument("--verbose", action="store_true", help="Print every written frame.")

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

start_time = time.time()
_phase_time = [start_time]


def report_phase(name):
	"""Print the time spent since the previous phase, to see where a conversion spends its time."""
	now = time.time()
	print(f"TIME {name}: {now - _phase_time[0]:.1f}s", flush=True)
	_phase_time[0] = now


# Default thread count as rawTiffStructureToZarr.py: a quarter of the CPUs this process may run on
# (a SLURM job's allocation, not the whole node), at least 16 and at most 32; the reading threads
# feed one writer.
try:
	AVAILABLE_CPUS = len(os.sched_getaffinity(0))
except AttributeError:  # not on Linux
	AVAILABLE_CPUS = os.cpu_count() or 1
if ARG.j < 0:
	ARG.j = min(AVAILABLE_CPUS, max(16, min(AVAILABLE_CPUS // 4, 32)))
print("Reading TIFF files with %s, %s (%d CPUs available)" % ("tifffile" if tifffile is not None else "PIL",
	  "%d threads" % ARG.j if ARG.j > 0 else "no threads", AVAILABLE_CPUS))


def read_tiff(path):
	"""The image of a TIFF file as a numpy array (tifffile if available, else PIL)."""
	if tifffile is not None:
		return tifffile.imread(path)
	with Image.open(path) as im:
		return np.array(im)


# Sizes of the input TIFF files and the written DEN files, for the size report
SIZES = {}


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
		#Check it is sorted by time and raise error if not
		if not df["time"].is_monotonic_increasing:
			raise ValueError("Dataframe is not sorted by time, can not write to Zarr")
	#Check image_file column exists and dataset is sorted by image_file
	if "image_file" not in df.columns:
		raise ValueError("Dataframe does not contain image_file column, can not write to Zarr")
	if not df["image_file"].is_monotonic_increasing:
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")
	inputTifFiles = [ x.decode("utf-8") if isinstance(x, bytes) else x for x in df["image_file"] ]
	tifFilesBasename = [ os.path.basename(f) for f in inputTifFiles ]
	if not all(tifFilesBasename[i] <= tifFilesBasename[i+1] for i in range(len(tifFilesBasename)-1)):
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")
	else:
		print(tifFilesBasename[0], tifFilesBasename[-1])
	# In some versions image_file is string but in some bytes
	files = [os.path.join(inputDir, (x.decode("utf-8") if isinstance(x, bytes) else x).lstrip("/")) for x in df["image_file"]]
	img = read_tiff(files[0])
	dimy = img.shape[0]
	dimx = img.shape[1]
	dimz = len(df)
	print("Creating %s %dx%dx%d" % (denFile, dimx, dimy, dimz))
	DEN.writeEmptyDEN(denFile, [dimx, dimy, dimz], force=True)

	def read(path):
		return np.asarray(read_tiff(path), dtype=np.float32)
	# The frames are read in parallel and written in order by this thread (imap keeps the order)
	pool = Pool(ARG.j) if ARG.j > 0 else None
	frames = pool.imap(read, files, chunksize=1) if pool is not None else map(read, files)
	try:
		for i, img in enumerate(frames):
			if img.shape != (dimy, dimx):
				raise ValueError("Shape mismatch in %s: got %s, expected %s" % (files[i], img.shape, (dimy, dimx)))
			if ARG.verbose or i % 100 == 0 or i == dimz - 1:
				print("Writing %d-th file %s of shape %d,%d into %s" %
					  (i, os.path.basename(files[i]), img.shape[0], img.shape[1], denFile))
			DEN.writeFrame(denFile, i, img, force=True)
	finally:
		if pool is not None:
			pool.terminate()
	SIZES[os.path.basename(denFile)] = (sum(os.path.getsize(f) for f in files), os.path.getsize(denFile))

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

report_phase("HDF5 scan table")
writeDenFile(dark, inputDir, os.path.join(ARG.outputDir, "dar.den"), ARG.force)
report_phase(f"dar.den ({len(dark)} frames)")
writeDenFile(white, inputDir, os.path.join(ARG.outputDir, "ref.den"), ARG.force)
report_phase(f"ref.den ({len(white)} frames)")

if ARG.fix_corrupted_h5 and scan.empty:
	print("HDF5 file is corrupted. Scanning directory for TIFF files to create img.den...")
	dark_files = set(dark["image_file"])
	white_files = set(white["image_file"])
	exclude_files = dark_files.union(white_files)
	scan = scanForTiffFiles(inputDir, exclude_files)

writeDenFile(scan, inputDir, os.path.join(ARG.outputDir, "img.den"), ARG.force)
report_phase(f"img.den ({len(scan)} frames)")

# DEN files are float32, so twice the size of 16-bit TIFF files
for name, (tiff, den) in SIZES.items():
	print(f"SIZE   {name}: TIFF {tiff / 1024**3:.3f} GiB -> DEN {den / 1024**3:.3f} GiB ({100 * den / tiff:.1f}% of TIFF)")
tiff, den = sum(t for t, _ in SIZES.values()), sum(d for _, d in SIZES.values())
print(f"SIZE   total: TIFF {tiff / 1024**3:.3f} GiB -> DEN {den / 1024**3:.3f} GiB ({100 * den / tiff:.1f}% of TIFF)")
elapsed = int(time.time() - start_time)
print("END rawTiffStructureToDen.py in %02d:%02d:%02ds" % (elapsed // 3600, elapsed % 3600 // 60, elapsed % 60))
