#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2026

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
import zarr
from multiprocessing.dummy import Pool, Lock  # threads
import multiprocessing as mp
from multiprocessing import Value
import traceback
from termcolor import colored
import shutil
from timeit import default_timer as timer
import json
import numcodecs
from numcodecs import Blosc, GZip, VLenUTF8
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("inputh5")
parser.add_argument("outputZarr")
parser.add_argument(
	"--raw-dir",
	default=None,
	type=str,
	help=
	"Provide raw directory where to find files, by default parrent directory of inputh5."
)
parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files.")
parser.add_argument("--force", action="store_true")
parser.add_argument('--compression', type=str,
					choices=['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
					default='blosc-zstd',
					help="Compression type (default: blosc-zstd).")
parser.add_argument('--clevel', type=int, default=5,
					help="Compression level (default: 5).")
parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
parser.add_argument("--zip", action="store_true", help="Use zip store for Zarr output instead of directory store.")
parser.add_argument("--verbose", action="store_true")

try:
	_out = io.StringIO()
	_err = io.StringIO()
	with redirect_stdout(_out):
		with redirect_stderr(_err):
			ARG = parser.parse_args(sys.argv[1:] or ['--help'])
except SystemExit as err:
	print("Program to produce Zarr file from raw TIFF structure. Usage:")
	#possibly parser.print_help()
	sys.stderr.write(_err.getvalue())
	sys.stdout.write(_out.getvalue())
	sys.exit(err.code)
#ARG = parser.parse_args()
#ARG = parser.parse_args(["/home/user/desy_example_data/syn0101_17L_Ti_12w_000_nexus.h5", "/tmp/desy", "--force"])

ARG.inputh5 = str(Path(ARG.inputh5).resolve(strict=True))
ARG.outputZarr = str(Path(ARG.outputZarr).resolve(strict=False))

#Set up threading
if ARG.j < 0:
	ARG.j = mp.cpu_count()
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
elif ARG.j == 0:
	print("No threading will be used ARG.j=0.")
else:
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, mp.cpu_count()))

_write_lock = None

def _init_writer_worker(lock):
	global _write_lock
	_write_lock = lock
#END threading setup

#To create dataframe with given columns
def insertToDf(df, dat, name):
	time = dat["%s/time" % (name)]
	value = dat["%s/value" % (name)]
	for i in range(len(value)):
		t = time[i]
		v = value[i]
		df.loc[t][name] = v

def get_compressor(name, clevel=5, zarrv3=False, outtype=None, endian="little"):
	"""
	Return a zarr-compatible compressor/codec based on name and Zarr format version.
	
	Parameters
	----------
	name : str
		Compression name (e.g., 'none', 'zstd', 'blosc-zstd', 'lz4', 'gzip', ...).
	clevel : int
		Compression level (meaning depends on the codec; Zstd/Blosc: 0..9 typical).
	zarrv3 : bool
		If True, return a Zarr v3 codec *pipeline* (list) suitable for `codecs=...`.
		If False, return a single compressor object (e.g., for Zarr v2 `compressor=`).
	outtype : Optional[Union[np.dtype, type, str]]
		Array dtype (e.g., np.uint16, 'uint16', np.dtype('uint16')). Used to set
		Blosc `typesize` (bytes per element). If None, defaults to itemsize=1.
	endian : str
		'little' or 'big' – used for v3 BytesCodec. Defaults to 'little'.
	"""
	# Derive element size in bytes; default to 1 if unknown
	if outtype is not None:
		try:
			itemsize = np.dtype(outtype).itemsize
		except Exception:
			itemsize = 1
	else:
		itemsize = 1

	if not zarrv3:
		from numcodecs import Blosc, GZip as NcGZip
		# Old style compressors (zarr v2 compatible)
		if name == 'none':
			return None
		elif name == 'zstd' or name == 'blosc-zstd':
			return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
		elif name == 'lz4' or name == 'blosc-lz4':
			return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
		elif name == 'gzip' or name == 'blosc-zlib':
			return GZip(level=clevel)
		elif name == 'blosc' or name == 'blosc-blosclz':
			return Blosc(cname='blosclz', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
		else:
			raise ValueError(f"Unknown compression type: {name}")
	else:
		# ---- Zarr v3 codecs (lazy import for safety) ----
		try:
			import zarr.codecs as codecs
		except ImportError:
			raise ImportError(
				"Zarr v3 codec system not available in this version of zarr. "
				"Please upgrade to zarr>=2.18.0."
			)
		# Map names to codecs
		codecs_chain = []
		if name == 'none':
			print("No compression selected for Zarr v3, returning empty codec chain.")
		elif name == "zstd":
			codecs_chain.append(codecs.ZstdCodec(level=clevel))
		elif name == "lz4":
			codecs_chain.append(codecs.LZ4Codec(level=clevel))
		elif name == "gzip":
			codecs_chain.append(codecs.GzipCodec(level=clevel))
		elif name == "blosc" or name == "blosc-blosclz":
			codecs_chain.append(
				codecs.BloscCodec(
					cname=codecs.BloscCname.blosclz,
					clevel=clevel,
					shuffle="shuffle",
					typesize=itemsize,
				)
			)
		elif name == "blosc-lz4":
				codecs_chain.append(
						codecs.BloscCodec(
						cname=codecs.BloscCname.lz4,
						clevel=clevel,
						shuffle="shuffle",
						typesize=itemsize,
						)
				)
		elif name == "blosc-lz4hc":
			codecs_chain.append(
					codecs.BloscCodec(
					cname=codecs.BloscCname.lz4hc,
					clevel=clevel,
					shuffle="shuffle",
					typesize=itemsize,
					)
			)
		elif name == "blosc-snappy":
			codecs_chain.append(
					codecs.BloscCodec(
					cname=codecs.BloscCname.snappy,
					clevel=clevel,
					shuffle="shuffle",
					typesize=itemsize,
					)
			)
		elif name == "blosc-zlib":
			codecs_chain.append(
					codecs.BloscCodec(
					cname=codecs.BloscCname.zlib,
					clevel=clevel,
					shuffle="shuffle",
					typesize=itemsize,
					)
			)
		elif name == "blosc-zstd":
			codecs_chain.append(
					codecs.BloscCodec(
					cname=codecs.BloscCname.zstd,
					clevel=clevel,
					shuffle="shuffle",
					typesize=itemsize,
					)
			)
		else:
			raise ValueError(f"Unknown compressor type '{name}' for Zarr v3")
		return codecs_chain


def tiffImageToArrayIndex(tiffFile, outArray, kIndex):
	"""Read a single TIFF image and write it to the specified index in the Zarr/numpy array."""
	# ---- Basic array checks ----
	if outArray is None or not hasattr(outArray, "shape") or not hasattr(outArray, "dtype"):
		raise TypeError("outArray must have 'shape' and 'dtype' attributes (NumPy/Zarr-like).")

	if len(outArray.shape) != 3:
		raise ValueError(f"outArray must be 3D (Z, Y, X); got shape={outArray.shape}")
	dimz, dimy, dimx = outArray.shape
	if not (0 <= kIndex < dimz):
		raise IndexError(f"kIndex {kIndex} out of bounds for array with shape {outArray.shape}")
	target_dtype = np.dtype(outArray.dtype)
	# ---- Read TIFF ----
	try:
		with Image.open(tiffFile) as im:
			img = np.array(im)
	except Exception as e:
		raise ValueError(f"Failed to read TIFF file '{tiffFile}': {e}")
	# ---- Check shape ----
	if img.shape != (dimy, dimx):
		raise ValueError(
			f"Shape mismatch in TIFF '{tiffFile}': got {img.shape}, expected {(dimy, dimx)}"
		)
	# ---- Cast to output dtype ALWAYS ----
	if img.dtype != target_dtype:
		try:
			img = img.astype(target_dtype, copy=False)
		except Exception as e:
			raise ValueError(
				f"Failed to convert image dtype from {img.dtype} to {target_dtype}: {e}"
			)
	outArray[kIndex, :, :] = img

def tiffImageToArrayIndex_worker(tiffFile, outArray, kIndex):
	"""Worker function for multiprocessing that wraps tiffImageToArrayIndex and captures exceptions."""
	try:
		tiffImageToArrayIndex(tiffFile, outArray, kIndex)
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": None}
	except Exception as e:
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": traceback.format_exc()}

progress = Value('i', 0)   # atomic int

def progress_callback(result):
	# result is what the worker returned
	with progress.get_lock():
		progress.value += 1
		count = progress.value
	if ARG.verbose and (count % 100 == 0 or result["error"] is not None):
		print(f"Written {count}/{result['n_img']} frames, current: {result['kIndex']+1}/{result['n_img']} "
			  f"({os.path.basename(result['tiffFile'])})")

#To write dataframe to 
def writeZarrArray(df, zarrArray, inputDir):
	progress = Value('i', 0)   # atomic int
	if df.empty:
		raise ValueError("Dataframe is empty, can not write to Zarr")
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
	inputTifFiles = [ os.path.join(inputDir, f.lstrip("/")) for f in inputTifFiles ]
	# Write images
	if ARG.j == 0:
		for i, f in enumerate(inputTifFiles):
			start = timer()
			tiffImageToArrayIndex(f, zarr_array, i)
			if ARG.verbose and (i % 100 == 0 or i == n_images - 1):
				print(f"Written frame {i+1}/{n_images} ({os.path.basename(f)}) in {timer()-start:.3f}s")
	else:
		results = []
		with Pool(ARG.j, initializer=_init_writer_worker, initargs=(Lock(),)) as pool:
			for i, f in enumerate(inputTifFiles):
				res = pool.apply_async(tiffImageToArrayIndex_worker, args=(f, zarrArray, i), callback=progress_callback)
				results.append(res)
			# Wait for all tasks to complete and check for exceptions
			pool.close()
			pool.join()
		errors = []
		for res in results:
			r = res.get()  # This will re-raise any exception from the worker
			if r["error"] is not None:
				errors.append((r["tiffFile"], r["kIndex"], r["error"]))
		if len(errors) > 0:
			print(colored(f"Encountered {len(errors)} errors during TIFF processing:", "red"))
			for tiffFile, kIndex, error in errors:
				print(colored(f"Error processing '{tiffFile}' at index {kIndex}:\n{error}", "red"))
			raise RuntimeError(f"{len(errors)} errors occurred during TIFF processing. See above for details.")
	if ARG.verbose:
		print(colored(f"Zarrarray written to {zarrArray.path} with shape {zarrArray.shape} and dtype {zarrArray.dtype}", "green"))


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


# Check if the Zarr file already exists and handle based on force flag
if os.path.exists(ARG.outputZarr):
	if ARG.force:
		shutil.rmtree(ARG.outputZarr)
	else:
		raise IOError(f"File {ARG.outputZarr} exists, use --force to overwrite")

df = PETRA.scanDataset(ARG.inputh5, includeCurrent=True)
experimentInfo = PETRA.getExperimentInfo(ARG.inputh5)
export = {}
#Test if df is not empty and determine numpy type of image_file TIFF and its dimensions
if df.empty:
	raise ValueError("Dataframe is empty, no data to write to Zarr file %s" % (os.path.realpath(ARG.outputZarr)))
else:
	tiff_file_str = df["image_file"].iloc[0]#In some versions this is string but in some bytes
	if isinstance(tiff_file_str, bytes):
		tiff_file_str = tiff_file_str.decode("utf-8")
	tiff_file_path = os.path.join(inputDir, tiff_file_str.lstrip("/"))
	if not os.path.exists(tiff_file_path):
		raise FileNotFoundError(f"TIFF file {tiff_file_path} does not exist, cannot determine image dimensions and dtype.")
	with Image.open(tiff_file_path) as img:
		img_array = np.array(img)
		dimy, dimx = img_array.shape
		dtype = img_array.dtype
		print(f"Determined image dimensions: {dimx}x{dimy}, dtype: {dtype}")

import numpy as np

def sanitize_for_json(obj):
	"""
	Recursively convert obj so it can be JSON-serialized:
	- numpy scalars -> Python scalars
	- numpy arrays	-> lists
	- dict/list/tuple -> recurse
	- leave Python scalars/None/str as-is
	"""
	# NumPy scalars -> Python scalars
	if isinstance(obj, (np.generic,)):	# covers np.float32, np.int64, np.bool_, etc.
		return obj.item()
	# NumPy arrays -> lists (be careful with very large arrays; attrs should stay small)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	# Mappings -> recurse
	if isinstance(obj, dict):
		return {str(k): sanitize_for_json(v) for k, v in obj.items()}
	# Sequences -> recurse element-wise
	if isinstance(obj, (list, tuple)):
		return [sanitize_for_json(x) for x in obj]
	# Everything else: assume already JSON-safe (str, int, float, bool, None)
	return obj



def copy_h5_arrays_into_group(
	h5,
	zarr_params_group: zarr.Group,
	compressor = None,
	prefer_src_chunks: bool = True,
	chunk_bytes: int = 64 * 1024 * 1024,  # ~64 MB streaming slabs
	verbose: bool = True,
) -> None:
	"""
	Copy all HDF5 N-D datasets from `h5_path` (under `src_group`) into the given Zarr group.
	Preserves shapes, dtypes, attributes and subgroup structure.
	"""
	attributes = dict(h5.attrs)
	for entry in list(h5):
		if verbose:
			print(f"Processing HDF5 entry: {h5.name}/{entry}  type={type(h5[entry])}")
		if isinstance(h5[entry], h5py.Dataset):
			# Store scalar datasets as 1D arrays to preserve dtype and attributes, even if they are shape=() in HDF5
			if h5[entry].shape == ():
				val = h5[entry][()]
				if isinstance(val, (bytes, np.bytes_)):
					val = val.decode("utf-8")
				if isinstance(val, np.generic):
					val = val.item()
				attributes[entry] = val
			elif h5[entry].dtype == np.dtype("O") or h5[entry].dtype.kind == 'S':
				# Handle object/string arrays by converting to UTF-8 strings in Zarr
				str_array = h5[entry][:]
				if str_array.flatten().size > 0:
					val = str_array.flatten()[0]
					if isinstance(val, (bytes, np.bytes_)):
						str_array = np.char.decode(str_array.astype('S'), 'utf-8', errors='replace')
				za = zarr_params_group.create_array(
					entry,
					shape=str_array.shape,
					dtype="string",
				)
				za[:] = str_array
			else:
				za = zarr_params_group.create_array(
					entry,
					shape=h5[entry].shape,
					dtype=h5[entry].dtype,
					compressors=compressor,
				)
				za[:] = h5[entry][:]
				if verbose:
					print(f"[dataset] {h5.name}/{entry}  shape={h5[entry].shape}	dtype={h5[entry].dtype}  -> zarr dtype={za.dtype}  chunks={za.chunks}")
		elif isinstance(h5[entry], h5py.Group):
			subgroup = zarr_params_group.require_group(entry)
			copy_h5_arrays_into_group(h5[entry], subgroup)
		else:
			raise ValueError(f"HDF5 entry '{entry}' is neither a group nor a dataset, cannot copy.")
	#Update group attributes at the end to avoid issues with copying datasets that have attributes with the same name as group attributes
	for k, v in attributes.items():
		if isinstance(v, (bytes, np.bytes_)):
			v = v.decode("utf-8")
		if isinstance(v, np.generic):
			v = v.item()
		zarr_params_group.attrs[k] = v
	if verbose:
		print("Finished copying HDF5 arrays into Zarr group:", zarr_params_group.path)


dark = df.loc[df["image_key"] == 2]
white = df.loc[df["image_key"] == 1]
scan = df.loc[df["image_key"] == 0]

df_json = json.loads(df.to_json(orient="split", date_format="iso"))

chunk_shape = (1, dimy, dimx)  # Chunking by individual images
outtype = dtype  # Keep the same dtype as the original TIFF images
dar_count = len(dark)
ref_count = len(white)
img_count = len(scan)

export["dimx"] = dimx
export["dimy"] = dimy
export["dtype"] = str(dtype)
export["img_count"] = img_count
export["ref_count"] = ref_count
export["dar_count"] = dar_count
export["h5_path"] = ARG.inputh5
export["output_zarr_path"] = ARG.outputZarr
export["compression"] = {}
export["compression"]["name"] = ARG.compression
export["compression"]["clevel"] = ARG.clevel

experimentInfo["export"] = export

experimentInfo_sanitized = sanitize_for_json(experimentInfo)

if ARG.zip or ARG.outputZarr.endswith(".zip"):
	store = zarr.storage.ZipStore(ARG.outputZarr, mode='w')
else:
	store = ARG.outputZarr

zarr_top_level = zarr.open_group(
			store=store,
			mode='w',
			attributes=experimentInfo_sanitized
		)

codec = get_compressor(ARG.compression, ARG.clevel, zarrv3=True, outtype=dtype)
zarr_array_ref = zarr_top_level.create_array('ref', shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
zarr_array_dar = zarr_top_level.create_array('dar', shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
zarr_array_img = zarr_top_level.create_array('img', shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)

zarr_params = zarr_top_level.create_group(name="params", attributes=df_json)
h5 = h5py.File(ARG.inputh5, "r")
if "entry" in h5:
	h5 = h5["entry"]
copy_h5_arrays_into_group(h5, zarr_params, verbose=ARG.verbose)

writeZarrArray(dark, zarr_array_dar, inputDir)
writeZarrArray(white, zarr_array_ref, inputDir)

if ARG.fix_corrupted_h5 and scan.empty:
	print("HDF5 file is corrupted. Scanning directory for TIFF files to create img.den...")
	dark_files = set(dark["image_file"])
	white_files = set(white["image_file"])
	exclude_files = dark_files.union(white_files)
	scan = scanForTiffFiles(inputDir, exclude_files)

writeZarrArray(scan, zarr_array_img, inputDir)
