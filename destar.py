#!/usr/bin/env python
"""
2023-2025

@author: Vojtech Kulvait
@license: GNU GPL v3 or later

This script removes transient bright pixels (e.g. hot pixels or cosmic ray hits) 
from a 3D image stack (time or depth series) stored in the DEN format. It works 
by applying an iterative median filter to each frame, identifying outlier pixels 
that deviate from their local neighborhood by more than a specified number of 
standard deviations, and replacing them with the corresponding median value.

Key features:
- Configurable neighborhood size, threshold, and number of iterations.
- Per-frame processing using multiprocessing for performance.
- Optional verbose output showing the percentage of corrected pixels per frame.
- At the end, prints total number and percentage of pixels corrected across the dataset.

This method is useful when a dark frame is insufficient to remove non-repeatable 
artifacts such as cosmic rays or temporally unstable hot pixels.

For alternative method, which filter just bright spots based on average of nearby pixles, see Algotom remove_zinger method in https://algotom.readthedocs.io/en/latest/toc/api/algotom.prep.removal.html#algotom.prep.removal.check_zinger_size
Note that Algotom method works probably better on sinogram than projection frames.
"""

import argparse
from denpy import DEN 
import zarr
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import convolve
from multiprocessing.dummy import Process, Lock, Pool
import multiprocessing
import time
import traceback
import warnings


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("inputFile", help="File to deconvolve with the kernel slice at once, DEN or Zarr format, in Zarr use zarr:/array/path syntax")
parser.add_argument("outputFile", help="Convolution output file, DEN or Zarr format, in Zarr use zarr:/array/path syntax")
parser.add_argument("--output-mask", help="Output file for the binary mask of detected outliers, DEN or Zarr format, in Zarr use zarr:/array/path syntax", default=None)
parser.add_argument("--filter-size", type=int, default=5, help="Size parameter to the scipy.ndimage.median_filter or Zinger kernel size.")
parser.add_argument("--filter-threshold", type=float, default=3.0, help="Number of standard deviations to substitute data.")
parser.add_argument("--iterations", type=int, default=1, help="Number of iterations.")
parser.add_argument("--zinger-algorithm", help="Use Zinger correction algorithm described in https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849", action="store_true")
parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
ARG = parser.parse_args()

inputIsZarr = False

if ":" in ARG.inputFile:
	zarTokens = ARG.inputFile.split(":", 1)
	zarStorePath = zarTokens[0]
	zarPath = zarTokens[1]
	if zarPath == "/":
		zarPath = ""
	if zarStorePath.endswith(".zip"):
		zarInputStore = zarr.storage.ZipStore(zarStorePath)
	else:
		zarInputStore = zarr.storage.LocalStore(zarStorePath)
	print(f"Opening Zarr array from store '{zarStorePath}' with path '{zarPath}'")
	inputArray = zarr.open_array(zarInputStore, mode="r", path=zarPath)
	inputIsZarr = True
	if len(inputArray.shape) != 3:
		raise ValueError("Input Zarr array must be 3D, but got shape %s"%(inputArray.shape,))
	zdim, ydim, xdim = inputArray.shape
	dimspec = (xdim, ydim, zdim)
else:
	header = DEN.readHeader(ARG.inputFile)
	dimspec = header["dimspec"]
	xdim = np.uint64(dimspec[0])
	ydim = np.uint64(dimspec[1])
	zdim = np.uint64(dimspec[2])

frameSize = xdim * ydim
totalSize = frameSize * zdim

print(f"Starting processing file '{ARG.inputFile}' containing {zdim} frames of size {xdim}x{ydim} to produce '{ARG.outputFile}'")
print(f"Filter size: {ARG.filter_size}, Threshold: {ARG.filter_threshold}, Iterations: {ARG.iterations}")

if ARG.j < 0:
	ARG.j = multiprocessing.cpu_count()
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
elif ARG.j == 0:
	print("No threading will be used ARG.j=0.")
else:
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))


DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True)
if ARG.output_mask is not None:
	DEN.writeEmptyDEN(ARG.output_mask, dimspec, force=True, elementtype=np.dtype('<u1'))

# Global write_lock for workers
write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l 


def delete_large_components(mask: np.ndarray, max_size: int) -> np.ndarray:
	"""
	Remove connected components larger than `max_size` pixels.

	Parameters
	----------
	mask : np.ndarray (bool)
		Binary mask where True indicates detected outliers.
	max_size : int
		Maximum allowed size (in pixels) of connected components to keep.

	Returns
	-------
	np.ndarray (bool)
		Mask with only small connected components retained.
	"""
	if mask.dtype != bool:
		mask = mask.astype(bool)
	labels, num = ndi.label(mask)
	if num == 0:
		return mask  # already boolean
	# Compute size of each labeled component (labels start at 1)
	sizes = ndi.sum(mask, labels, index=np.arange(1, num + 1))
	# Identify labels to keep (small enough components)
	keep_labels = np.where(sizes <= max_size)[0] + 1
	# Build filtered mask
	filtered_mask = np.isin(labels, keep_labels)
	return filtered_mask

def remove_hot_pixels(frame, iterations, filter_size, correct_threshold_abs_sigma=3.0, correct_threshold_abs=None, correct_threshold_rel_sigma=None, correct_threshold_rel=None, zinger_algorithm=False, filter_large_components=False, large_component_minpixcount=10, epsilon=1e-6):
	assert any([
    correct_threshold_abs_sigma is not None,
    correct_threshold_abs is not None,
    correct_threshold_rel_sigma is not None,
    correct_threshold_rel is not None
]), "At least one threshold must be set"
	xi = frame.astype(np.float32)
	if zinger_algorithm:
		# Zinger algorithm described in https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849
		if filter_size % 2 == 0:
			warnings.warn("Zinger algorithm expects odd filter size, but got %d. Effective filter size will be %d."%(filter_size, filter_size+1))
			filter_size += 1
		kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
		size = filter_size // 2
		offsets = [ 
			(-size, -size), (-size, 0), (-size, size),
			(0, -size),					(0, size),
			(size, -size),	(size, 0), (size, size),
		]
		for di, dj in offsets:
			kernel[size + di, size + dj] = 1.0 
		kernel /= kernel.sum()
	mask_corrupted_pixel = np.zeros_like(xi, dtype=bool)
	for _ in range(iterations):
		if zinger_algorithm:
			frame_filtered = convolve(xi, kernel, mode="reflect")
		else:
			frame_filtered = median_filter(xi, size=filter_size, mode="reflect")
		frame_dif = xi - frame_filtered
		frame_filtered_sign = np.sign(frame_filtered)
		frame_filtered_sign[frame_filtered_sign == 0] = 1
		frame_dif_rel = frame_dif / (frame_filtered + epsilon * frame_filtered_sign)
		flt = np.zeros_like(xi, dtype=bool)
		if correct_threshold_abs_sigma is not None:
			frame_dif_std = np.std(frame_dif)
			flt |= np.abs(frame_dif) > correct_threshold_abs_sigma * frame_dif_std
		if correct_threshold_abs is not None:
			flt |= np.abs(frame_dif) > correct_threshold_abs
		if correct_threshold_rel_sigma is not None:
			frame_dif_rel_std = np.std(frame_dif_rel)
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel_sigma * frame_dif_rel_std
		if correct_threshold_rel is not None:
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel
		xi[flt] = frame_filtered[flt]
		mask_corrupted_pixel |= flt
	if filter_large_components:
		mask_corrupted_pixel = delete_large_components(mask_corrupted_pixel, large_component_minpixcount)
		xi[mask_corrupted_pixel == False] = frame[mask_corrupted_pixel == False]
	corrected_pixel_count = int(mask_corrupted_pixel.sum())
	return xi, mask_corrupted_pixel, corrected_pixel_count

# Lock is protected by finally clause to avoid deadlocks
def processFrame(ARG, k):
	try:
		# Read (no lock unless DEN requires one)
		if inputIsZarr:
			f = inputArray[k]
		else:
			f = DEN.getFrame(ARG.inputFile, k)
		f_filtered, f_corrupted_pixel, corrected_pixels = remove_hot_pixels(f, iterations=ARG.iterations, filter_size=ARG.filter_size, correct_threshold_abs_sigma=ARG.filter_threshold, zinger_algorithm=ARG.zinger_algorithm)
		# Write (locked)
		if write_lock:
			write_lock.acquire()
		try:
			DEN.writeFrame(ARG.outputFile, k, f_filtered, force=True)
			if ARG.output_mask is not None:
				DEN.writeFrame(ARG.output_mask, k, f_corrupted_pixel.astype(np.uint8), force=True)
		finally:
			if write_lock:
				write_lock.release()
		return {"k": k, "pixels": corrected_pixels, "error": None}
	except Exception:
		return {"k": k, "pixels": 0, "error": traceback.format_exc()}

class FakeAsyncResult:
	def __init__(self, value):
		self._value = value
	def get(self):
		return self._value

results = []
if ARG.j == 0:
	for k in range(zdim):
		res = processFrame(ARG, k)
		results.append(FakeAsyncResult(res))
else:
	lock = Lock()
	tp = Pool(processes=ARG.j, initializer=init_worker, initargs=(lock,))
	for k in range(zdim):
		res = tp.apply_async(processFrame, args=(ARG, k))
		results.append(res)
	tp.close()
	tp.join()

errors = []
total_pixels_corrected = 0
total_frames_sucessful = 0
for result in results:
	r = result.get()
	k = r["k"]
	if r["error"] is not None:
		errors.append((r["k"], r["error"]))
	else:
		total_pixels_corrected += r["pixels"]
		total_frames_sucessful += 1
		if ARG.verbose:
			corrected_fraction = r["pixels"] / frameSize
			print("Frame %d: %d pixels corrected, fraction: %.2f%%"%(r["k"], r["pixels"], corrected_fraction*100))

if len(errors) > 0:
	print("The following frames raised exceptions:")
	for (k, error) in errors:
		print(f"Frame {k} exception:\n{error}")
	print(f"{len(errors)} frames raised exceptions.")
	if total_frames_sucessful > 0:
		total_pixels_fraction = total_pixels_corrected / (total_frames_sucessful*frameSize)
		print("From total %d frames corrected in '%s' with %d pixels corrected, fraction: %.2f%%"%(total_frames_sucessful, ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
	else:
		total_pixels_fraction = 0.0
		print("Processing failed for all frames, no pixels corrected in '%s'."%(ARG.outputFile))
else:
	total_pixels_fraction = total_pixels_corrected / totalSize
	print("Sucessfully created '%s' with %d pixels corrected, fraction: %.2f%%"%(ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))

print("END destar.py")
