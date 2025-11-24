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
import numpy as np
from scipy.ndimage import median_filter
from multiprocessing.dummy import Process, Lock, Pool
import multiprocessing
import time
import traceback

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("inputFile", help="File to deconvolve with the kernel slice at once")
parser.add_argument("outputFile", help="Convolution output")
parser.add_argument("--filter-size", type=int, default=3, help="Size parameter to the scipy.ndimage.median_filter")
parser.add_argument("--filter-threshold", type=float, default=3.0, help="Number of standard deviations to substitute data.")
parser.add_argument("--iterations", type=int, default=3, help="Number of iterations.")
parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
ARG = parser.parse_args()


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

# Global write_lock for workers
write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l 

# Lock is protected by finally clause to avoid deadlocks
def processFrame(ARG, k):
	try:
		# Read (no lock unless DEN requires one)
		f = DEN.getFrame(ARG.inputFile, k)
		xi = np.array(f)
		# filtering loop
		pixel_corrected_mask = np.zeros_like(xi, dtype=bool)
		for i in range(ARG.iterations):
			mf = median_filter(xi, size=ARG.filter_size)
			dif = xi - mf
			std = np.std(dif)
			lim = ARG.filter_threshold * std
			flt = np.abs(dif) > lim
			xi[flt] = mf[flt]
			pixel_corrected_mask |= flt
		corrected_pixels = int(pixel_corrected_mask.sum())
		# Write (locked)
		if write_lock:
			write_lock.acquire()
		try:
			DEN.writeFrame(ARG.outputFile, k, xi, force=True)
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
	total_pixels_fraction = total_pixels_corrected / (total_frames_sucessful*frameSize)
	print("From total %d frames corrected in '%s' with %d pixels corrected, fraction: %.2f%%"%(total_frames_sucessful, ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
else:
	total_pixels_fraction = total_pixels_corrected / totalSize
	print("Sucessfully created '%s' with %d pixels corrected, fraction: %.2f%%"%(ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))

print("END destar.py")
