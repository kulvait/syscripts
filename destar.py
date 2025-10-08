#!/usr/bin/env python
"""
Created on Wed May  9 12:59:30 2023

@author: Vojtech Kulvait

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("inputFile", help="File to deconvolve with the kernel slice at once")
parser.add_argument("outputFile", help="Convolution output")
parser.add_argument("--filter-size", type=int, default=3, help="Size parameter to the scipy.ndimage.median_filter")
parser.add_argument("--filter-threashold", type=float, default=3.0, help="Number of standard deviations to substitute data.")
parser.add_argument("--iterations", type=int, default=3, help="Number of iterations.")
parser.add_argument("-j", type=int, default=-1, help="Number of threads, 0 for no threading -1 for automatic estimate.")
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
print(f"Filter size: {ARG.filter_size}, Threshold: {ARG.filter_threashold}, Iterations: {ARG.iterations}")

DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True)

# The function to median filter individual frames, returns the number of pixels corrected.
def processFrame(ARG, k, lck=None):
	if lck is not None:
		lck.acquire()
	f = DEN.getFrame(ARG.inputFile, k)
	if lck is not None:
		lck.release()
	xi = np.array(f)
	pixel_corrected_mask =  np.zeros_like(xi, dtype=bool)
	for i in range(ARG.iterations):
		mf = median_filter(xi, size=ARG.filter_size)
		dif = xi - mf
		std = np.std(dif)
		lim = ARG.filter_threashold * std
		flt = np.abs(dif) > lim
		xi[flt] = mf[flt]
		pixel_corrected_mask |= flt
	corrected_pixels = np.sum(pixel_corrected_mask)
	corrected_fraction = corrected_pixels / frameSize
	if lck is not None:
		lck.acquire()
	DEN.writeFrame(ARG.outputFile, k, xi, force=True)
	if ARG.verbose:
		print("Frame %d: %d pixels corrected, fraction: %.2f%%"%(k, corrected_pixels, corrected_fraction*100))
	if lck is not None:
		lck.release()
	return corrected_pixels

if ARG.j == 0:
	for k in range(zdim):
		processFrame(ARG, k)
else:
	if ARG.j < 0:
		ARG.j = multiprocessing.cpu_count()
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
	else:
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))
	lock = Lock()
	tp = Pool(processes=ARG.j) 
	results = []
	for k in range(zdim):
		res = tp.apply_async(processFrame, args=(ARG, k, lock))
		results.append(res)
	while True:
		time.sleep(1)
		# catch exception if results are not ready yet
		try:
			ready = [result.ready() for result in results]
			successful = [result.successful() for result in results]
		except Exception:
			continue
		# exit loop if all tasks returned success
		if all(successful):
			total_pixels_corrected = sum([result.get() for result in results])
			total_pixels_fraction = total_pixels_corrected / totalSize
			print("Sucessfully created '%s' with %d pixels corrected, fraction: %.2f%%"%(ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
			break
		# raise exception reporting exceptions received from workers
		if all(ready) and not all(successful):
			total_pixels_corrected = sum([result.get() for result in results if result.successful()])
			total_frames_successful = sum([result.successful() for result in results])
			total_pixels_fraction = total_pixels_corrected / (total_frames_successful*frameSize)
			print("Out of %d tasks, %d failed"%(len(results), len(results)-total_frames_successful))
			print("From total %d frames corrected in '%s' with %d pixels corrected, fraction: %.2f%%"%(total_frames_sucessful, ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
			raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
