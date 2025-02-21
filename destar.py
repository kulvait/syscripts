#!/usr/bin/env python
"""
Created on Wed May  9 12:59:30 2023

@author: Vojtech Kulvait

This script does the convolution of the image with given PSF
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
xdim = np.uint32(dimspec[0])
ydim = np.uint32(dimspec[1])
zdim = np.uint32(dimspec[2])

DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True)


def processFrame(ARG, k, lck=None):
	if lck is not None:
		lck.acquire()
	f = DEN.getFrame(ARG.inputFile, k)
	if lck is not None:
		lck.release()
	xi = np.array(f)
	for i in range(ARG.iterations):
		mf = median_filter(xi, size=ARG.filter_size)
		dif = xi - mf
		std = np.std(dif)
		lim = ARG.filter_threashold * std
		flt = np.abs(dif) > lim
		xi[flt] = mf[flt]  
	if lck is not None:
		lck.acquire()
	DEN.writeFrame(ARG.outputFile, k, xi, force=True)
	if ARG.verbose:
		print("Processed k=%d"%(k))
	if lck is not None:
		lck.release()

if ARG.j == 0:
	for k in range(zdim):
		if ARG.verbose:
			print ("Start %d"%(k))
		f = DEN.getFrame(ARG.inputFile, k)
		xi = np.array(f)
		for i in range(ARG.iterations):
			mf = median_filter(xi, size=ARG.filter_size)
			dif = xi - mf
			std = np.std(dif)
			lim = ARG.filter_threashold * std
			flt = np.abs(dif) > lim
			xi[flt] = mf[flt]  
		DEN.writeFrame(ARG.outputFile, k, xi, force=True)
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
			break
		# raise exception reporting exceptions received from workers
		if all(ready) and not all(successful):
			raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
