#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
import os
import sys
import numpy as np
import algotom.prep.removal as rem
from denpy import DEN
import multiprocessing as mp
from multiprocessing.dummy import Process, Lock, Pool



parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--gauss", action="store_true", help="Use Gaussian filter instead of median filter. [defaults to False]")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--snr", default=3.0, type=float, help="Ratio (>1.0) for stripe detection. Greater is less sensitive. [defaults to 3.0]")
parser.add_argument("--la-size", default=51, type=int, help="Window size of the median filter to remove large stripes. [defaults to 51]")
parser.add_argument("--sm-size", default=21, type=int, help="Window size of the median filter to remove small to medium stripes. [defaults to 21]")
parser.add_argument("--stripe-fitting", action="store_true", help="Use stripe fitting method instead of filtering. [defaults to False]")
parser.add_argument("--stripe-fitting-order", default=2, type=int, help="Order of the polynomial for stripe fitting method. [defaults to 2]")
parser.add_argument("--stripe-fitting-sigma", default=10, type=int, help="Sigma of the Gaussian windown in x direction. Smaller is stronger smoothing. [defaults to 10]")
parser.add_argument("--stripe-fitting-chunks", default=1, type=int, help="Number of chunks to split the sinogram into for stripe fitting. [defaults to 1]")
parser.add_argument("--stripe-fitting-sort", action="store_true", help="Sort the intensities in each column before fitting. [defaults to False]")
parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")

ARG = parser.parse_args()

if os.path.exists(ARG.outputDen):
	if not ARG.force:
		print("File %s exists add --force to overwrite!" % (ARG.outputDen))
		exit(1)

header = DEN.readHeader(ARG.inputDen)
if len(header["dimspec"]) != 3:
	raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!" %
					(arg.inputFile, len(header["dimspec"])))
dimspec = header["dimspec"]
dimz = np.uint32(dimspec[2])

if ARG.threads == -1:
	ARG.threads = mp.cpu_count()
	if ARG.verbose:
		print("Removing strips from sinogram %s writing to %s data with threads=mp.cpu_count()=%d."%(ARG.inputDen, ARG.outputDen, ARG.threads))


ntype = np.float32
DEN.writeEmptyDEN(ARG.outputDen, header["dimspec"], elementtype=ntype, force=True)

write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l

def process_frame(k,inputFile,outputFile,snr,la_size,sm_size,gaussianFilter,verbose,stripe_fitting,stripe_order,stripe_sigma,stripe_chunks,stripe_sort):
	try:
		f = DEN.getFrame(inputFile, k)
		# Build options dictionary conditionally
		options = {"method": "gaussian_filter", "para1": (1, 21)} if gaussianFilter else None
		# Filtering method
		f_cor = rem.remove_all_stripe(
			f, snr, la_size, sm_size,
			**({"options": options} if options is not None else {})
		)
		f_cor = f_cor.astype(np.float32)
		# Optional stripe-fitting
		if stripe_fitting:
			f_cor = rem.remove_stripe_based_fitting(
				f_cor,
				order=stripe_order,
				sigma=stripe_sigma,
				sort=stripe_sort,
				num_chunk=stripe_chunks,
				**({"options": options} if options is not None else {})
			)
			f_cor = f_cor.astype(np.float32)
		# Write output frame
		if write_lock:
			write_lock.acquire()
		try:
			DEN.writeFrame(outputFile, k, f_cor, force=True)
		finally:
			if write_lock:
				write_lock.release()
		if verbose:
			print(f"Frame {k} processed.")
		return {"k": k, "status": "success"}
	except Exception as e:
		print(f"Error processing frame {k}: {e}")
		return {"k": k, "error": traceback.format_exc()}


snr = ARG.snr
la_size = ARG.la_size
sm_size = ARG.sm_size
#Pool has to be initialized after the function definition
lock = Lock()
pool = Pool(ARG.threads, initializer=init_worker, initargs=(lock,))

print("START removeSinogramStrips.py %s"%" ".join(sys.argv[1:]), flush=True)
results = []
for k in range(dimz):
	res = pool.apply_async(process_frame, args=(k, ARG.inputDen, ARG.outputDen, snr, la_size, sm_size, ARG.gauss, ARG.verbose, ARG.stripe_fitting, ARG.stripe_fitting_order, ARG.stripe_fitting_sigma, ARG.stripe_fitting_chunks, ARG.stripe_fitting_sort))
	results.append(res)
#process_frame(k, ARG.inputDen, ARG.outputDen, snr, la_size, sm_size, ARG.gauss, ARG.verbose, ARG.stripe_fitting, ARG.stripe_fitting_order, ARG.stripe_fitting_sigma, ARG.stripe_fitting_chunks, ARG.stripe_fitting_sort)

pool.close()
pool.join()


errors = []
for res in results:
	r = res.get()
	if "error" in r:
		errors.append((r["k"], r["error"]))

if len(errors) > 0:
	print("Errors occurred in the following frames:")
	for k, err in errors:
		print(f"Frame {k} error: {err}")

print("END removeSinogramStrips.py", flush=True)
