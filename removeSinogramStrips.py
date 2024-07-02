#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
import os
import numpy as np
import algotom.prep.removal as rem
from denpy import DEN
import multiprocessing as mp



parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--snr", default=3.0, type=float, help="Ratio (>1.0) for stripe detection. Greater is less sensitive. [defaults to 3.0]")
parser.add_argument("--la-size", default=51, type=int, help="Window size of the median filter to remove large stripes. [defaults to 51]")
parser.add_argument("--sm-size", default=21, type=int, help="Window size of the median filter to remove small to medium stripes. [defaults to 21]")
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
DEN.writeEmptyDEN(ARG.outputDen, header["dimspec"],
                  elementtype=ntype, force=True)

def process_frame(k, inputFile, outputFile, snr, la_size, sm_size, verbose=False):
	f = DEN.getFrame(inputFile, k)
	f_cor = rem.remove_all_stripe(f, snr, la_size, sm_size)
	f_cor = f_cor.astype(ntype)
	DEN.writeFrame(outputFile, k, f_cor, force=True)
	if verbose:
		print("Frame %d processed." % k)
	return k


snr = ARG.snr
la_size = ARG.la_size
sm_size = ARG.sm_size
#Pool has to be initialized after the function definition
pool = mp.Pool(ARG.threads)

for k in range(dimz):
	pool.apply_async(process_frame, args=(k, ARG.inputDen, ARG.outputDen, snr, la_size, sm_size, ARG.verbose))

pool.close()
pool.join()
