#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13.11.2024

Based on https://algotom.github.io/toc/api/algotom.post.postprocessing.html#algotom.post.postprocessing.remove_ring_based_wavelet_fft

@author: Vojtěch Kulvait
"""
import argparse
import os
import numpy as np
import algotom.post.postprocessing as rem
from denpy import DEN
import multiprocessing as mp



parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--size", default=1, type=float, help="Damping parameter. Larger is stronger [defaults to 1]")
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
		print("Removing zingers from data %s writing to %s data with threads=mp.cpu_count()=%d."%(ARG.inputDen, ARG.outputDen, ARG.threads))


ntype = np.float32
DEN.writeEmptyDEN(ARG.outputDen, header["dimspec"],
                  elementtype=ntype, force=True)

def process_frame(k, inputFile, outputFile, size, verbose=False):
	f = DEN.getFrame(inputFile, k)
	f_cor = rem.remove_ring_based_wavelet_fft(f, size=size)
	f_cor = f_cor.astype(ntype)
	DEN.writeFrame(outputFile, k, f_cor, force=True)
	if verbose:
		print("Frame %d processed." % k)
	return k

#Pool has to be initialized after the function definition
pool = mp.Pool(ARG.threads)

for k in range(dimz):
	pool.apply_async(process_frame, args=(k, ARG.inputDen, ARG.outputDen, ARG.size, ARG.verbose))
#	process_frame(k, ARG.inputDen, ARG.outputDen, ARG.size, ARG.verbose)

pool.close()
pool.join()
