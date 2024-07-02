#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

@author: Vojtech Kulvait

Frame product. Faster implementation in dentk-frameproduct, where however inverse is not computed and matrix is transposed with respect to this output.
"""

import argparse
from denpy import DEN
from pyftvp import prox_tv
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file for TV minimization.")
parser.add_argument("outputFile", help="DEN file output")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")
parser.add_argument("--epsilon", default=1e-5, type=float, help="Epsilon of the prox_tv call [default: %(default)s].")
parser.add_argument("--lam", default=0.01, type=float, help="Lambda of the prox_tv call [default: %(default)s].")
ARG = parser.parse_args()



if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")


infIN = DEN.readHeader(ARG.inputFile)
if infIN["dimcount"] == 2:
	frameCount = 1
elif infIN["dimcount"] == 3:
	frameCount = infIN["dimspec"][2]
else:
	raise ValueError("File %s has %d dimensions but only 2 or 3 are supported."%(ARG.inputFile))
dimx = infIN["dimspec"][0]
dimy = infIN["dimspec"][1]


DEN.writeEmptyDEN(ARG.outputFile, infIN["dimspec"], force=ARG.force)

for k in range(frameCount):
	start = time.time()
	A = DEN.getFrame(ARG.inputFile, k)
	TVA = prox_tv(A, ARG.lam, epsilon=ARG.epsilon)
	TVA = TVA[0]
	#TVA = TVA.reshape((dimy, dimx))
	TVA = TVA[:,:,0]
	TVA = np.float32(TVA)
	DEN.writeFrame(ARG.outputFile, k, TVA, force=ARG.force)
	total = time.time() - start
	print("Processing %05d/%05d frame from %s took %0.2fs."%(k+1, frameCount, ARG.inputFile, total))
