#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

@author: Vojtech Kulvait

Frame product. Faster implementation in dentk-frameproduct, where however inverse is not computed and matrix is transposed with respect to this output.
"""

import argparse
from denpy import DEN
import numpy as np
import time

import pylops
from scipy import misc
import pyproximal


parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file for TV minimization.")
parser.add_argument("outputFile", help="DEN file output")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")
parser.add_argument("--epsilon", default=1e-5, type=float, help="Epsilon of the prox_tv call [default: %(default)s].")
parser.add_argument("--lam", default=0.01, type=float, help="Lambda of the prox_tv call [default: %(default)s].")
parser.add_argument('--proxop',
                    default=None,
                    const='L2',
                    nargs='?',
                    choices=['L2', 'TVA', 'TVI', "TVIPG", "TVIPD", 'L1TVA', 'L1TVI'],
                    help='list servers, storage, or both (default: %(default)s)')
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

sampling = 1.
Gop = pylops.Gradient(dims=(dimy, dimx), sampling=sampling, edge=False,
                      kind='forward', dtype='float32')
L = 8. / sampling ** 2 # maxeig(Gop^H Gop)sampling = 1.

for k in range(frameCount):
	start = time.time()
	A = DEN.getFrame(ARG.inputFile, k)
	tau = 1.
	sigma = None
	if ARG.proxop is not None:
		if ARG.proxop in ["L1TVA", "L1TVI"]:
			#\sigma ||x-g||_1
			proxf = pyproximal.L1(g=A.ravel())
		#	proxf = pyproximal.L2(b=A.ravel())
			#proxf = pyproximal.L2(b=A.ravel())
		else:
			#sigma/2 ||x-b||
			proxf = pyproximal.L2(b=A.ravel())
	if ARG.proxop == "L2":
		sigma = 2
		tau = 1.
		mu = 1. / (tau*L)
		proxg = pyproximal.L2(sigma=sigma)
	elif ARG.proxop == "TVA":
		sigma = 0.01
		tau = 1.
		mu = tau / L
		#\sigma ||x||_1
		proxg = pyproximal.L1(sigma=sigma)
	elif ARG.proxop == "TVI":
		sigma = 0.01
		tau = 1.
		mu = tau / L
		proxg = pyproximal.L21(ndim=2, sigma=sigma)
	elif ARG.proxop == "TVIPG":
		#Very slow ~100s per projection compared to ~6 by other methods
		sigma = 0.01
		tau = 1./L
		proxg = pyproximal.TV(dims=A.shape, sigma=sigma)
	elif ARG.proxop == "TVIPD":
		sigma = 0.01
		tau = 1 / np.sqrt(L)
		mu = 1. / (tau*L)
		proxg = pyproximal.L21(ndim=2, sigma=sigma)
	elif ARG.proxop == "L1TVA":
		sigma = 0.01
		tau = 1.
		mu = tau / L
		#\sigma ||x||_1
		proxg = pyproximal.L1(sigma=sigma)
	elif ARG.proxop == "L1TVI":
		sigma = 0.01
		#tau = 1 / np.sqrt(L)
		#mu = 1. / (tau*L)
		#More
		#sigma = 1000.0
		#tau = 0.1 / np.sqrt(L)
		#mu = 0.1 / (tau*L)
		tau = 1.
		mu = 1. / (tau*L)
		tau = 0.95 / (np.sqrt(L))
		mu = 0.95 / np.sqrt(L)
		sigma = 10
		proxg = pyproximal.L21(ndim=2, sigma=sigma)
	if ARG.proxop is not None:
		#argmin_x f(x) + g(Ax)
		if ARG.proxop in ["TVIPG"]:
			OUT = pyproximal.optimization.primal.ProximalGradient(proxf, proxg,
                                                     tau=tau,
                                                     x0=np.zeros_like(A.ravel()),
                                                     niter=100, acceleration="fista")
		elif ARG.proxop in ["TVIPD", "L1TVA", "L1TVI"]:
#			OUT =  pyproximal.optimization.primaldual.PrimalDual(proxf, proxg,
#                                                     A=Gop, tau=tau,
#                                                     mu=mu, theta=1.0,
#                                                     x0=np.zeros_like(A.ravel()),
#                                                     niter=100, show=True)
			OUT =  pyproximal.optimization.primaldual.AdaptivePrimalDual(proxf, proxg,
                                                     A=Gop, tau=tau,
                                                     mu=mu,
                                                     x0=np.zeros_like(A.ravel()),
                                                     #x0=np.copy(A).ravel(),
                                                     niter=100, show=False)[0]
		else:
			OUT = pyproximal.optimization.primal.LinearizedADMM(proxf, proxg,
                                                     A=Gop, tau=tau,
                                                     mu=mu,
                                                     x0=np.zeros_like(A.ravel()),
                                                     niter=100)[0]
		OUT = OUT.reshape((dimy, dimx))
	else:
		OUT = A
	DEN.writeFrame(ARG.outputFile, k, OUT, force=ARG.force)
	total = time.time() - start
	print("Processing %05d/%05d frame from %s took %0.2fs."%(k+1, frameCount, ARG.inputFile, total))
