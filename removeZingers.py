#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 2025

@author: Vojtěch Kulvait
"""
import argparse
import os
import numpy as np
import algotom.prep.removal as rem
from denpy import DEN
import multiprocessing
from multiprocessing import Pool, Lock
import traceback



parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--threshold", default=0.05, type=float, help="Threshold for zinger detection. [defaults to 0.05, recomended range 0.05-0.1]")
parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
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
dimz = np.uint64(dimspec[2])

if ARG.j < 0:
	ARG.j = multiprocessing.cpu_count()
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
elif ARG.j == 0:
	print("No threading will be used ARG.j=0.")
else:
	print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))


ntype = np.float32
DEN.writeEmptyDEN(ARG.outputDen, header["dimspec"],
				  elementtype=ntype, force=True)

# Global write_lock for workers
write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l

def processFrame(ARG, k):
	try:
		f = DEN.getFrame(ARG.inputDen, k)
		f_cor = rem.remove_zinger(f, ARG.threshold)
		f_cor = f_cor.astype(ntype)
		if write_lock:
			write_lock.acquire()
		try:
			DEN.writeFrame(ARG.outputDen, k, f_cor, force=True)
		finally:
			if write_lock:
				write_lock.release()
		return {"k": k, "error": None}
	except Exception as e:
		return {"k": k, "error": traceback.format_exc()}

class FakeAsyncResult:
	def __init__(self, value):
		self._value = value
	def get(self):
		return self._value

#Pool has to be initialized after the function definition
results = []
if ARG.j == 0:
	for k in range(zdim):
		res = processFrame(ARG, k)
		results.append(FakeAsyncResult(res))
else:
	lock = Lock()
	tp = Pool(processes=ARG.j, initializer=init_worker, initargs=(lock,))
	for k in range(dimz):
		res = tp.apply_async(processFrame, args=(ARG, k))
		results.append(res)
	tp.close()
	tp.join()

errors = []
for result in results:
	r = result.get()
	k = r["k"]
	if r["error"] is not None:
		errors.append((k, r["error"]))
	else:
		if ARG.verbose:
			print("Frame %d processed." % k)

if len(errors) > 0:
	print("The following frames raised exceptions:")
	for (k, error) in errors:
		print(f"Frame {k} exception:\n{error}")
	print(f"{len(errors)} frames raised exceptions.")
else:
	print("All frames processed successfully.")

print("END removeZingers.py")
