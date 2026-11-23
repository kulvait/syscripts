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
from scipy.ndimage import convolve
from scipy import ndimage as ndi
from denpy import DEN
import multiprocessing
from multiprocessing import Pool, Lock
import traceback

def select_zinger(mask, max_size):
    """
    Keep connected components smaller than max_size pixels.
    """
    labels, num = ndi.label(mask)
    if num == 0:
        return mask.astype(np.float32)
    sizes = ndi.sum(mask, labels, index=np.arange(1, num + 1))
    keep_labels = np.where(sizes <= max_size)[0] + 1
    mat_out = np.isin(labels, keep_labels)
    return mat_out.astype(np.float32)

def remove_zinger_safe(frame, zinger_threshold, size=2, check_size=False, median_filter=True):
    """
    Safe and clean replacement for Algotom remove_zinger().

    Improvements
    ------------
    - No circular wraparound artefacts (np.roll removed)
    - Uses convolution with the same sparse neighbour pattern
    - Robust normalization avoiding division by zero
    - Threshold comparable to original implementation

    Parameters
    ----------
    frame : array_like
        2D projection or sinogram image.
    zinger_threshold : float
        Zinger detection zinger_threshold, higher is less sensitive.
        Recommended range [0.05, 0.1].
    size : int
        Zinger size parameter.
    check_size : bool
        Optional size filtering via select_zinger().

    Returns
    -------
    ndarray
        Zinger-removed image.
    """
    frame = np.asarray(frame, dtype=np.float32)
    k = 2 * size + 1
    if median_filter:
        frame_ave = ndi.median_filter(frame, size=k, mode="reflect")
    else:
        # --- Build sparse kernel frameching Algotom neighbour offsets ---
        kernel = np.zeros((k, k), dtype=np.float32)
        offsets = [
            (-size, -size), (-size, 0), (-size, size),
            (0, -size),                (0, size),
            (size, -size),  (size, 0), (size, size),
        ]
        for di, dj in offsets:
            kernel[size + di, size + dj] = 1.0
        kernel /= kernel.sum()
        # --- Local average using reflect padding (no periodic artefacts) ---
        frame_ave = convolve(frame, kernel, mode="reflect")

    # --- Robust normalization ---
    eps = 1e-9 * np.mean(np.abs(frame_ave))
    eps = max(eps, np.finfo(frame_ave.dtype).eps)

    frame_nor = (frame - frame_ave) / (frame_ave + eps*np.sign(frame_ave))

    # --- Detect positive outliers ---
    mask_corrected_pixel = (np.abs(frame_nor) > zinger_threshold).astype(np.float32)

    # Optional size filtering
    if check_size:
        mask_corrected_pixel = select_zinger(mask_corrected_pixel, size)

    # --- Replace zingers with local average ---
    out = frame.copy()
    out[mask_corrected_pixel > 0] = frame_ave[mask_corrected_pixel > 0]

    return out





parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--threshold", default=0.1, type=float, help="Threshold for zinger detection, higher is less sensitive. [defaults to 0.05, recomended range 0.05-0.1]")
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
		f_cor = remove_zinger_safe(f, ARG.threshold)
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
