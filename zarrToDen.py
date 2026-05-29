#!/usr/bin/env python
"""
This script converts a 3D image stack (time or depth series) stored in the Zarr format to the DEN format.

Created: 03/2026, use core functionality of destar.py from syscripts

@author: Vojtech Kulvait
@license: GNU GPL v3


Hot pixel / zinger removal tool for 2D/3D image stacks.
	
	INPUT / OUTPUT FORMAT
	---------------------
	The file format is inferred from the presence of ':' in the path.
	
	Zarr format:
	  If the argument contains a colon ':' it is interpreted as:
	
		  /path/to/zarr_container:/path/inside/zarr
	
	  Examples:
		  data.zarr:/volume
		  /data/experiment.zarr:/group/subgroup/array
	
	  - Left side  → filesystem path to Zarr store
	  - Right side → internal array path inside the store
	
	  Zarr storage type is inferred from the left-side suffix:
	
		  *.zarr		→ directory-based Zarr store
		  *.zip, *.zar	→ zipped Zarr store (read/write as archive)
	
	DEN format:
	  If the argument does NOT contain ':' it is interpreted as a DEN file:
	
		  /path/to/file.den
	
	Notes:
	- Zarr supports parallel writes (no locking required).
	- DEN writes are synchronized (thread-locked).
"""

import argparse
from denpy import DEN 
from denpy import ZAR
import zarr
import numpy as np
from multiprocessing.dummy import Process, Lock, Pool
import multiprocessing
import time
import sys
import traceback
import logging
import asyncio
from imagecodecs.numcodecs import register_codecs as register_numcodecs_codecs
from imagecodecs.zarr import register_codecs as register_zarr_codecs
register_numcodecs_codecs()
register_zarr_codecs()


# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)
log.propagate = False # Prevent log messages from being propagated to the root logger


# Global variables for controling I/O and synchronization in workers
inputIsZarr = False
outputIsZarr = False
inputArray = None
outputArray = None
maskOutputArray = None
outputType = np.float32

# Global write_lock for workers
write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l 

# Lock is protected by finally clause to avoid deadlocks
def processFrame(ARG, k):
	try:
		# Read (no lock unless DEN requires one)
		if inputIsZarr:
			f = inputArray[k]
		else:
			f = DEN.getFrame(ARG.inputFile, k)
		if write_lock:
			write_lock.acquire()
			try:
				if not outputIsZarr:
					DEN.writeFrame(ARG.outputFile, k, f.astype(outputType), force=True)
				else:
					outputArray[k] = f.astype(outputType)
			finally:
				if write_lock:
					write_lock.release()
		return {"k": k, "error": None}
	except Exception:
		return {"k": k, "error": traceback.format_exc()}

class FakeAsyncResult:
	def __init__(self, value):
		self._value = value
	def get(self):
		return self._value

def main():
	start_time = time.time()
	log.info("zarrToDen.py started with arguments: %s"%(sys.argv[1:],))
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument("inputFile", help="Input array, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("outputFile", help="Output array, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
	parser.add_argument("--keep-input-dtype", help="Keep input data type in output, otherwise output is float32.", action="store_true")
	parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
	ARG = parser.parse_args()
	
	global inputArray
	global inputIsZarr
	global outputArray
	global outputIsZarr
	global maskOutputArray
	global maskIsZarr
	global outputType
	
	if ":" in ARG.inputFile:
		zarTokens = ARG.inputFile.split(":", 1)
		zarStorePath = zarTokens[0]
		zarPath = zarTokens[1]
		if zarPath == "/":
			zarPath = ""
		if zarStorePath.endswith(".zip") or zarStorePath.endswith(".zar"):
			zarInputStore = zarr.storage.ZipStore(zarStorePath)
		else:
			zarInputStore = zarr.storage.LocalStore(zarStorePath)
		log.info(f"Opening Zarr array from store '{zarStorePath}' with path '{zarPath}'")
		inputArray = zarr.open_array(zarInputStore, mode="r", path=zarPath)
		inputIsZarr = True
		if len(inputArray.shape) != 3:
			raise ValueError("Input Zarr array must be 3D, but got shape %s"%(inputArray.shape,))
		zdim, ydim, xdim = inputArray.shape
		dimspec = (xdim, ydim, zdim)
		inputType = inputArray.dtype
	else:
		header = DEN.readHeader(ARG.inputFile)
		dimspec = header["dimspec"]
		xdim = np.uint64(dimspec[0])
		ydim = np.uint64(dimspec[1])
		zdim = np.uint64(dimspec[2])
		inputType = header["type"]
	
	if ARG.keep_input_dtype:
		outputType = inputType
	else:
		outputType = np.float32
	
	frameSize = xdim * ydim
	totalSize = frameSize * zdim
	
	outputZarStorePath = None
	if ":" in ARG.outputFile:
		zarTokens = ARG.outputFile.split(":", 1)
		outputZarStorePath = zarTokens[0]
		zarPath = zarTokens[1]
		if zarPath == "/":
			zarPath = ""
		if outputZarStorePath.endswith(".zip") or outputZarStorePath.endswith(".zar"):
			zarOutputStore = zarr.storage.ZipStore(outputZarStorePath, mode="a")
			zarOutputStore._sync_open() # After this fix might be removed https://github.com/zarr-developers/zarr-python/issues/3846
		else:
			zarOutputStore = zarr.storage.LocalStore(outputZarStorePath, mode="a")
		outputIsZarr = True
		try:
			outputArray = zarr.open_array(zarOutputStore, mode="r+", path=zarPath)
			if outputArray.shape != (zdim, ydim, xdim) or outputArray.dtype != outputType:
				asyncio.run(zarOutputStore.delete(zarPath))
				log.warning(f"Existing Zarr array in store '{outputZarStorePath}' with path '{zarPath}' has incompatible shape {outputArray.shape} or dtype {outputArray.dtype}, expected shape {(zdim, ydim, xdim)} and dtype {outputType}. It will be overwritten and for some stores (e.g. ZipStore) this leads to NotImplementedError")	
				outputArrayExists = False
			else:
				log.info(f"Using existing Zarr array in store '{outputZarStorePath}' with path '{zarPath}'")
				outputArrayExists = True
		except zarr.errors.ArrayNotFoundError:
			outputArrayExists = False
		if not outputArrayExists:
			log.info(f"Creating Zarr array in store '{outputZarStorePath}' with path '{zarPath}'")
			codec = ZAR.get_compressor(ARG.zarr_compression, clevel=ARG.zarr_clevel, zarrv2=False, dtype=outputType)
			outputArray = zarr.create_array(
					store=zarr.storage.StorePath(zarOutputStore, zarPath),
					shape=(zdim, ydim, xdim),
					chunks=(1, ydim, xdim),
					dtype=outputType,
					compressors=codec,
					zarr_format=3,
					overwrite=True,
				)
	else:
		DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True, elementtype=outputType)
	
	log.info(f"Starting processing file '{ARG.inputFile}' containing {zdim} frames of size {xdim}x{ydim} to produce '{ARG.outputFile}'")
	
	if ARG.j < 0:
		ARG.j = multiprocessing.cpu_count()
		log.info("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
	elif ARG.j == 0:
		log.info("No threading will be used ARG.j=0.")
	else:
		log.info("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))
	
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
	
	if len(errors) > 0:
		log.error(f"{zdim - total_frames_sucessful}/{zdim} frames raised the following exceptions:")
		for (k, error) in errors:
			log.error(f"Frame {k} exception:\n{error}")
		else:
			total_pixels_fraction = 0.0
			log.error("Processing failed for all frames, no pixels corrected in '%s'."%(ARG.outputFile))
	else:
		total_pixels_fraction = total_pixels_corrected / totalSize
	end_time = time.time()
	elapsed_time = end_time - start_time
	seconds = elapsed_time % 60
	minutes = int(elapsed_time // 60)
	hours = int(minutes // 60)
	minutes = minutes % 60
	formatted_time = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else (f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s")
	log.info("END zarrToDer.py, elapsed time: %s."%(formatted_time,))

if __name__ == "__main__":
	main()
