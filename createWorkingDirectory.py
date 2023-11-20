#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 2023

@author: Vojtech Kulvait

This scripts creates soft links in the directory structure
"""
import argparse
import glob
import os
import sys
import shutil
import pydicom
import re
import h5py
import random
from pathlib import Path
from denpy import DICOM
from denpy import PETRA
from denpy import UTILS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("rawDir")
parser.add_argument("workingDir")
parser.add_argument("--processed-dir", default=None)
parser.add_argument("--pattern", default=None, help="Regex patern to match against scanned directories [defults to None].")
parser.add_argument("--random-item-count", default=None, type=int, help="Maximum count of items to process, chosen randomly [defaults to None].")
parser.add_argument("--processed-only", action="store_true")
parser.add_argument("--params-update", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--singledir", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--verbose", action="store_true")
ARG = parser.parse_args()
rawDir = ARG.rawDir

if ARG.singledir:
	subDirs = [rawDir]
else:
	subDirs = next(os.walk(rawDir))[1]
	subDirs = [os.path.join(rawDir, x) for x in subDirs]

subDirsLen = len(subDirs)
if ARG.verbose:
	print("There is %d item in subDirs list to be processed."%subDirsLen)

if ARG.pattern is not None:
	regexp = re.compile(ARG.pattern)
	subDirs = [ x for x in subDirs if regexp.search(os.path.basename(x)) is not None]
	if len(subDirs) < subDirsLen:
		subDirsLen = len(subDirs)
		if ARG.verbose:
			print("There is %d item in subDirs list to be processed after filtering."%subDirsLen)

if ARG.random_item_count is not None and len(subDirs) > ARG.random_item_count:
	subDirs = random.choices(subDirs, k=ARG.random_item_count)

subDirs.sort()

def getInfo(directory):
	if not os.path.isdir(directory):
		return {}
	h5files = glob.glob(os.path.join(directory, "*.h5"))
	if len(h5files)!=1:
		print("Excluding directory %s because there is %d h5 files."%(directory, len(h5files)))
		return {}
	h5 = h5files[0]
	try:
		out={}
		scanData = PETRA.scanDataset(h5)
		out["h5"] = os.path.realpath(h5)
		out["rawdir"] = os.path.realpath(directory)
		out["scanData"] = scanData
		for f in scanData["image_file"]:
			f = f.lstrip("/")
			if not os.path.exists(os.path.join(directory, f)):
				print("Excluding directory %s as file %s does not exist."%(directory, os.path.join(directory, f)))
				return {}
		return out
	except:
		print("Excluding directory %s because there was error parsing h5 file %s"%(directory, h5))
		return {}

for d in subDirs:
	info = getInfo(d)
	if len(info)==0:
		continue
	basename = os.path.basename(info["rawdir"])
	workdir  = os.path.join(ARG.workingDir, basename)
	params={}
	params["h5"]=info["h5"]
	params["raw"]=info["rawdir"]
	params["workdir"]=os.path.realpath(workdir)
	print("Found file %s in %s"%(info["h5"], info["rawdir"]))
	if ARG.processed_dir is not None:
		processeddir = os.path.join(ARG.processed_dir, basename)
		if os.path.exists(processeddir):
			print("There is processed dir %s"%processeddir)
			params["processed"]=os.path.realpath(processeddir)
	if "processed" in params:
		logfile = glob.glob(os.path.join(params["processed"], '**/reco.log'), recursive=True)
		if len(logfile) == 0 and ARG.processed_only:
			continue
		elif len(logfile) != 0:
			params["logfile"] = os.path.realpath(logfile[0])
			logfile = open(params["logfile"], "r")
			logcontent = dict(map(str.strip, l.split(':', 1)) for l in logfile.readlines())
			if "rotation_axis_offset_reco" in logcontent:
				params["rotation_axis_offset_reco"] = logcontent["rotation_axis_offset_reco"]
			if "effective_pixel_size" in logcontent:
				params["effective_pixel_size"] = logcontent["effective_pixel_size"].rsplit(" ")[0]
			if "sample_detector_distance" in logcontent:
				params["sample_detector_distance"] = logcontent["sample_detector_distance"].rsplit(" ")[0]
	else:
		if ARG.processed_only:
			continue
	h5 = h5py.File(info["h5"], 'r')
	if "/entry/hardware/camera1" in h5:
		cam = "camera1"
	elif "/entry/hardware/camera" in h5:
		cam = "camera"
	else:
		raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."%info["h5"])
	params["pixel_size_camera"]=h5["entry/hardware/%s/pixelsize"%cam][0]
	params["pixel_size_magnification"]=h5["entry/hardware/%s/magnification"%cam][0]
	params["pixel_size_x"]=float(params["pixel_size_camera"])/float(params["pixel_size_magnification"])
	params["pixel_size_y"]=float(params["pixel_size_camera"])/float(params["pixel_size_magnification"])
	if "entry/hardware/%s/sensorsize_x"%cam in h5:
		pdimx = int(h5["entry/hardware/%s/sensorsize_x"%cam][0])
		params["pdimx"]="%d"%pdimx
		params["projection_size_x"]=pdimx*float(params["pixel_size_x"])
	if "entry/hardware/%s/sensorsize_y"%cam in h5:
		pdimy = int(h5["entry/hardware/%s/sensorsize_y"%cam][0])
		params["pdimy"]="%d"%pdimy
		params["projection_size_y"]=pdimy*float(params["pixel_size_y"])
	if ARG.params_update:
		if os.path.exists(workdir): 
			UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
	elif not ARG.dry_run:
		if os.path.exists(workdir): 
			if ARG.force:
				print("Removing existing %s"%workdir)
				shutil.rmtree(workdir)
			else:
				print("Skipping existing but updating params file %s"%workdir)
				UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
				continue
		Path(workdir).mkdir(parents=True, exist_ok=True)
		if "processed" in params:
			os.symlink(params["processed"], os.path.join(workdir, "processed"))
		os.symlink(params["h5"], os.path.join(workdir, "h5"))
		os.symlink(params["raw"], os.path.join(workdir, "raw"))
		UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
