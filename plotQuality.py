#!/usr/bin/env python
"""
Created 05/2026

@author: Vojtěch Kulvait
"""

import argparse
import os
import sys
import json
from pathlib import Path
from denpy import DEN
from algotom.rec.reconstruction import dfi_reconstruction
from skimage import measure
import logging
import numpy as np
import matplotlib.pyplot as plt

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


class JsonCache:
	def __init__(self, path=None):
		self.entries = {}
		if path is not None:
			self.path = path
		if os.path.exists(path):
			self.load(path)

	def _make_key(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		return (
			int(sinogram_ind),
			round(float(cor), 6),
			str(reconstruction),
			bool(apply_log),
		)

	def load(self, path):
		path = Path(path)
		with open(path, "r") as f:
			raw = json.load(f)
		self.entries = {}
		for entry in raw:
			key = self._make_key(
				sinogram_ind=entry["sinogram_ind"],
				cor=entry["cor"],
				reconstruction=entry["reconstruction"],
				apply_log=entry["apply_log"],
			)
			self.entries[key] = entry

	def save(self, path = None):
		if path is None:
			path = self.path
		if path is None:
			log.warning("No path specified for saving cache.")
			return
		path = Path(path)
		with open(path, "w") as f:
			json.dump(
				list(self.entries.values()),
				f,
				indent=2,
			)

	def get_entry(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
		create=False,
	):
		key = self._make_key(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		entry = self.entries.get(key)
		if entry is None and create:
			entry = {
				"sinogram_ind": int(sinogram_ind),
				"cor": float(cor),
				"reconstruction": reconstruction,
				"apply_log": apply_log,
				"scores": {},
			}
			self.entries[key] = entry
		return entry

	def get_json(self):
		return list(self.entries.values())

	def get_score(
		self,
		method,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		if entry is None:
			return None
		return entry["scores"].get(method)

	def set_score(
		self,
		method,
		value,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		entry["scores"][method] = value

	def get_scores(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		return entry["scores"]

parser = argparse.ArgumentParser()
parser.add_argument("inputSinograms")
parser.add_argument("--frame-index", type=int, default=None)
parser.add_argument("--pixel-window", type=int, default=None)
parser.add_argument("--metric", type=str, default=None)
parser.add_argument("--verbose", action="store_true")
ARG = parser.parse_args()

header = DEN.readHeader(ARG.inputSinograms)
if header["dimcount"] < 3:
	print("File %s is %d-dimensional, too few dimensions to estimate quality." % (ARG.inputSinograms, header["dimcount"]), file=sys.stderr)
	sys.exit(1)

if ARG.frame_index is not None:
	if ARG.frame_index < 0 or ARG.frame_index >= header["dimspec"][2]:
		print("Frame index %d is out of range [0, %d)." % (ARG.frame_index, header["dimspec"][2]), file=sys.stderr)
		sys.exit(1)
else:
	ARG.frame_index = header["dimspec"][2] // 2
sinogram = DEN.getFrame(ARG.inputSinograms, ARG.frame_index)
angles = np.linspace(0, np.pi, sinogram.shape[0], endpoint=False)
cacheFile = ARG.inputSinograms + ".cor_estimation_cache.json"
cache = JsonCache(cacheFile)
sinogramSize = sinogram.shape[1]
allEntries = cache.get_json()
allEntries = [entry for entry in allEntries if entry["sinogram_ind"] == ARG.frame_index and entry["reconstruction"] == "dfi" and not entry["apply_log"]]
if ARG.pixel_window is not None:
	center = sinogram.shape[1] / 2 - 0.5
	window_start = center - ARG.pixel_window
	window_end = center + ARG.pixel_window
	allEntries = [entry for entry in allEntries if window_start <= entry["cor"] <= window_end]

def plot_quality(methods, entries):
	#Sort entries by cor
	entries = sorted(entries, key=lambda e: e["cor"])
	center = sinogram.shape[1] / 2 - 0.5
	cor = [entry["cor"] - center for entry in entries]
	for method in methods:
		y = np.array([e["scores"].get(method, np.nan) for e in entries])
		plt.plot(cor, y, label=method)
	plt.axvline(0.0, color="red", linestyle="--", label="Center of rotation")
	plt.xlabel("Center of rotation offset in pixels")
	plt.ylabel("Quality score")
	plt.title("Quality scores for different centers of rotation")
	plt.legend()
	plt.grid()
	plt.show()

if ARG.metric is not None:
	plot_quality([ARG.metric], allEntries)
else:
	plot_quality(["blur_effect_3", "blur_effect_11", "blur_effect_30"], allEntries)




