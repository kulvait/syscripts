#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""

import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import re
from denpy import PETRA
from denpy import UTILS
from denpy import DEN

parser = argparse.ArgumentParser()
parser.add_argument("inputDirectories", nargs="+")
parser.add_argument("--write-params", action="store_true")
parser.add_argument("--binning", type=int, default=1)
parser.add_argument("--beam-profile", type=str, default=None)
parser.add_argument("--rotation-center-file", default=None, help="Use value stored in file to determine rotation center")
parser.add_argument("--log", type=str, default=None, help="Directory to save plots")
parser.add_argument("--cover-beam-all", action="store_true")
parser.add_argument("--verbose", action="store_true")

ARG = parser.parse_args()
import matplotlib
if ARG.verbose is False:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize experiments list
experiments = []
directories = ARG.inputDirectories
directories.sort()

# Process each directory
for directory in directories:
	experiment = {}
	experiment["directory"] = directory
	experiment["params"] = os.path.join(directory, "params")
	experiment["h5"] = os.path.join(directory, "h5")
	if not os.path.isdir(directory):
		print("Directory %s does not exist" % (directory))
		continue
	if not os.path.exists(experiment["params"]):
		print("Directory %s does not contain params file %s!" % (directory, experiment["params"]))
		continue
	if not os.path.exists(experiment["h5"]):
		print("Directory %s does not contain h5 file %s!" % (directory, experiment["h5"]))
		continue
	info = PETRA.getExperimentInfo(experiment["h5"])
	experiment["info"] = info
	if "s_stage_z" in info["setup"]:
		experiment["s_stage_z"] = -info["setup"]["s_stage_z"]
		zpos = experiment["s_stage_z"]
	experiments.append(experiment)

# Determine minimum z position
minzpos = min([experiment["s_stage_z"] for experiment in experiments])

# Assign pixel positions and sort experiments
for experiment in experiments:
	experiment["zpos_zero"] = experiment["s_stage_z"] - minzpos
	experiment["pix_size"] = experiment["info"]["pix_size"] * ARG.binning
	experiment["pix_height"] = int(experiment["info"]["camera"]["roi_height"] * experiment["info"]["pix_size"] / experiment["pix_size"])
	experiment["zpos_zero_pix"] = int(round(experiment["zpos_zero"] / experiment["pix_size"]))
	experiment["zpos_max_pix"] = experiment["zpos_zero_pix"] + experiment["pix_height"] - 1

experiments.sort(key=lambda x: x["zpos_zero_pix"])

# Suggest y ranges based on overlaps
for i, experiment in enumerate(experiments):
	zpos = experiment["zpos_zero"]
	zpos_pix = experiment["zpos_zero_pix"]
	zwidth_pix = experiment["pix_height"]
	zwidth = experiment["info"]["camera"]["roi_height"] * experiment["info"]["pix_size"]
	if i > 0:
		if experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_height"] > zpos_pix:
			overlap_width = experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_height"] - zpos_pix
			overlap_center = overlap_width // 2
			experiments[i-1]["suggested_maxpix"] = experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_height"] - overlap_center - 1
			experiment["suggested_minpix"] = experiments[i-1]["suggested_maxpix"] + 1
		else:
			experiment["suggested_minpix"] = zpos_pix
			experiments[i-1]["suggested_maxpix"] = experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_height"] - 1
	if i == 0:
		experiment["suggested_minpix"] = 0
	if i == len(experiments) - 1:
		experiment["suggested_maxpix"] = zpos_pix + zwidth_pix - 1

# Determine beam y ranges
if ARG.beam_profile is not None:
	for experiment in experiments:
		gammaProfile = os.path.join(experiment["directory"], ARG.beam_profile)
		gammainfo = DEN.readHeader(gammaProfile)
		GAMMA = DEN.getFrame(gammaProfile, 0)
		if GAMMA.shape[0] != experiment["pix_height"]:
			raise ValueError(f"Dimension mismatch: GAMMA ydim ({GAMMA.shape[0]}) does not match experiment pix_height ({experiment['pix_height']})")
		# Sum GAMMA along x dimension
		gamma_sum = np.sum(GAMMA, axis=1)
		gamma_sum = gaussian_filter1d(gamma_sum, 1)
		# Determine beam region
		# Calculate threshold based on median and standard deviation
		pivot_gamma = np.percentile(gamma_sum, 33)
		median_gamma = np.median(gamma_sum)
		std_gamma = np.std(gamma_sum[gamma_sum > median_gamma])
		threshold = pivot_gamma - 3 * std_gamma
		beam_fromy = np.argmax(gamma_sum > threshold)
		beam_toy = len(gamma_sum) - np.argmax(gamma_sum[::-1] > threshold) - 1
		# Add to experiment dictionary
		experiment["beam_fromy"] = beam_fromy
		experiment["beam_toy"] = beam_toy
		experiment["threshold"] = threshold
		experiment["gamma_sum"] = gamma_sum
		experiment["dimx"] = gammainfo["dimspec"][0]
		print("For threadhold %f the beam is present in y range [%d, %d] with values [%f, %f]" % (threshold, beam_fromy, beam_toy, gamma_sum[beam_fromy], gamma_sum[beam_toy]))
		# Plot gamma_sum

# Adjust suggested ranges based on beam presence
for i, experiment in enumerate(experiments):
	if "beam_fromy" in experiment and "beam_toy" in experiment:
		if i > 0:
			prev_experiment = experiments[i-1]
			if "beam_fromy" in prev_experiment and "beam_toy" in prev_experiment:
				prev_beam_to = prev_experiment["beam_toy"] + prev_experiment["zpos_zero_pix"]
				next_beam_from = experiment["beam_fromy"] + experiment["zpos_zero_pix"]
				center_beam_pos = (prev_beam_to + next_beam_from) // 2
				prev_experiment["suggested_maxpix_beam"] = min(center_beam_pos - prev_experiment["zpos_zero_pix"], prev_experiment["pix_height"] - 1)
				experiment["suggested_minpix_beam"] = max(0, center_beam_pos + 1 - experiment["zpos_zero_pix"])
		if i == 0:
			if ARG.cover_beam_all:
				experiment["suggested_minpix_beam"] = 0
			else:
				experiment["suggested_minpix_beam"] = max(0, experiment["beam_fromy"])
		if i == len(experiments) - 1:
			if ARG.cover_beam_all:
				experiment["suggested_maxpix_beam"] = experiment["pix_height"] - 1
			else:
				experiment["suggested_maxpix_beam"] = min(experiment["beam_toy"], experiment["pix_height"] - 1)
		

# Print experiment details
for experiment in experiments:
	zpos = experiment["zpos_zero"]
	zpos_pix = experiment["zpos_zero_pix"]
	zwidth_pix = experiment["pix_height"]
	zwidth = experiment["info"]["camera"]["roi_height"] * experiment["info"]["pix_size"]
	experiment["suggested_minindex"] = experiment["suggested_minpix"] - zpos_pix
	experiment["suggested_maxindex"] = experiment["suggested_maxpix"] - zpos_pix
	print(f"For experiment {experiment['directory']} there is s_stage_z = {experiment['s_stage_z']}")
	print(f"The z pos {experiment['s_stage_z']} transformed [{zpos}, {zpos + zwidth}] in pix [{zpos_pix}, {zpos_pix + zwidth_pix}) suggested [{experiment['suggested_minpix']}, {experiment['suggested_maxpix']}] zero based [{experiment['suggested_minindex']}, {experiment['suggested_maxindex']}]")
	if ARG.beam_profile is not None:
		print(f"Beam present in y range [{experiment['beam_fromy']}, {experiment['beam_toy']}]")
		print(f"Beam-based suggested range: [{experiment.get('suggested_minpix_beam', 'N/A')}, {experiment.get('suggested_maxpix_beam', 'N/A')}]")
	print()
	if ARG.beam_profile is not None and (ARG.log is not None or ARG.verbose):
		plt.figure()
		plt.plot(experiment["gamma_sum"])
		plt.axvline(x=experiment["beam_fromy"], color='r', linestyle='--', label='beam_fromy')
		plt.axvline(x=experiment["beam_toy"], color='r', linestyle=':', label='beam_toy')
		plt.axvline(x=experiment["suggested_minindex"], color='g', linestyle='--', label='suggested_minpix')
		plt.axvline(x=experiment["suggested_maxindex"], color='g', linestyle=':', label='suggested_maxpix')
		plt.axvline(x=experiment.get("suggested_minpix_beam", np.nan), color='b', linestyle='--', label='suggested_minpix_beam')
		plt.axvline(x=experiment.get("suggested_maxpix_beam", np.nan), color='b', linestyle=':', label='suggested_maxpix_beam')
		plt.title(f"Gamma Sum for {experiment['directory']} and threshold {experiment['threshold']}")
		plt.xlabel("y")
		plt.ylabel("Sum along x")
		plt.legend()
		plt.grid(True)
		if ARG.log:
			os.makedirs(ARG.log, exist_ok=True)
			plot_filename = os.path.join(ARG.log, f"{os.path.basename(experiment['directory'])}_gamma_sum.pdf")
			plt.savefig(plot_filename)
		if ARG.verbose:
			plt.show()
		else:
			plt.close()

def estimateStitchedSinogramWidth(dimx_orig, center_of_rotation_before_stitiching):
	overlap = min(2*center_of_rotation_before_stitiching + 1, 2 * (dimx_orig - center_of_rotation_before_stitiching) - 1)
	if overlap - np.floor(overlap) > 0.5:
		outWidth = 2 * dimx_orig - np.floor(overlap) - 1
	else:
		outWidth = 2 * dimx_orig - np.floor(overlap)
	print("Estimating stitched sinogram width for dimx=%d, center_of_rotation_before_stitiching=%f, overlap=%f, outWidth=%d" % (dimx_orig, center_of_rotation_before_stitiching, overlap, outWidth))
	return outWidth


# Estimate sinogram width
median_sinogram_width = None
if ARG.rotation_center_file is not None and ARG.beam_profile is not None:
	#Beamprofile needed for experiment["dimx"] to be defined
	for experiment in experiments:
		cod = (experiment["dimx"] - 1.0) / 2.0
		rcfile = os.path.join(experiment["directory"], ARG.rotation_center_file)
		rcf = open(rcfile, "r")
		rotation_center_pix = None
		for x in rcf.readlines():
			if re.search("^rotation_center_offset_pix_interp=", x): 
				rotation_center_offset_pix = float(x.rsplit("=",1)[1])
				rotation_center_offset_pix /= ARG.binning
		experiment["rotation_center_pix"] = rotation_center_offset_pix
		experiment["stitched_sinogram_width"] = estimateStitchedSinogramWidth(experiment["dimx"], cod + rotation_center_offset_pix)
	sinogram_widths = [experiment["stitched_sinogram_width"] for experiment in experiments]
	median_sinogram_width = np.median(sinogram_widths)
	print("sinogram_withs=%s, median_sinogram_width=%d" % (sinogram_widths, median_sinogram_width))

if ARG.write_params:
	for experiment in experiments:
		params = UTILS.readParamsFile(experiment["params"])
		params["suggested_minindex"] = experiment["suggested_minindex"]
		params["suggested_maxindex"] = experiment["suggested_maxindex"]
		if ARG.beam_profile is not None:
			params["suggested_minindex_beam"] = experiment.get("suggested_minpix_beam", np.nan)
			params["suggested_maxindex_beam"] = experiment.get("suggested_maxpix_beam", np.nan)
			params["beam_fromy"] = experiment["beam_fromy"]
			params["beam_toy"] = experiment["beam_toy"]
		if median_sinogram_width is not None:
			params["stitched_sinogram_width"] = int(median_sinogram_width)
		params["zpos_zero_pix"] = experiment["zpos_zero_pix"]
		UTILS.writeParamsFile(params, experiment["params"])
