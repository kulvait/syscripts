#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""

import argparse
import os
from denpy import PETRA
from denpy import UTILS


parser = argparse.ArgumentParser()
parser.add_argument("inputDirectories", nargs="+")
parser.add_argument("--write-params", action="store_true")
parser.add_argument("--binning", type=int, default=1)

ARG = parser.parse_args()

experiments = []
directories = ARG.inputDirectories
directories.sort()
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
		#This way I can consider s_stage_z of the position at 0 and z_pos will be increasing
		zpos = experiment["s_stage_z"]
	experiments.append(experiment)

#Now I assign pixel with the dataset with highest s_stage_z this value in mm
minzpos = min([experiment["s_stage_z"] for experiment in experiments])

for experiment in experiments:
	experiment["zpos_zero"] = experiment["s_stage_z"]-minzpos
	experiment["pix_size"] = experiment["info"]["pix_size"]*ARG.binning
	experiment["pix_width"] = int(experiment["info"]["camera"]["roi_height"]*experiment["info"]["pix_size"]/experiment["pix_size"])
	experiment["zpos_zero_pix"] = int(round(experiment["zpos_zero"]/experiment["pix_size"]))
	experiment["zpos_max_pix"] = experiment["zpos_zero_pix"] + experiment["pix_width"] - 1

experiments.sort(key=lambda x: x["zpos_zero_pix"])

for i, experiment in enumerate(experiments):
	zpos = experiment["zpos_zero"]
	zpos_pix = experiment["zpos_zero_pix"]
	zwidth_pix = experiment["pix_width"]
	zwidth = experiment["info"]["camera"]["roi_height"]*experiment["info"]["pix_size"]
	if i > 0:
		if experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_width"] > zpos_pix:
			overlap_width = experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_width"] - zpos_pix
			overlap_center = overlap_width//2
			experiments[i-1]["suggested_maxpix"] = experiments[i-1]["zpos_zero_pix"] + experiments[i-1]["pix_width"] - overlap_center - 1
			experiment["suggested_minpix"] = experiments[i-1]["suggested_maxpix"] + 1
	if i == 0:
		experiment["suggested_minpix"] = 0
	if i == len(experiments) - 1:
		experiment["suggested_maxpix"] = zpos_pix + zwidth_pix - 1

for experiment in experiments:
	zpos = experiment["zpos_zero"]
	zpos_pix = experiment["zpos_zero_pix"]
	zwidth_pix = experiment["pix_width"]
	zwidth = experiment["info"]["camera"]["roi_height"]*experiment["info"]["pix_size"]
	experiment["suggested_minindex"] = experiment["suggested_minpix"] - zpos_pix
	experiment["suggested_maxindex"] = experiment["suggested_maxpix"] - zpos_pix
	print("For experiment %s there is s_stage_z = %s" % (experiment["directory"], experiment["s_stage_z"]))
	print("The z pos %f transformed [%f, %f] in pix [%d, %d) suggested [%d, %d] zero based [%d, %d]"%(experiment["s_stage_z"], zpos, zpos + zwidth, zpos_pix, zpos_pix + zwidth_pix, experiment["suggested_minpix"], experiment["suggested_maxpix"], experiment["suggested_minindex"], experiment["suggested_maxindex"]))
	print()

if ARG.write_params:
	for experiment in experiments:
		params = UTILS.readParamsFile(experiment["params"])
		params["suggested_minindex"] = experiment["suggested_minindex"]
		params["suggested_maxindex"] = experiment["suggested_maxindex"]
		params["zpos_zero_pix"] = experiment["zpos_zero_pix"]
		UTILS.writeParamsFile(params, experiment["params"])
