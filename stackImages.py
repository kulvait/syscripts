#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
import os
import subprocess
import tempfile
import shutil
from termcolor import colored
from denpy import UTILS
from denpy import DEN

# Function to check if directory contains the required extFile
def contains_ext_file(directory, ext_file):
	return os.path.isfile(os.path.join(directory, ext_file))

# Function to execute shell commands
def execute_commands(commands):
	for command in commands:
		subprocess.run(command, shell=True)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("workingDirectory", type=str, help="Working directory")
parser.add_argument("extFile", type=str, help="File to stack")
parser.add_argument("outputExtFile", type=str, help="Output file")
parser.add_argument("--verbose", action="store_true")
ARG = parser.parse_args()

# Initialize experiments list
experiments = []

# Get list of subdirectories
subdirectories = [os.path.join(ARG.workingDirectory, subdirectory) for subdirectory in os.listdir(ARG.workingDirectory) 
				  if os.path.isdir(os.path.join(ARG.workingDirectory, subdirectory))]

# Sort subdirectories
subdirectories.sort()

# Variable to store dimz to ensure it's the same across all directories
dimz_global = None
dimx_global = []

# Process each subdirectory
for directory in subdirectories:
	# Check if the directory contains the required extFile
	if contains_ext_file(directory, ARG.extFile):
		# Read the params file and add to experiments list
		par = UTILS.readParamsFile(os.path.join(directory, "params"))
		par["directory"] = directory
		par["dirname"] = os.path.basename(directory)
		par["extFile"] = os.path.join(directory, ARG.extFile)
		extinfo = DEN.readHeader(par["extFile"])
		# Ensure extinfo["dimspec"] is a three-dimensional vector
		if len(extinfo["dimspec"]) != 3:
			raise ValueError(f"dimspec in {par['extFile']} is not a three-dimensional vector")
		dimx, dimy, dimz = extinfo["dimspec"]
		par["dimx"] = dimx
		par["dimy"] = dimy
		par["dimz"] = dimz
		# Check that dimz is consistent across all directories
		if dimz_global is None:
			dimz_global = dimz
		elif dimz != dimz_global:
			raise ValueError(f"dimz in {par['extFile']} is not consistent with other files")
		if dimx not in dimx_global:
			dimx_global.append(dimx)
		experiments.append(par)
		if ARG.verbose:
			print(f"Added parameters from {directory}")

for experiment in experiments:
	print(f"zpos_zero_pix in {experiment['extFile']} is {experiment['zpos_zero_pix']} suggested_minindex_beam is {experiment['suggested_minindex_beam']} suggested_maxindex_beam is {experiment['suggested_maxindex_beam']} (dimx, dimy, dimz) = ({experiment['dimx']}, {experiment['dimy']}, {experiment['dimz']})")
#exit()

# Check that dimx is consistent across all directories
if len(dimx_global) > 1:
	raise ValueError(f"dimx is not consistent across directories: {dimx_global}")

# Sort experiments list by zpos_zero_pix
experiments.sort(key=lambda x: int(x['zpos_zero_pix']))


#For testing restrict experiments to first 3 elements
#experiments = experiments[:3]

# Create temporary directory inside workingDirectory
temp_dir = tempfile.mkdtemp(dir=ARG.workingDirectory)

#First swap command
commands = []
for experiment in experiments:
	command = f"dentk-swapaxes {experiment['extFile']} {temp_dir}/{experiment['dirname']}.swap.den"
	commands.append(command)
	if ARG.verbose:
		print(f"Generated command: {command}")
# Execute all commands
print(colored("Executing %d dentk-swapaxes"%len(commands), "green"))
execute_commands(commands)

#Then select range
commands = []
for experiment in experiments:
	minindex = int(experiment["suggested_minindex_beam"])
	maxindex = int(experiment["suggested_maxindex_beam"])
	slabHeight = maxindex - minindex + 1
	if slabHeight == experiment["dimy"]:
		shutil.move(f"{temp_dir}/{experiment['dirname']}.swap.den", f"{temp_dir}/{experiment['dirname']}.cat.den")
		print(f"Slab height is equal to dimy, no need to crop")
	else:
		if dimy + 1 < maxindex:
			raise ValueError(f"dimy + 1 < maxindex in {experiment['extFile']}")
		else:
			command = f"dentk-cat -f {minindex}-{maxindex} {temp_dir}/{experiment['dirname']}.swap.den {temp_dir}/{experiment['dirname']}.cat.den"
			commands.append(command)
		if ARG.verbose:
			print(f"Generated command: {command}")
# Execute all commands
print(colored("Executing %d dentk-cat"%len(commands), "green"))
execute_commands(commands)

#Then merge cropped
mergecommand = f"dentk-merge {temp_dir}/stacked_swp.den "
for experiment in experiments:
	mergecommand += f"{temp_dir}/{experiment['dirname']}.cat.den "
print(colored(f"Executing {mergecommand}", "green"))
execute_commands([mergecommand])

#Then swap back
swapcommand = f"dentk-swapaxes --force {temp_dir}/stacked_swp.den {ARG.workingDirectory}/{ARG.outputExtFile}"
print(colored(f"Executing {swapcommand}", "green"))
execute_commands([swapcommand])

print(colored("Cleaning up", "green"))
shutil.rmtree(temp_dir)
# Further processing of experiments...
