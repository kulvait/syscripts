#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

@author: Vojtech Kulvait

Tries to detect rotation center based on input extinction data


This script processes data to get volume information from the two consequent volumes that goes next to each other
"""

import argparse
import glob
import os
import errno
import sys
import h5py
import math
from denpy import DEN
from denpy import UTILS
from denpy import PETRA
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument("inputFileA", help="DEN file A to create matrix of scalar products")
parser.add_argument("inputFileB", help="DEN file B to create matrix of scalar products")
parser.add_argument("matrixProduct", help="Output DEN with matrix product of A and B")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
ARG = parser.parse_args()

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

equalInputs = False
if ARG.inputFileA == ARG.inputFileB:
	equalInputs = True

infA = DEN.readHeader(ARG.inputFileA)
infB = DEN.readHeader(ARG.inputFileB)

if infA["dimcount"] != 2 :
	if infA["dimcount"] != 3 and infA["dimspec"][2] != 1:
		raise ValueError("infA[\"dimcount\"] = %d "%(infA["dimcount"]))

if infB["dimcount"] != 2 :
	if infB["dimcount"] != 3 and infB["dimspec"][2] != 1:
		raise ValueError("infB[\"dimcount\"] = %d "%(infB["dimcount"]))

rowCountA = infA["dimspec"][1]
colCountA = infA["dimspec"][0]
rowCountB = infB["dimspec"][1]
colCountB = infB["dimspec"][0]

if colCountA != rowCountB:
	raise ValueError("Incompatible matrices colCountA=%d rowCountB=%d"%(colCountA, rowCountB))


A = DEN.getFrame(ARG.inputFileA, 0)
B = DEN.getFrame(ARG.inputFileB, 0)
F = np.dot(A, B)

DEN.storeNdarrayAsDEN(ARG.matrixProduct, F, force=ARG.force)
