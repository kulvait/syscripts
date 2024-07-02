#!/usr/bin/env python
"""
Created on Wed May	8 12:59:30 2019

@author: Vojtech Kulvait

Frame product. Faster implementation in dentk-frameproduct, where however inverse is not computed and matrix is transposed with respect to this output.
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
import multiprocessing as mp
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="DEN file containing matrix to invert")
parser.add_argument("outputFile", help="Output DEN with scalar products")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
ARG = parser.parse_args()

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

inf = DEN.readHeader(ARG.inputFile)
dimspec = inf["dimspec"]

if inf["dimcount"] != 2 and ( inf["dimcount"] != 3 or inf["dimspec"][2] != 1):
	raise ValueError("Input file %s is not 2D matrix."%ARG.inputFile)

rowCount = inf["dimspec"][1]
colCount = inf["dimspec"][0]

if rowCount != colCount:
	raise ValueError("Input file %s is not square matrix."%ARG.inputFile)

F = DEN.getFrame(ARG.inputFile, 0)
I = np.linalg.inv(F)
DEN.storeNdarrayAsDEN(ARG.outputFile, I, force=ARG.force)
