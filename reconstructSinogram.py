#!/usr/bin/env python
"""
Created 05/2026

@author: Vojtěch Kulvait
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from denpy import DEN
from algotom.rec.reconstruction import dfi_reconstruction
from skimage import measure
import logging

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



parser = argparse.ArgumentParser()
parser.add_argument("inputSinograms")
parser.add_argument("outputReconstruction")
parser.add_argument("--frame-index", type=int, default=None)
parser.add_argument("--offset", type=float, default=None)
parser.add_argument("--cor", type=float, default=None)
parser.add_argument("--apply-log", action="store_true")
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
center = sinogram.shape[1] / 2 - 0.5
cor = ARG.offset + center if ARG.offset is not None else center
if ARG.cor is not None:
	cor = ARG.cor
x = dfi_reconstruction(sinogram, cor, angles=angles, apply_log=ARG.apply_log)
DEN.storeNdarrayAsDEN(ARG.outputReconstruction, x.astype(np.float32), force=True)
