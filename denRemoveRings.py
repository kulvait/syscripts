#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import argparse
import os
import numpy as np
import tomopy
from denpy import DEN

parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")

ARG = parser.parse_args()

if os.path.exists(ARG.outputDen):
	if not ARG.force:
		print("File %s exists add --force to overwrite!" % (ARG.outputDen))

den = DEN.getNumpyArray(ARG.inputDen)
out = tomopy.remove_ring(den)
DEN.storeNdarrayAsDEN(ARG.outputDen, out, force=ARG.force)
