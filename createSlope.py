#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 5.7.2024

@author: kulvaitv
"""
import argparse
import numpy as np
from denpy import DEN

parser = argparse.ArgumentParser()
parser.add_argument("inputImg")
parser.add_argument("outputImg")
parser.add_argument("--force", action="store_true")

ARG = parser.parse_args()




def createSlopeVec(fitFile, frameCount):
	inf = DEN.readHeader(fitFile)
	fitVec = np.zeros([frameCount, inf["shape"][1], inf["shape"][2]], dtype=np.float32)
	slope = np.linspace(0, 1,  inf_fit["dimspec"][0])
	slope = np.tile(slope, [inf_fit["dimspec"][1], 1])
	for k in np.arange(frameCount):
		f = DEN.getFrame(fitFile, k) 
		f = np.multiply(f, slope)
		fitVec[k] = f
	return fitVec


inf_img = DEN.readHeader(ARG.inputImg)
frameCount = inf_img["dimspec"][2]
dimx = inf_img["dimspec"][0]
dimy = inf_img["dimspec"][1]

slope = np.linspace(0, 1,  dimx)
slope = np.tile(slope, [dimy, 1])


output = np.zeros([2*frameCount, dimy, dimx], dtype=np.float32)
for k in np.arange(frameCount):
	f = DEN.getFrame(ARG.inputImg, k)
	output[k] = f
	f = np.multiply(f, slope)
	output[frameCount + k] = f


DEN.writeEmptyDEN(ARG.outputImg, [dimx, dimy, 2*frameCount], elementtype=np.dtype("<f4"), force=ARG.force)
for k in np.arange(2*frameCount):
	DEN.writeFrame(ARG.outputImg, k, output[k], force=True)
