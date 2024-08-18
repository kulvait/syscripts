#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb	9 19:28:11 2024

@author: kulvaitv
"""



import tensorflow as tf
import tensorflow_probability as tfp
#print(tf.__version__)
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse
from denpy import DEN
import copy
import os

parser = argparse.ArgumentParser()
parser.add_argument("inputLogImg")
parser.add_argument("inputFitBasis")
parser.add_argument("outputExt")
parser.add_argument("--scale-basis", type=str, default=None, help="Initial estimate of basis weights, e.g. singular values.")
parser.add_argument("--basis-size", type=int, default=5, help="Number of basis elements from inputFitBasis to fit to extinctions.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")

ARG = parser.parse_args()
inf_fit = DEN.readHeader(ARG.inputFitBasis)
inf_img = DEN.readHeader(ARG.inputLogImg)
if inf_fit["dimspec"][0] != inf_img["dimspec"][0] or inf_fit["dimspec"][1] != inf_img["dimspec"][1]:
	raise ArgumentException("Incompatible dimensions of input files")


basisSize = ARG.basis_size
if ARG.basis_size > 0:
	if inf_fit["dimspec"][2] < ARG.basis_size:
		raise ArgumentException("ARG.basis_size=%d not in %s"%(ARG.basis_size, ARG.inputFitBasis))
else:
	basisSize = inf_fit["dimspec"][3]

def createFitVec(fitFile, frameCount):
	inf = DEN.readHeader(fitFile)
	fitVec = np.zeros([frameCount, inf["shape"][1], inf["shape"][2]], dtype=np.float32)
	for k in np.arange(frameCount):
		fitVec[k] = DEN.getFrame(fitFile, k)
	return fitVec

fitVec = createFitVec(ARG.inputFitBasis, basisSize)
fitScale = None
if ARG.scale_basis is not None:
	fitScale = DEN.getNumpyArray(ARG.scale_basis)
	fitScale = fitScale.flatten()
	fitScale = fitScale[:basisSize]

def getMinimizer(b, fitVec, fitscale):
	b0 = copy.deepcopy(b)
	shape0=b.shape[0]
	shape1=b.shape[1]
	n =fitVec.shape[0]
	b0.shape=[shape0, shape1, 1]
	def minimizer(x):
		y_tf = tf.constant(b0, dtype=tf.float32)
		for k in np.arange(n):
			vec = fitscale[k] * fitVec[k]
			vec.shape = [shape0, shape1, 1]
			y_tf -= x[k]*tf.constant(vec, dtype=tf.float32)
		return tf.image.total_variation(y_tf)
	return minimizer

DEN.writeEmptyDEN(ARG.outputExt, [inf_fit["dimspec"][0], inf_fit["dimspec"][1], inf_img["dimspec"][2]], elementtype=np.dtype("<f4"), force=ARG.force)

for k in np.arange(inf_img["dimspec"][2]):
	b = DEN.getFrame(ARG.inputLogImg, k)
	x0 = np.ones(basisSize, dtype=np.float32)
	if ARG.scale_basis is None:
		fitScale = np.zeros(basisSize, dtype=np.float32)
		for i in np.arange(basisSize):
			p = np.tensordot(b, fitVec[i])
			fitScale[i] = p
	x = tf.Variable(x0, dtype=tf.float32)
	minimizer = getMinimizer(b, fitVec, fitScale)
	loss_fn = lambda: minimizer(x)
	losses = tfp.math.minimize(loss_fn, optimizer = tf.keras.optimizers.Adam(learning_rate=0.05), num_steps=200)
	#plt.plot(losses)
	x = x.numpy()
	xscaled = fitScale * x
	if ARG.verbose:
		print("Fit for k=%d x=%s xscaled=%s"%(k, x, xscaled))
		#plt.plot(losses)
	out = b - np.tensordot(xscaled, fitVec, axes=[0,0])
	DEN.writeFrame(ARG.outputExt, k,  out, force=True)
