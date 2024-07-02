#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb	9 19:28:11 2024

@author: kulvaitv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#tf.disable_v2_behavior()
import tensorflow_probability as tfp
print("Tensorflow version %s"%(tf.__version__))
import numpy as np
import matplotlib.pyplot as plt
import argparse
from denpy import DEN
import copy
import os
#import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("inputImg")
parser.add_argument("inputFitBasis")
parser.add_argument("outputExt")
parser.add_argument("--scale-basis", type=str, default=None, help="Initial estimate of basis weights, e.g. singular values.")
parser.add_argument("--basis-size", type=int, default=5, help="Number of basis elements from inputFitBasis to fit to extinctions.")
parser.add_argument("--alpha", type=str, help="Alpha mask for fitting.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--lbfgs", action="store_true")
parser.add_argument("--fit-slope", action="store_true")
nonlinearity = parser.add_mutually_exclusive_group()
nonlinearity.add_argument("--tv", action="store_true", help="Total variation fit")
nonlinearity.add_argument("--l1", action="store_true", help="L1 fit")

ARG = parser.parse_args()

inf_img = DEN.readHeader(ARG.inputImg)
inf_fit = DEN.readHeader(ARG.inputFitBasis)
if inf_fit["dimspec"][0] != inf_img["dimspec"][0] or inf_fit["dimspec"][1] != inf_img["dimspec"][1]:
	raise ArgumentException("Incompatible dimensions of input files")
frameCount = inf_img["dimspec"][2]

basisSize = ARG.basis_size
if ARG.basis_size > 0:
	if inf_fit["dimspec"][2] < ARG.basis_size:
		raise ArgumentException("ARG.basis_size=%d not in %s"%(ARG.basis_size, ARG.inputFitBasis))
else:
	basisSize = inf_fit["dimspec"][3]


def createSVDVec(fitFile, frameCount, alpha=None):
	inf = DEN.readHeader(fitFile)
	fitVec = np.zeros([frameCount, inf["shape"][1], inf["shape"][2]], dtype=np.float32)
	for k in np.arange(frameCount):
		f = DEN.getFrame(fitFile, k) 
		if alpha is not None:
			fitVec[k] = np.multiply(f, alpha)
		else:
			fitVec[k] = f
	return fitVec


def createSlopeVec(fitFile, frameCount, alpha=None):
	inf = DEN.readHeader(fitFile)
	fitVec = np.zeros([frameCount, inf["shape"][1], inf["shape"][2]], dtype=np.float32)
	slope = np.linspace(0, 1,  inf_fit["dimspec"][0])
	slope = np.tile(slope, [inf_fit["dimspec"][1], 1])
	for k in np.arange(frameCount):
		f = DEN.getFrame(fitFile, k) 
		f = np.multiply(f, slope)
		if alpha is not None:
			fitVec[k] = np.multiply(f,  alpha)
		else:
			fitVec[k] = f
	return fitVec


def createFitVec(fitFile, frameCount, alpha=None):
	if ARG.fit_slope:
		svd = createSVDVec(fitFile, frameCount, alpha)
		slope = createSlopeVec(fitFile, frameCount, alpha)
		return np.vstack([svd, slope])
	else:
		return createSVDVec(fitFile, frameCount, alpha)

def getMinimizer(b, fitVec, fitscale=None, dtype=tf.float32):
	b0 = copy.deepcopy(b)
	shape0=b.shape[0]
	shape1=b.shape[1]
	n =fitVec.shape[0]
	b0.shape=[shape0, shape1, 1]
	zeros = np.zeros([shape0, shape1, 1], dtype=np.float32)
	ones = np.ones([shape0, shape1, 1], dtype=np.float32)
	softzeros = ones * np.log(2) #Softplus(0) = log(2)
	def minimizer(x):
		ff = tf.constant(zeros, dtype=dtype)
		b_tf = tf.constant(b0, dtype=dtype)
		ones_tf = tf.constant(ones, dtype=dtype)
		softzeros_tf = tf.constant(softzeros, dtype=dtype)
		for k in np.arange(n):
			if fitscale is None:
				vec = fitVec[k]
			else:
				vec = fitscale[k] * fitVec[k]
			vec.shape = [shape0, shape1, 1]
			ff += x[k]*tf.constant(vec, dtype=dtype)
		#dif_tf = tf.math.add(tf.abs(tf.math.subtract(tf.math.divide_no_nan(ff, b_tf), ones_tf)), tf.abs(tf.math.subtract(tf.math.divide_no_nan(b_tf, ff), ones_tf)))
		#dif_tf = tf.math.subtract(tf.math.divide_no_nan(ff, b_tf), ones_tf)
		dif_tf = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(ff, b_tf))), softzeros_tf)
		if ARG.l1:
			return tf.reduce_sum(tf.abs(dif_tf))
		else:
			return tf.image.total_variation(dif_tf)
	return minimizer

#For use in L-BFGS
def get_value_and_grads(minimizer, x):
#	return tfp.math.value_and_gradient(minimizer, x)
	with tf.GradientTape() as tape:
		tape.watch(x)
		loss = minimizer(x)
		grads = tape.gradient(loss, x)
	return loss, grads

alpha=None
if ARG.alpha is not None:
	alpha = DEN.getFrame(ARG.alpha, 0)

fitVec = createFitVec(ARG.inputFitBasis, basisSize, alpha)
if alpha is not None:
	basisVec = createFitVec(ARG.inputFitBasis, basisSize)
else:
	basisVec = fitVec

fitScale = None
if ARG.scale_basis is not None:
	fitScale = DEN.getNumpyArray(ARG.scale_basis)
	fitScale = fitScale.flatten()
	fitScale = fitScale[:basisSize]
	if ARG.fit_slope:
		fitScale = np.concatenate([fitScale, 0.01 * fitScale])
	if ARG.verbose:
		print("Using fitScale=%s"%fitScale)


DEN.writeEmptyDEN(ARG.outputExt, [inf_fit["dimspec"][0], inf_fit["dimspec"][1], inf_img["dimspec"][2]], elementtype=np.dtype("<f4"), force=ARG.force)

#Initial extimate based on the firts frame

def getScalingEstimates(b, fitVec):
	b_weight = np.tensordot(b, b)
	fitScale = np.ones(len(fitVec), dtype=np.float32)
	for i in np.arange(len(fitVec)):
		p = np.tensordot(b, fitVec[i])
		if i < basisSize:
			fitScale[i] = b_weight / p
		else:
			fitScale[i] = (b_weight / p) * 0.01 #Downweighting slope
	return fitScale


b0 = DEN.getFrame(ARG.inputImg, 0)
if alpha is not None:
	b0 = np.multiply(b0, alpha)
#fitScale = np.ones(2*basisSize, dtype=np.float32)
fitScale = getScalingEstimates(b0, fitVec)

x_zero = np.ones(len(fitScale), dtype=np.float32)
x0 = x_zero
np.set_printoptions(precision=1)


def compute_lbfgs(minimizer, x0):
	def value_and_gradients_function(x):
		loss, grads = get_value_and_grads(minimizer, x)
		return loss, grads
	init_position = tf.Variable(x0, dtype=tf.float32)
	opt = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=value_and_gradients_function,initial_position=init_position)
	return opt

def compute_adam(minimizer, x0):
	x = tf.Variable(x0, dtype=tf.float32)
	loss_fn = lambda: minimizer(x)
	losses = tfp.math.minimize(loss_fn, optimizer = tf.keras.optimizers.Adam(learning_rate=0.05), num_steps=200)
	return x, losses


#def compute_kth(k):
#	b = DEN.getFrame(ARG.inputImg, k)
#	if alpha is not None:
#		b = np.multiply(b, alpha)
#	minimizer = getMinimizer(b, fitVec, fitScale, dtype=tf.float32)
#	fitval_init = minimizer(x_zero).numpy()
#	if ARG.lbfgs:
#		opt = compute_lbfgs(minimizer, x0)
#		print("LBFGS computed k=%d fitval_init = %s"%(k, fitval_init))
#		x = opt.position.numpy().astype(np.float32)
#		if not opt.converged:
#			print("ERROR: LBFGS optimization did not converge")
#		if np.any(np.isnan(x)):
#			x, losses = compute_adam(minimizer, x0)
#			x = x.numpy()
#	else:
#		x, losses = compute_adam(minimizer, x0)
#		x = x.numpy()
#		#plt.plot(losses)
#	#x0 = x#Update initial estimate
#	fitval_loss = minimizer(x).numpy()
#	xscaled = fitScale * x
#	if ARG.verbose:
#		print("Fit for k=%d x=%s xscaled=%s fit initial=%.2e after=%.2e ratio = %.3f"%(k, x, xscaled, fitval_init, fitval_loss, fitval_loss/fitval_init))
#		#plt.plot(losses)
#	out = np.tensordot(xscaled, basisVec, axes=[0,0])
#	DEN.writeFrame(ARG.outputExt, k,  out, force=True)

#pool = mp.Pool(10)
#for k in np.arange(inf_img["dimspec"][2]):
#	pool.apply_async(compute_kth, args=(k,))
#pool.close()
#pool.join()

#See https://stackoverflow.com/questions/39415263/how-to-run-tensorflow-on-multiple-core-and-threads
with tf.device('/cpu:0'):
	for k in np.arange(frameCount):
		b = DEN.getFrame(ARG.inputImg, k)
		if alpha is not None:
			b = np.multiply(b, alpha)
		minimizer = getMinimizer(b, fitVec, fitScale, dtype=tf.float32)
		fitval_init = minimizer(x_zero).numpy()
		if ARG.lbfgs:
			opt = compute_lbfgs(minimizer, x0)
			x = opt.position.numpy().astype(np.float32)
			if not opt.converged:
				print("ERROR: LBFGS optimization did not converge")
			if np.any(np.isnan(x)):
				x, losses = compute_adam(minimizer, x0)
			else:
				x0 = x #Update initial estimate
		else:
			x, losses = compute_adam(minimizer, x0)
			x = x.numpy()
			x0 = x #Update initial estimate
			#plt.plot(losses)
		fitval_loss = minimizer(x).numpy()
		xscaled = fitScale * x
		if ARG.verbose:
			print("Fit for k=%d x=%s xscaled=%s fit initial=%.2e after=%.2e ratio = %.3f"%(k, x, xscaled, fitval_init, fitval_loss, fitval_loss/fitval_init))
			#plt.plot(losses)
		out = np.tensordot(xscaled, basisVec, axes=[0,0])
		DEN.writeFrame(ARG.outputExt, k,  out, force=True)




