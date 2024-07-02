#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 28.6.2024

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
parser.add_argument("--alpha", type=str, help="Alpha mask for l1 fitting.")
parser.add_argument("--gamma", type=str, help="Alpha mask for total variation fitting.")
parser.add_argument("--report-transmission", type=str, default=None, help="Online transmission reporting.")
parser.add_argument("--report-extinction", type=str, help="Online extinction reporting.")
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
			fitVec[k] = np.multiply(f, alpha)
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

def reduce_array_by_factor(arr, k):
	# Ensure the input array dimensions are larger than k
	assert arr.shape[0] >= k and arr.shape[1] >= k, "Array dimensions must be larger than the reduction factor k"
	
	# Calculate the shape of the reduced array
	new_shape = (arr.shape[0] // k, arr.shape[1] // k)
	
	# Initialize the reduced array
	reduced_arr = np.zeros(new_shape)
	
	# Fill in the reduced array
	for i in range(new_shape[0]):
		for j in range(new_shape[1]):
			subarray = arr[i*k:(i+1)*k, j*k:(j+1)*k]
			reduced_arr[i, j] = np.mean(subarray)
	
	return reduced_arr

class AlphaGammaMinimizer:
	def __init__(self, b, fitVec, initScale=None, alpha=None, gamma=None, weight_l1=1, weight_tv=1, weight_neg=1, dtype=tf.float32):
		self.b = copy.deepcopy(b)
		self.reduce_k = 5
		self.fitVec = fitVec
		self.initScale = initScale
		self.dtype = dtype
		shape0, shape1 = b.shape
		shape0reduced = shape0 // self.reduce_k
		shape1reduced = shape1 // self.reduce_k
		n = fitVec.shape[0]
		self.shape0 = shape0
		self.shape1 = shape1
		self.n = n
		#Create arrays and their reduced versions
		self.alpha = np.ones_like(b) if alpha is None else copy.deepcopy(alpha)
		self.gamma = np.ones_like(b) if gamma is None else copy.deepcopy(gamma)
		self.gamma_reduced = reduce_array_by_factor(self.gamma, self.reduce_k)
		self.b_alpha = np.multiply(self.b, self.alpha)
		self.b_gamma = np.multiply(self.b, self.gamma)
		self.b_reduced = reduce_array_by_factor(self.b, self.reduce_k)
		self.b_gamma_reduced = np.multiply(self.b_reduced, self.gamma_reduced)
		self.alpha.shape = [shape0, shape1, 1]
		self.gamma.shape = [shape0, shape1, 1]
		self.gamma_reduced.shape = [shape0reduced, shape1reduced, 1]
		self.b_alpha.shape = [shape0, shape1, 1]
		self.b_gamma.shape = [shape0, shape1, 1]
		self.b_gamma_reduced.shape = [shape0reduced, shape1reduced, 1]
		self.fitVecAlpha = np.zeros((n, shape0, shape1, 1), dtype=np.float32)
		self.fitVecGamma = np.zeros((n, shape0, shape1, 1), dtype=np.float32)
		self.fitVecGamma_reduced = np.zeros((n, shape0reduced, shape1reduced, 1), dtype=np.float32)
		for i in range(n):
			v = fitVec[i]
			v_reduced = reduce_array_by_factor(v, self.reduce_k)
			v.shape = [shape0, shape1, 1]
			v_reduced.shape = [shape0reduced, shape1reduced, 1]
			self.fitVecAlpha[i] = initScale[i] * np.multiply(v, self.alpha)
			self.fitVecGamma[i] = initScale[i] * np.multiply(v, self.gamma)
			self.fitVecGamma_reduced[i] = initScale[i] * np.multiply(v_reduced, self.gamma_reduced)
		self.b.shape = [shape0, shape1, 1]
		self.ones = np.ones([shape0, shape1, 1], dtype=np.float32)
		self.zeros = np.zeros([shape0, shape1, 1], dtype=np.float32)
		self.softzeros = self.ones * np.log(2) # Softplus(0) = log(2)
		self.weight_l1 = weight_l1
		self.weight_neg = weight_neg
		self.weight_tv = weight_tv

	def getTVTensor(self, B, FF):
		#dif_tv = tf.math.divide_no_nan(FF, B)
		#dif_tv = tf.math.log(tf.math.softplus(B)) - tf.math.log(tf.math.softplus(FF))
		dif_tv = tf.math.divide_no_nan(FF, B) + tf.math.divide_no_nan(B, FF)
		return self.weight_tv * tf.image.total_variation(dif_tv)

	def getL1Tensor(self, B, FF):
		ones_tf = tf.constant(self.ones, dtype=self.dtype)
		softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, B))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(B, FF), ones_tf)), softzeros_tf)
		#dif_l1 = tf.math.subtract(FF, B)
		return self.weight_l1 * tf.reduce_sum(tf.abs(dif_l1))

	def getGammaNegativeTensor(self, B, FF):
		zeros_tf = tf.constant(self.zeros, dtype=self.dtype)
		neg_l2 = tf.math.maximum(tf.math.subtract(B, FF), zeros_tf)
		return self.weight_neg * tf.math.sqrt(tf.reduce_sum(tf.square(neg_l2)))

	def getMinimizer(self):
		def minimizer(x):
			b_alpha_tf = tf.constant(self.b_alpha, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			b_gamma_reduced_tf = tf.constant(self.b_gamma_reduced, dtype=self.dtype)
			ones_tf = tf.constant(self.ones, dtype=self.dtype)
			ff_alpha = x[0] * tf.constant(self.fitVecAlpha[0], dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			ff_gamma_reduced = x[0] * tf.constant(self.fitVecGamma_reduced[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_alpha += x[k] * tf.constant(self.fitVecAlpha[k], dtype=self.dtype)
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
				ff_gamma_reduced += x[k] * tf.constant(self.fitVecGamma_reduced[k], dtype=self.dtype)
			ret = None
			if self.weight_l1 > 0:
				ret = self.getL1Tensor(b_alpha_tf, ff_alpha)
			if self.weight_tv > 0:
				if ret is None:
					ret = self.getTVTensor(b_gamma_reduced_tf, ff_gamma_reduced)
				else:
					ret += self.getTVTensor(b_gamma_reduced_tf, ff_gamma_reduced)
			if self.weight_neg > 0:
				if ret is None:
					ret = self.getGammaNegativeTensor(b_gamma_tf, ff_gamma)
				else:
					ret += self.getGammaNegativeTensor(b_gamma_tf, ff_gamma)
			return ret
		return minimizer

	def getL1Minimizer(self):
		def l1_minimizer(x):
			b_alpha_tf = tf.constant(self.b, dtype=self.dtype)
			ff_alpha = x[0] * tf.constant(self.fitVecAlpha[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_alpha += x[k] * tf.constant(self.fitVecAlpha[k], dtype=self.dtype)
			return self.getL1Tensor(b_alpha_tf, ff_alpha)
		return l1_minimizer

	def getTVMinimizer(self):
		def tv_minimizer(x):
			ones_tf = tf.constant(self.ones, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma_reduced, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma_reduced[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma_reduced[k], dtype=self.dtype)
			return self.getTVTensor(b_gamma_tf, ff_gamma)
		return tv_minimizer
	
	def getNegMinimizer(self):
		def neg_minimizer(x):
			ones_tf = tf.constant(self.ones, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
			return self.getGammaNegativeTensor(b_gamma_tf, ff_gamma)
		return neg_minimizer

	def reportMinimizerValue(self, x, x0=None):
		l1_minimizer = self.getL1Minimizer()
		tv_minimizer = self.getTVMinimizer()
		neg_minimizer = self.getNegMinimizer()
		l1_value = l1_minimizer(x).numpy()
		tv_value = tv_minimizer(x).numpy()
		neg_value = neg_minimizer(x).numpy()
		report = f"x: {x}"
		if x0 is not None:
			l1_value_x0 = l1_minimizer(x0).numpy()
			tv_value_x0 = tv_minimizer(x0).numpy()
			neg_value_x0 = neg_minimizer(x0).numpy()
			l1_ratio = l1_value / l1_value_x0 if l1_value_x0 != 0 else float('inf')
			tv_ratio = tv_value / tv_value_x0 if tv_value_x0 != 0 else float('inf')
			neg_ratio = neg_value / neg_value_x0 if neg_value_x0 != 0 else float('inf')
			if self.weight_l1 > 0:
				report += f", L1: {l1_value:.2e} (Ratio: {l1_ratio:.3f})"
			if self.weight_tv > 0:
				report += f", TV: {tv_value:.2e} (Ratio: {tv_ratio:.3f})"
			if self.weight_neg > 0:
				report += f", NEG: {neg_value:.2e} (Ratio: {neg_ratio:.3f})"
			report += "."
		else:
			if self.weight_l1 > 0:
				report += f", L1: {l1_value:.2e}"
			if self.weight_tv > 0:
				report += f", TV: {tv_value:.2e}"
			if self.weight_neg > 0:
				report += f", NEG: {neg_value:.2e}"
			report += "."
		return report

class AlphaGammaConsecutiveMinimizer:
	def __init__(self, b, fitVec, ffoverb_prev, initScale=None, alpha=None, gamma=None, weight_l1=1, weight_tv=1, weight_neg=1, weight_cns_l1=1, weight_cns_tv=1, dtype=tf.float32):
		self.reduce_k = 5
		self.numpyshape = b.shape
		self.numpyshape_reduced = (b.shape[0] // self.reduce_k, b.shape[1] // self.reduce_k)
#		self.tensorshape = self.numpyshape
#		self.tensorshape_reduced = self.numpyshape_reduced
#For https://www.tensorflow.org/api_docs/python/tf/image/total_variation
		self.tensorshape = self.numpyshape + (1,)
		self.tensorshape_reduced = self.numpyshape_reduced + (1,)
		self.weight_l1 = weight_l1
		self.weight_neg = weight_neg
		self.weight_tv = weight_tv
		self.weight_cns_l1 = weight_cns_l1
		self.weight_cns_tv = weight_cns_tv
		self.dtype = dtype
		n = fitVec.shape[0]
		self.n = n
		self.initScale = initScale
#Copy relevant arrays
		self.b = copy.deepcopy(b)
		self.ffoverb_prev = copy.deepcopy(ffoverb_prev)
		self.fitVec = fitVec
		self.alpha = np.ones_like(b) if alpha is None else copy.deepcopy(alpha)
		self.gamma = np.ones_like(b) if gamma is None else copy.deepcopy(gamma)
#Maniplate with arrays to get relevant values
		self.b_reduced = reduce_array_by_factor(self.b, self.reduce_k)
		self.ffoverb_prev_reduced = reduce_array_by_factor(self.ffoverb_prev, self.reduce_k)
		self.gamma_reduced = reduce_array_by_factor(self.gamma, self.reduce_k)
		self.b_alpha = np.multiply(self.b, self.alpha)
		self.b_gamma = np.multiply(self.b, self.gamma)
		self.b_gamma_reduced = np.multiply(self.b_reduced, self.gamma_reduced)
		self.ffoverb_prev_alpha = np.multiply(self.ffoverb_prev, self.alpha)
		self.ffoverb_prev_gamma = np.multiply(self.ffoverb_prev, self.gamma)
		self.ffoverb_prev_gamma_reduced = np.multiply(self.ffoverb_prev_reduced, self.gamma_reduced)
		self.ones = np.ones(self.tensorshape, dtype=np.float32)
		self.zeros = np.zeros(self.tensorshape, dtype=np.float32)
		self.softzeros = self.ones * np.log(2) # Softplus(0) = log(2)
#Adjust shapes
		self.b.shape = self.tensorshape
		self.ffoverb_prev.shape = self.tensorshape
		self.alpha.shape = self.tensorshape
		self.gamma.shape = self.tensorshape
		self.b_reduced.shape = self.tensorshape_reduced
		self.ffoverb_prev_reduced.shape = self.tensorshape_reduced
		self.gamma_reduced.shape = self.tensorshape_reduced
		self.b_alpha.shape = self.tensorshape
		self.b_gamma.shape = self.tensorshape
		self.b_gamma_reduced.shape = self.tensorshape_reduced
		self.ffoverb_prev_alpha.shape = self.tensorshape
		self.ffoverb_prev_gamma.shape = self.tensorshape
		self.ffoverb_prev_gamma_reduced.shape = self.tensorshape_reduced
		self.fitVecAlpha = np.zeros((n, ) + self.tensorshape, dtype=np.float32)
		self.fitVecGamma = np.zeros((n, ) + self.tensorshape, dtype=np.float32)
		self.fitVecGamma_reduced = np.zeros((n, ) + self.tensorshape_reduced, dtype=np.float32)
		for i in range(n):
			v = fitVec[i]
			v_reduced = reduce_array_by_factor(v, self.reduce_k)
			v.shape = self.tensorshape
			v_reduced.shape = self.tensorshape_reduced
			self.fitVecAlpha[i] = initScale[i] * np.multiply(v, self.alpha)
			self.fitVecGamma[i] = initScale[i] * np.multiply(v, self.gamma)
			self.fitVecGamma_reduced[i] = initScale[i] * np.multiply(v_reduced, self.gamma_reduced)
#		self.alpha.shape = [shape0, shape1, 1]
#		self.gamma.shape = [shape0, shape1, 1]
#		self.gamma_reduced.shape = [shape0reduced, shape1reduced, 1]
#		self.b_alpha.shape = [shape0, shape1, 1]
#		self.b_gamma.shape = [shape0, shape1, 1]
#		self.b_gamma_reduced.shape = [shape0reduced, shape1reduced, 1]
#		self.ffoverb_prev.shape = [shape0, shape1, 1]
#		self.ffoverb_prev_reduced.shape = [shape0reduced, shape1reduced, 1]
#		self.b.shape = [shape0, shape1, 1]
	
	def getTVCnsTensor(self, PX, B, FF):
		#dif_tv = tf.math.divide_no_nan(FF, PF)
		#dif_tv = tf.math.log(tf.math.softplus(PF)) - tf.math.log(tf.math.softplus(FF))
		dif_tv = tf.math.subtract(PX, tf.math.divide_no_nan(FF, B))
		return self.weight_cns_tv * tf.image.total_variation(dif_tv)

	def getL1CnsTensor(self, PX, B, FF):
		#ones_tf = tf.constant(self.ones, dtype=self.dtype)
		#softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, PF))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(PF, FF), ones_tf)), softzeros_tf)
		dif_l1 = tf.math.subtract(PX, tf.math.divide_no_nan(FF, B))
		return self.weight_cns_l1 * tf.reduce_sum(tf.abs(dif_l1))

	def getTVTensor(self, B, FF):
		#dif_tv = tf.math.divide_no_nan(FF, B)
		#dif_tv = tf.math.log(tf.math.softplus(B)) - tf.math.log(tf.math.softplus(FF))
		dif_tv = tf.math.divide_no_nan(FF, B) + tf.math.divide_no_nan(B, FF)
		return self.weight_tv * tf.image.total_variation(dif_tv)

	def getL1Tensor(self, B, FF):
		ones_tf = tf.constant(self.ones, dtype=self.dtype)
		softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, B))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(B, FF), ones_tf)), softzeros_tf)
		#dif_l1 = tf.math.subtract(FF, B)
		return self.weight_l1 * tf.reduce_sum(tf.abs(dif_l1))

	def getGammaNegativeTensor(self, B, FF):
		zeros_tf = tf.constant(self.zeros, dtype=self.dtype)
		neg_l2 = tf.math.maximum(tf.math.subtract(B, FF), zeros_tf)
		return self.weight_neg * tf.math.sqrt(tf.reduce_sum(tf.square(neg_l2)))

	def getMinimizer(self):
		def minimizer(x):
			b_alpha_tf = tf.constant(self.b_alpha, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			b_gamma_reduced_tf = tf.constant(self.b_gamma_reduced, dtype=self.dtype)
			divprev_gamma_tf = tf.constant(self.ffoverb_prev_gamma, dtype=self.dtype)
			ones_tf = tf.constant(self.ones, dtype=self.dtype)
			ff_alpha = x[0] * tf.constant(self.fitVecAlpha[0], dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			ff_gamma_reduced = x[0] * tf.constant(self.fitVecGamma_reduced[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_alpha += x[k] * tf.constant(self.fitVecAlpha[k], dtype=self.dtype)
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
				ff_gamma_reduced += x[k] * tf.constant(self.fitVecGamma_reduced[k], dtype=self.dtype)
			ret = None
			if self.weight_l1 > 0:
				ret = self.getL1Tensor(b_alpha_tf, ff_alpha)
			if self.weight_tv > 0:
				t = self.getTVTensor(b_gamma_reduced_tf, ff_gamma_reduced)
				if ret is None:
					ret = t
				else:
					ret += t
			if self.weight_neg > 0:
				t = self.getGammaNegativeTensor(b_gamma_tf, ff_gamma)
				if ret is None:
					ret = t
				else:
					ret += t
			if self.weight_cns_l1 > 0:
				t = self.getL1CnsTensor(divprev_gamma_tf, b_gamma_tf, ff_gamma)
				if ret is None:
					ret = t
				else:
					ret += t
			if self.weight_cns_tv > 0:
				t = self.getTVCnsTensor(divprev_gamma_tf, b_gamma_tf, ff_gamma)
				if ret is None:
					ret = t
				else:
					ret += t
			return ret
		return minimizer

	def getL1Minimizer(self):
		def l1_minimizer(x):
			b_alpha_tf = tf.constant(self.b, dtype=self.dtype)
			ff_alpha = x[0] * tf.constant(self.fitVecAlpha[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_alpha += x[k] * tf.constant(self.fitVecAlpha[k], dtype=self.dtype)
			return self.getL1Tensor(b_alpha_tf, ff_alpha)
		return l1_minimizer

	def getTVMinimizer(self):
		def tv_minimizer(x):
			b_gamma_tf = tf.constant(self.b_gamma_reduced, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma_reduced[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma_reduced[k], dtype=self.dtype)
			return self.getTVTensor(b_gamma_tf, ff_gamma)
		return tv_minimizer
	
	def getNegMinimizer(self):
		def neg_minimizer(x):
			ones_tf = tf.constant(self.ones, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
			return self.getGammaNegativeTensor(b_gamma_tf, ff_gamma)
		return neg_minimizer

	def getCnsL1Minimizer(self):
		def cns_l1_minimizer(x):
			divprev_gamma_tf = tf.constant(self.ffoverb_prev_gamma, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
			return self.getL1CnsTensor(divprev_gamma_tf, b_gamma_tf, ff_gamma)
		return cns_l1_minimizer

	def reportState(self):
		DEN.storeNdarrayAsDEN("~/tmp/AGF_ffoverb_prev_gamma.den", self.ffoverb_prev_gamma, force=True)
	
	def reportGammaState(self, x):
		divprev_gamma_tf = tf.constant(self.ffoverb_prev_gamma, dtype=self.dtype)
		DEN.storeNdarrayAsDEN("AGF_ffoverb_prev_gamma.den", np.reshape(self.ffoverb_prev_gamma, self.numpyshape), force=True)
		DEN.storeNdarrayAsDEN("AGF_b_gamma.den", np.reshape(self.b_gamma, self.numpyshape), force=True)
		ffg = x[0] * self.fitVecGamma[0]
		for k in range(1, self.n):
			ffg += x[k] * self.fitVecGamma[k]
		for k in range(self.n):
			DEN.storeNdarrayAsDEN("AGF_fitVecGamma_%d.den"%k, np.reshape(self.fitVecGamma[k], self.numpyshape), force=True)
		DEN.storeNdarrayAsDEN("AGF_ff_gamma.den", np.reshape(ffg, self.numpyshape), force=True)

	def getCnsTVMinimizer(self):
		def cns_tv_minimizer(x):
			divprev_gamma_tf = tf.constant(self.ffoverb_prev_gamma, dtype=self.dtype)
			b_gamma_tf = tf.constant(self.b_gamma, dtype=self.dtype)
			ff_gamma = x[0] * tf.constant(self.fitVecGamma[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_gamma += x[k] * tf.constant(self.fitVecGamma[k], dtype=self.dtype)
			return self.getTVCnsTensor(divprev_gamma_tf, b_gamma_tf, ff_gamma)
		return cns_tv_minimizer

	def reportMinimizerValue(self, x, x0=None):
		l1_minimizer = self.getL1Minimizer()
		tv_minimizer = self.getTVMinimizer()
		neg_minimizer = self.getNegMinimizer()
		cns_l1_minimizer = self.getCnsL1Minimizer()
		cns_tv_minimizer = self.getCnsTVMinimizer()
		l1_value = l1_minimizer(x).numpy()
		tv_value = tv_minimizer(x).numpy()
		neg_value = neg_minimizer(x).numpy()
		cns_l1_value = cns_l1_minimizer(x).numpy()
		cns_tv_value = cns_tv_minimizer(x).numpy()
		report = f"x: {x}\n"
		if x0 is not None:
			l1_value_x0 = l1_minimizer(x0).numpy()
			tv_value_x0 = tv_minimizer(x0).numpy()
			neg_value_x0 = neg_minimizer(x0).numpy()
			cns_l1_value_x0 = cns_l1_minimizer(x0).numpy()
			cns_tv_value_x0 = cns_tv_minimizer(x0).numpy()
			l1_ratio = l1_value / l1_value_x0 if l1_value_x0 != 0 else float('inf')
			tv_ratio = tv_value / tv_value_x0 if tv_value_x0 != 0 else float('inf')
			neg_ratio = neg_value / neg_value_x0 if neg_value_x0 != 0 else float('inf')
			cns_l1_ratio = cns_l1_value / cns_l1_value_x0 if cns_l1_value_x0 != 0 else float('inf')
			cns_tv_ratio = cns_tv_value / cns_tv_value_x0 if cns_tv_value_x0 != 0 else float('inf')
			if self.weight_l1 > 0:
				report += f", L1: {l1_value:.2e} (Ratio: {l1_ratio:.3f})"
			if self.weight_tv > 0:
				report += f", TV: {tv_value:.2e} (Ratio: {tv_ratio:.3f})"
			if self.weight_neg > 0:
				report += f", NEG: {neg_value:.2e} (Ratio: {neg_ratio:.3f})"
			if self.weight_cns_l1 > 0:
				report += f", CNS L1: {cns_l1_value:.2e} (Ratio: {cns_l1_ratio:.3f})"
			if self.weight_cns_tv > 0:
				report += f", CNS TV: {cns_tv_value:.2e} (Ratio: {cns_tv_ratio:.3f})"
			report += "."
		else:
			if self.weight_l1 > 0:
				report += f", L1: {l1_value:.2e}"
			if self.weight_tv > 0:
				report += f", TV: {tv_value:.2e}"
			if self.weight_neg > 0:
				report += f", NEG: {neg_value:.2e}"
			if self.weight_cns_l1 > 0:
				report += f", CNS L1: {cns_l1_value:.2e}"
			if self.weight_cns_tv > 0:
				report += f", CNS TV: {cns_tv_value:.2e}"
			report += "."
		return report

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

gamma=None
if ARG.gamma is not None:
	gamma = DEN.getFrame(ARG.gamma, 0)

basisVec = createFitVec(ARG.inputFitBasis, basisSize)

if alpha is not None:
	fitVec = createFitVec(ARG.inputFitBasis, basisSize, alpha)
else:
	fitVec = basisVec

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
if ARG.report_transmission is not None:
	DEN.writeEmptyDEN(ARG.report_transmission, [inf_fit["dimspec"][0], inf_fit["dimspec"][1], inf_img["dimspec"][2]], elementtype=np.dtype("<f4"), force=ARG.force)
if ARG.report_extinction is not None:
	DEN.writeEmptyDEN(ARG.report_extinction, [inf_fit["dimspec"][0], inf_fit["dimspec"][1], inf_img["dimspec"][2]], elementtype=np.dtype("<f4"), force=ARG.force)

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

x_zero = np.zeros(len(fitScale), dtype=np.float32)
x_zero[0] = 1.0
x0 = x_zero


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

np.set_printoptions(precision=2)

with tf.device('/cpu:0'):
	FFOVERB = None
	for k in np.arange(frameCount):
		b = DEN.getFrame(ARG.inputImg, k)
		#AGM = AlphaGammaMinimizer(b, basisVec, fitScale, alpha, gamma, weight_l1=0.25, weight_tv=1, dtype=tf.float32)
		#AGM = AlphaGammaMinimizer(b, basisVec, fitScale, alpha, gamma, weight_l1=1, weight_tv=500, dtype=tf.float32)
		if FFOVERB is not None:
			AGM = AlphaGammaConsecutiveMinimizer(b, basisVec, FFOVERB, initScale=fitScale, alpha=alpha, gamma=gamma, weight_l1=1, weight_tv=0, weight_neg=0, weight_cns_l1=1, weight_cns_tv=0, dtype=tf.float32)
		#	AGM.reportGammaState(x0)
		#	exit()
		else:
			AGM = AlphaGammaMinimizer(b, basisVec, fitScale, alpha, gamma, weight_l1=1, weight_tv=0, weight_neg=0, dtype=tf.float32)
		minimizer = AGM.getMinimizer()
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
			print("Frame k=%d, fitScale=%s, x_scaled=%s, x=%s"%(k, fitScale, xscaled,  AGM.reportMinimizerValue(x, x_zero)))
			#print("Fit for k=%d x=%s xscaled=%s fit initial=%.2e after=%.2e ratio = %.3f"%(k, x, xscaled, fitval_init, fitval_loss, fitval_loss/fitval_init))
			#plt.plot(losses)
		out = np.tensordot(xscaled, basisVec, axes=[0,0])
		DEN.writeFrame(ARG.outputExt, k,  out, force=True)
		FFOVERB = out / b
		if ARG.report_transmission is not None:
			out = np.maximum(out, 1e-5)
			tra = b / out
			DEN.writeFrame(ARG.report_transmission, k, tra, force=True)
		if ARG.report_extinction is not None:
			ratio = out / b
			ratio = np.maximum(ratio, 1e-5) #Avoid log(0)
			ext = np.log(ratio)
			DEN.writeFrame(ARG.report_extinction, k, ext, force=True)
