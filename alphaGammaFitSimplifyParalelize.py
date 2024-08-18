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
import sys
import time
import termcolor
from functools import reduce
import multiprocessing as mp

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
parser.add_argument("--report-first-extinction", type=str, help="First extinction reporting.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--lbfgs", action="store_true")
parser.add_argument("--fit-slope", action="store_true")
parser.add_argument("--first-fit", type=str, default=None, help="Object to do the first fit with to be used for subsecutive fits.")
parser.add_argument("--first-fit-size", type=int, default=None, help="Size of the basis for the first fit.")
nonlinearity = parser.add_mutually_exclusive_group()
nonlinearity.add_argument("--tv", action="store_true", help="Total variation fit")
nonlinearity.add_argument("--l1", action="store_true", help="L1 fit")

ARG = parser.parse_args()

# Get the number of available CPU cores
num_threads = mp.cpu_count()

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = str(num_threads)
# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


inf_img = DEN.readHeader(ARG.inputImg)
inf_fit = DEN.readHeader(ARG.inputFitBasis)
if inf_fit["dimspec"][0] != inf_img["dimspec"][0] or inf_fit["dimspec"][1] != inf_img["dimspec"][1]:
	raise ArgumentException("Incompatible dimensions of input files")
frameCount = inf_img["dimspec"][2]

basisSize = ARG.basis_size
if ARG.basis_size > 0:
	if inf_fit["dimspec"][2] < ARG.basis_size:
		raise ValueError("ARG.basis_size=%d not in %s"%(ARG.basis_size, ARG.inputFitBasis))
else:
	basisSize = inf_fit["dimspec"][3]

def softplus(x): 
	return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# Function to compute anisotropic total variation with precomputed masks
def masked_anisotropic_total_variation(image, mask_x, mask_y):
	# Compute differences in the x direction
	dx = image[:, 1:] - image[:, :-1]
	dx = tf.where(mask_x, dx, tf.zeros_like(dx))
	# Compute differences in the y direction
	dy = image[1:, :] - image[:-1, :]
	dy = tf.where(mask_y, dy, tf.zeros_like(dy))
	# Compute anisotropic total variation
	tv = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
	return tv

# Function to compute isotropic total variation with precomputed masks
def masked_isotropic_total_variation(image, mask_x, mask_y):
	# Compute differences in the x direction
	dx = image[:, 1:] - image[:, :-1]
	dx = tf.where(mask_x, dx, tf.zeros_like(dx))
	# Compute differences in the y direction
	dy = image[1:, :] - image[:-1, :]
	dy = tf.where(mask_y, dy, tf.zeros_like(dy))
	# Compute isotropic total variation
	tv = tf.sqrt(tf.reduce_sum(tf.square(dx)) + tf.reduce_sum(tf.square(dy)))
	return tv

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
		self.reduce_k = 5
		self.numpyshape = b.shape
		self.numpyshape_reduced = (b.shape[0] // self.reduce_k, b.shape[1] // self.reduce_k)
		self.weight_l1 = weight_l1
		self.weight_neg = weight_neg
		self.weight_tv = weight_tv
		self.dtype = dtype
		self.n = fitVec.shape[0]
		self.initScale = initScale
#Copy relevant arrays
		self.b = copy.deepcopy(b)
		self.b_reduced = reduce_array_by_factor(self.b, self.reduce_k)
		self.fitVec = copy.deepcopy(fitVec)
		self.alpha = np.ones_like(b) if alpha is None else copy.deepcopy(alpha)
		self.gamma = np.ones_like(b) if gamma is None else copy.deepcopy(gamma)
		self.gamma_reduced = reduce_array_by_factor(self.gamma, self.reduce_k)
		self.fitVec_reduced = np.zeros((self.n, ) + self.numpyshape_reduced, dtype=np.float32)
		for i in range(self.n):
			v = fitVec[i]
			v_reduced = reduce_array_by_factor(v, self.reduce_k)
			if initScale is not None:
				self.fitVec[i] = initScale[i] * v
				self.fitVec_reduced[i] = initScale[i] * v_reduced
			else:
				self.fitVec[i] = v
				self.fitVec_reduced[i] = v_reduced
		self.ones = np.ones(self.numpyshape, dtype=np.float32)
		self.zeros = np.zeros(self.numpyshape, dtype=np.float32)
		self.softzeros = self.ones * np.log(2) # Softplus(0) = log(2)
		self.weight_l1 = weight_l1
		self.weight_neg = weight_neg
		self.weight_tv = weight_tv
		self.alpha = self.alpha > 0
		self.gamma = self.gamma > 0
		self.gamma_reduced = self.gamma_reduced > 0
		self.gamma_x_mask = tf.constant(self.gamma[:,1:] & self.gamma[:,:-1], dtype=tf.bool)
		self.gamma_y_mask = tf.constant(self.gamma[1:,:] & self.gamma[:-1,:], dtype=tf.bool)
		self.gamma_reduced_x_mask = tf.constant(self.gamma_reduced[:,1:] & self.gamma_reduced[:,:-1], dtype=tf.bool)
		self.gamma_reduced_y_mask = tf.constant(self.gamma_reduced[1:,:] & self.gamma_reduced[:-1,:], dtype=tf.bool)
	
	def getTVTensorReduced(self, B, FF):
		#The following does not converge
		#TVI = tf.math.subtract(tf.math.log(tf.math.softplus(FF)), tf.math.log(B))
		#The symmetric sum does not converge
		#TVI = tf.math.divide_no_nan(tf.math.softplus(FF), B) + tf.math.divide_no_nan(B, tf.math.softplus(FF))
		#The following converges but is not symmetric
		#TVI = tf.math.divide_no_nan(tf.math.softplus(FF), B)
		#The following converges as well
		TVI = tf.math.divide(FF, B) + tf.math.divide(B, FF)
		return self.weight_tv * masked_isotropic_total_variation(TVI, self.gamma_reduced_x_mask, self.gamma_reduced_y_mask)

	def getTVTensor(self, B, FF):
		#The following does not converge
		#TVI = tf.math.subtract(tf.math.log(tf.math.softplus(FF)), tf.math.log(B))
		#The symmetric sum does not converge
		#TVI = tf.math.divide_no_nan(tf.math.softplus(FF), B) + tf.math.divide_no_nan(B, tf.math.softplus(FF))
		#The following converges but is not symmetric
		#TVI = tf.math.divide_no_nan(tf.math.softplus(FF), B)
		#The following converges as well
		TVI = tf.math.divide(FF, B) + tf.math.divide(B, FF)
		return self.weight_tv * masked_isotropic_total_variation(TVI, self.gamma_x_mask, self.gamma_y_mask)

	def getL1Tensor(self, B, FF):
		#ones_tf = tf.constant(self.ones, dtype=self.dtype)
		#softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, B))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(B, FF), ones_tf)), softzeros_tf)
		dif_l1 = tf.math.subtract(FF, B)
		return self.weight_l1 * tf.reduce_sum(tf.abs(tf.boolean_mask(dif_l1, self.alpha)))

	def getGammaNegativeTensor(self, B, FF):
		zeros_tf = tf.constant(self.zeros, dtype=self.dtype)
		neg_l2 = tf.math.maximum(tf.math.subtract(B, FF), zeros_tf)
		return self.weight_neg * tf.math.sqrt(tf.reduce_sum(tf.square(tf.boolean_mask(neg_l2, self.gamma))))

	def getMinimizer(self):
		def minimizer(x):
			b_tf = tf.constant(self.b, dtype=self.dtype)
			b_reduced_tf = tf.constant(self.b_reduced, dtype=self.dtype)
			ff = reduce(tf.add, (x[i] * tf.constant(self.fitVec[i], dtype=self.dtype) for i in range(self.n)))
			ff_reduced = reduce(tf.add, (x[i] * tf.constant(self.fitVec_reduced[i], dtype=self.dtype) for i in range(self.n)))
			ret = 0
			if self.weight_l1 > 0:
				ret = self.getL1Tensor(b_tf, ff)
			if self.weight_tv > 0:
				ret += self.getTVTensorReduced(b_reduced_tf, ff_reduced)
			if self.weight_neg > 0:
				ret += self.getGammaNegativeTensor(b_tf, ff)
			return ret
		return minimizer

	def getL1Minimizer(self):
		def l1_minimizer(x):
			b_tf = tf.constant(self.b, dtype=self.dtype)
			ff = x[0] * tf.constant(self.fitVec[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff += x[k] * tf.constant(self.fitVec[k], dtype=self.dtype)
			return self.getL1Tensor(b_tf, ff)
		return l1_minimizer

	def getTVMinimizer(self):
		def tv_minimizer(x):
			b_reduced_tf = tf.constant(self.b_reduced, dtype=self.dtype)
			ff_reduced = x[0] * tf.constant(self.fitVec_reduced[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff_reduced += x[k] * tf.constant(self.fitVec_reduced[k], dtype=self.dtype)
			return self.getTVTensorReduced(b_reduced_tf, ff_reduced)
		return tv_minimizer
	
	def getNegMinimizer(self):
		def neg_minimizer(x):
			b_tf = tf.constant(self.b, dtype=self.dtype)
			ff = x[0] * tf.constant(self.fitVec[0], dtype=self.dtype)
			for k in range(1, self.n):
				ff += x[k] * tf.constant(self.fitVec[k], dtype=self.dtype)
			return self.getGammaNegativeTensor(b_tf, ff)
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
	#I expect imputs are softplus(b), softplus(ff_prev), ...
	def __init__(self, b, fitVec, ff_prev, b_prev, initScale=None, alpha=None, gamma=None, weight_l1=1, weight_tv=1, weight_neg=1, weight_cns_l1=1, weight_cns_tv=1, dtype=tf.float32):
		self.numpyshape = b.shape
		self.weight_l1 = weight_l1
		self.weight_neg = weight_neg
		self.weight_tv = weight_tv
		self.weight_cns_l1 = weight_cns_l1
		self.weight_cns_tv = weight_cns_tv
		self.dtype = dtype
		self.n = fitVec.shape[0]
		self.initScale = initScale
#Copy relevant arrays
		self.b = b
		self.fitVec = copy.deepcopy(fitVec)
		self.alpha = np.ones_like(b) if alpha is None else copy.deepcopy(alpha)
		self.gamma = np.ones_like(b) if gamma is None else copy.deepcopy(gamma)
		if initScale is not None:
			for i in range(self.n):
				v = fitVec[i]
				self.fitVec[i] = initScale[i] * v
		#self.ones = np.ones(self.numpyshape, dtype=np.float32)
		#self.zeros = np.zeros(self.numpyshape, dtype=np.float32)
		#self.softzeros = self.ones * np.log(2) # Softplus(0) = log(2)
		self.alpha = self.alpha > 0
		self.gamma = self.gamma > 0
		self.ff_prev = ff_prev
		self.b_prev = b_prev
		self.ext_prev = np.log(np.divide(ff_prev, b_prev))
		self.tf_b_alpha = tf.constant(self.b[self.alpha], dtype=self.dtype)
		self.tf_b_gamma = tf.constant(self.b[self.gamma], dtype=self.dtype)
		self.tf_ext_prev_gamma = tf.constant(self.ext_prev[self.gamma], dtype=self.dtype)
		self.tf_fitVec_alpha = [tf.constant(self.fitVec[k][self.alpha], dtype=self.dtype) for k in range(self.n)]
		self.tf_fitVec_gamma = [tf.constant(self.fitVec[k][self.gamma], dtype=self.dtype) for k in range(self.n)]
		if weight_tv > 0:
			self.reduce_k = 5
			self.b_reduced = reduce_array_by_factor(self.b, self.reduce_k)
			self.numpyshape_reduced = (b.shape[0] // self.reduce_k, b.shape[1] // self.reduce_k)
			self.gamma_reduced = self.gamma_reduced > 0
			self.gamma_x_mask = tf.constant(self.gamma[:,1:] & self.gamma[:,:-1], dtype=tf.bool)
			self.gamma_y_mask = tf.constant(self.gamma[1:,:] & self.gamma[:-1,:], dtype=tf.bool)
			self.gamma_reduced_x_mask = tf.constant(self.gamma_reduced[:,1:] & self.gamma_reduced[:,:-1], dtype=tf.bool)
			self.gamma_reduced_y_mask = tf.constant(self.gamma_reduced[1:,:] & self.gamma_reduced[:-1,:], dtype=tf.bool)
			self.gamma_reduced = reduce_array_by_factor(self.gamma, self.reduce_k)
			self.fitVec_reduced = np.zeros((self.n, ) + self.numpyshape_reduced, dtype=np.float32)
			for i in range(self.n):
				v = fitVec[i]
				v_reduced = reduce_array_by_factor(v, self.reduce_k)
				if initScale is not None:
					self.fitVec_reduced[i] = initScale[i] * v_reduced
				else:
					self.fitVec_reduced[i] = v_reduced
	
	def updateB(self, b):
		self.b = b
		self.tf_b_alpha = tf.constant(self.b[self.alpha], dtype=self.dtype)
		self.tf_b_gamma = tf.constant(self.b[self.gamma], dtype=self.dtype)

	def getL1Tensor(self, B, FF):
		#ones_tf = tf.constant(self.ones, dtype=self.dtype)
		#softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, B))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(B, FF), ones_tf)), softzeros_tf)
		#dif_l1 = tf.math.subtract(FF, B)
		#return self.weight_l1 * tf.reduce_sum(tf.abs(dif_l1))
		dif_l1 = tf.math.log(tf.math.divide(tf.math.softplus(FF), B))
		return self.weight_l1 * tf.reduce_sum(tf.abs(dif_l1))
	
	def getL1Minimizer(self):
		def l1_minimizer(x):
			ff_alpha = reduce(tf.add, (x[i] * self.tf_fitVec_alpha[i] for i in range(self.n)))
			return self.getL1Tensor(self.tf_b_alpha, ff_alpha)
		return l1_minimizer
	
	def getL1CnsTensor(self, EXTPREV, B, FF):
		#EXTPREV = tf.boolean_mask(EXTPREV, self.gamma)
		#B = tf.boolean_mask(B, self.gamma)
		#FF = tf.boolean_mask(FF, self.gamma)
		#The following pushes left tail of ext down
		#dif_l1 = tf.math.subtract(PX, tf.math.divide_no_nan(FF, B))
		#Mostly both will be larger than one, so let's try with logarithm
		dif_b = tf.math.log(tf.math.divide(tf.math.softplus(FF), B))
		dif_l1 = tf.math.subtract(EXTPREV, dif_b)
		#ones_tf = tf.constant(self.ones, dtype=self.dtype)
		#softzeros_tf = tf.constant(self.softzeros, dtype=self.dtype)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(ones_tf, tf.math.divide_no_nan(FF, PF))), softzeros_tf)
		#dif_l1 = tf.math.subtract(tf.math.softplus(tf.math.subtract(tf.math.divide_no_nan(PF, FF), ones_tf)), softzeros_tf)
		return self.weight_cns_l1 * tf.reduce_sum(tf.abs(dif_l1))
	
	def getCnsL1Minimizer(self):
		def cns_l1_minimizer(x):
			ff_gamma = reduce(tf.add, (x[i] * self.tf_fitVec_gamma[i] for i in range(self.n)))
			return self.getL1CnsTensor(self.tf_ext_prev_gamma, self.tf_b_gamma, ff_gamma)
		return cns_l1_minimizer

	def getTVCnsTensor(self, EXTPREV, B, FF):
		#dif_tv = tf.math.divide_no_nan(FF, PF)
		#dif_tv = tf.math.log(tf.math.softplus(PF)) - tf.math.log(tf.math.softplus(FF))
		#dif_tv = tf.math.subtract(EXTPREV, tf.math.log(tf.math.divide_no_nan(tf.math.softplus(FF), B)))
		dif_tv = tf.math.subtract(EXTPREV, FF)
		return self.weight_cns_tv * masked_anisotropic_total_variation(dif_tv, self.gamma_x_mask, self.gamma_y_mask)
	
	def getTVTensor(self, B, FF):
		#dif_tv = tf.math.divide_no_nan(FF, B)
		#dif_tv = tf.math.log(tf.math.softplus(B)) - tf.math.log(tf.math.softplus(FF))
		TVI = tf.math.divide(FF, B) + tf.math.divide(B, FF)
		return self.weight_tv * masked_isotropic_total_variation(TVI, self.gamma_reduced_x_mask, self.gamma_reduced_y_mask)

	def getGammaNegativeTensor(self, B, FF):
		zeros_tf = tf.zeros_like(B)
		neg_l2 = tf.math.maximum(tf.math.subtract(B, FF), zeros_tf)
		return self.weight_neg * tf.math.sqrt(tf.reduce_sum(tf.square(neg_l2)))

	def getMinimizer(self):
		def minimizer(x):
			ret = 0.0
			if self.weight_l1 > 0:
				ff_alpha = reduce(tf.add, (x[i] * self.tf_fitVec_alpha[i] for i in range(self.n)))
				ret += self.getL1Tensor(self.tf_b_alpha, ff_alpha)
			if self.weight_cns_l1 > 0:
				ff_gamma = reduce(tf.add, (x[i] * self.tf_fitVec_gamma[i] for i in range(self.n)))
				ret += self.getL1CnsTensor(self.tf_ext_prev_gamma, self.tf_b_gamma, ff_gamma)
			if self.weight_tv > 0:
				b_reduced_tf = tf.constant(self.b_reduced, dtype=self.dtype)
				ff_reduced_tf = reduce(tf.add, (x[i] * tf.constant(self.fitVec_reduced[i], dtype=self.dtype) for i in range(self.n)))
				ret += self.getTVTensor(b_reduced_tf, ff_reduced_tf)
			if self.weight_neg > 0:
				b_gamma = tf.constant(self.b[self.gamma], dtype=self.dtype)
				ff_gamma = reduce(tf.add, (x[i] * self.tf_fitVec_gamma[i] for i in range(self.n)))
				ret += self.getGammaNegativeTensor(b_gamma, ff_gamma)
			if self.weight_cns_tv > 0:
				ext_prev_tf = tf.constant(self.ext_prev, dtype=self.dtype)
				b_tf = tf.constant(self.b, dtype=self.dtype)
				ff = reduce(tf.add, (x[i] * tf.constant(self.fitVec[i], dtype=self.dtype) for i in range(self.n)))
				#ret += self.getTVCnsTensor(ext_prev_tf, b_tf, ff)
				#Try this
				ff_prev_tf = tf.constant(self.ff_prev, dtype=self.dtype)
				ret += self.getTVCnsTensor(ff_prev_tf, b_tf, ff)
			return ret
		return minimizer


	def getTVMinimizer(self):
		def tv_minimizer(x):
			b_reduced_tf = tf.constant(self.b_reduced, dtype=self.dtype)
			ff_reduced_tf = reduce(tf.add, (x[i] * tf.constant(self.fitVec_reduced[i], dtype=self.dtype) for i in range(self.n)))
			return self.getTVTensor(b_reduced_tf, ff_reduced_tf)
		return tv_minimizer
	
	def getNegMinimizer(self):
		def neg_minimizer(x):
			b_gamma = tf.constant(self.b[self.gamma], dtype=self.dtype)
			ff_gamma = reduce(tf.add, (x[i] * self.tf_fitVec_gamma[i] for i in range(self.n)))
			return self.getGammaNegativeTensor(b_gamma, ff_gamma)
		return neg_minimizer


	def reportState(self):
		DEN.storeNdarrayAsDEN("~/tmp/AGF_ffoverb_prev_gamma.den", self.ffoverb_prev_gamma, force=True)
	
	def reportGammaState(self, x):
		DEN.storeNdarrayAsDEN("AGF_ext_prev.den", self.ext_prev, force=True)
		DEN.storeNdarrayAsDEN("AGF_b.den", self.b, force=True)
		ffg = x[0] * self.fitVec[0]
		for k in range(1, self.n):
			ffg += x[k] * self.fitVec[k]
		for k in range(self.n):
			DEN.storeNdarrayAsDEN("AGF_fitVec_%d.den"%k, self.fitVec[k], force=True)
		DEN.storeNdarrayAsDEN("AGF_ff_x.den", ffg, force=True)

	def getCnsTVMinimizer(self):
		def cns_tv_minimizer(x):
			ext_prev_tf = tf.constant(self.ext_prev, dtype=self.dtype)
			b_tf = tf.constant(self.b, dtype=self.dtype)
			ff = reduce(tf.add, (x[i] * tf.constant(self.fitVec[i], dtype=self.dtype) for i in range(self.n)))
			return self.getTVCnsTensor(ext_prev_tf, b_tf, ff)
		return cns_tv_minimizer

	def reportMinimizerValue(self, x, x0=None):
		if self.weight_l1 > 0:
			l1_minimizer = self.getL1Minimizer()
			l1_value = l1_minimizer(x).numpy()
		if self.weight_tv > 0:
			tv_minimizer = self.getTVMinimizer()
			tv_value = tv_minimizer(x).numpy()
		if self.weight_neg > 0:
			neg_minimizer = self.getNegMinimizer()
			neg_value = neg_minimizer(x).numpy()
		if self.weight_cns_l1 > 0:
			cns_l1_minimizer = self.getCnsL1Minimizer()
			cns_l1_value = cns_l1_minimizer(x).numpy()
		if self.weight_cns_tv > 0:
			cns_tv_minimizer = self.getCnsTVMinimizer()
			cns_tv_value = cns_tv_minimizer(x).numpy()
		report = f"x: {x}\n"
		if x0 is not None:
			if self.weight_l1 > 0:
				l1_value_x0 = l1_minimizer(x0).numpy()
				l1_ratio = l1_value / l1_value_x0 if l1_value_x0 != 0 else float('inf')
				report += f", L1: {l1_value:.2e} (Ratio: {l1_ratio:.3f})"
			if self.weight_tv > 0:
				tv_value_x0 = tv_minimizer(x0).numpy()
				tv_ratio = tv_value / tv_value_x0 if tv_value_x0 != 0 else float('inf')
				report += f", TV: {tv_value:.2e} (Ratio: {tv_ratio:.3f})"
			if self.weight_neg > 0:
				neg_value_x0 = neg_minimizer(x0).numpy()
				neg_ratio = neg_value / neg_value_x0 if neg_value_x0 != 0 else float('inf')
				report += f", NEG: {neg_value:.2e} (Ratio: {neg_ratio:.3f})"
			if self.weight_cns_l1 > 0:
				cns_l1_value_x0 = cns_l1_minimizer(x0).numpy()
				cns_l1_ratio = cns_l1_value / cns_l1_value_x0 if cns_l1_value_x0 != 0 else float('inf')
				report += f", CNS L1: {cns_l1_value:.2e} (Ratio: {cns_l1_ratio:.3f})"
			if self.weight_cns_tv > 0:
				cns_tv_ratio = cns_tv_value / cns_tv_value_x0 if cns_tv_value_x0 != 0 else float('inf')
				cns_tv_value_x0 = cns_tv_minimizer(x0).numpy()
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


def compute_lbfgs(minimizer, x0, max_iterations=50, parallel_iterations=1):
	def value_and_gradients_function(x):
		loss, grads = get_value_and_grads(minimizer, x)
		return loss, grads
	init_position = tf.Variable(x0, dtype=tf.float32)
	opt = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=value_and_gradients_function,initial_position=init_position, max_iterations=max_iterations, parallel_iterations=parallel_iterations)
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

def formatTime(t_dif):
	t_dif = int(t_dif)
	hours = t_dif // 3600
	minutes = (t_dif % 3600) // 60
	seconds = t_dif % 60
	if hours == 0:
		if minutes == 0:
			return "%.2fs"%seconds
		else:
			return "%02dm %02ds"%(minutes, seconds)
	else:
		return "%02dh %02dm %02ds"%(hours, minutes, seconds)

def report_time(start_time, start_time_k, end_time_k, k, frameCount):
	elapsed_time = end_time_k - start_time
	elapsed_time_k = end_time_k - start_time_k
	remaining_time = elapsed_time / (k + 1) * (frameCount - k - 1)
	print("Frame %d/%d, elapsed time %s, elapsed time for frame %s, estimated remaining time %s"%(k, frameCount, formatTime(elapsed_time), formatTime(elapsed_time_k), formatTime(remaining_time)))

start_time = time.time()

ff_first = None
b_first = None

print("Scaling estimates for first fit based on b0 img: %s"%fitScale)
#Experimental code
if ARG.first_fit is not None:
	b0 = DEN.getFrame(ARG.first_fit, 0)
	if alpha is not None:
		b0 = np.multiply(b0, alpha)
	#fitScale = np.ones(2*basisSize, dtype=np.float32)
	fitScale = getScalingEstimates(b0, fitVec)
	print("Scaling estimates for first fit based on first fit img: %s"%fitScale)
#End experimental code

x_zero = np.zeros(len(fitScale), dtype=np.float32)
x_zero[0] = 1.0
x0 = x_zero

if ARG.report_first_extinction is not None:
	DEN.writeEmptyDEN(ARG.report_first_extinction, [inf_fit["dimspec"][0], inf_fit["dimspec"][1], 5+fitVec.shape[0]], elementtype=np.dtype("<f4"), force=ARG.force)

with tf.device('/cpu:0'):
	if ARG.first_fit is not None:
		#First we compute scaling estimates for the fit
		b_avg = DEN.getFrame(ARG.first_fit, 0)
		b_avg_alpha = b_avg * alpha
		b_avg_alpha_tensordot = np.tensordot(b_avg_alpha, b_avg_alpha)
		p = np.tensordot(b_avg_alpha, basisVec[0])
		x0 = np.zeros(basisVec.shape[0], dtype=np.float32)
		x0[0] = b_avg_alpha_tensordot / p
		print("Estimate for x0=%s"%x0)
		#x0 = np.zeros(basisVec.shape[0], dtype=np.float32)
		#x0[0] = 1.0
		AGM = AlphaGammaMinimizer(b0, basisVec, np.ones(basisVec.shape[0]), alpha, gamma, weight_l1=1, weight_tv=0, weight_neg=0, dtype=tf.float32)
		minimizer = AGM.getMinimizer()
		if ARG.lbfgs:
			opt = compute_lbfgs(minimizer, x0, max_iterations=500)
			x = opt.position.numpy().astype(np.float32)
			if not opt.converged:
				#print red
				print(termcolor.colored("ERROR: LBFGS optimization of scaling estimates did not converge", "red"))
				if not np.any(np.isnan(x)):
					x, losses = compute_adam(minimizer, x)
					x = x.numpy()
			if np.any(np.isnan(x)):
				x, losses = compute_adam(minimizer, x0)
				x = x.numpy()
		else:
			x, losses = compute_adam(minimizer, x0)
			x = x.numpy()
		fitScale = x
		if ARG.verbose:
			b0_alpha = DEN.getFrame(ARG.inputImg, 0)*alpha
			print("Initial fitScale=%s, compared to b_avg_estimates=%s, b0_estimates=%s"%(fitScale,  getScalingEstimates(b_avg_alpha, fitVec), getScalingEstimates(b0_alpha, fitVec)))

	#Compute properly scaled fit vector
	scaledBasisVec = np.zeros_like(basisVec)
	for i in np.arange(basisVec.shape[0]):
		scaledBasisVec[i] = fitScale[i] * basisVec[i]

	#Then I compute actual fit w.r.t. b_avg
	firstFitSize = basisVec.shape[0]
	if ARG.first_fit_size is not None:
		firstFitSize = ARG.first_fit_size
	print("Computing first fit with first_fit_size=%s"%(ARG.first_fit_size))
	AGM = AlphaGammaMinimizer(b_avg, scaledBasisVec[:firstFitSize], None, alpha, gamma, weight_l1=1, weight_tv=0, weight_neg=0, dtype=tf.float32)
	x0 = np.zeros(firstFitSize, dtype=np.float32)
	x0[0] = 1.0
	minimizer = AGM.getMinimizer()
	if ARG.lbfgs:
		opt = compute_lbfgs(minimizer, x0, max_iterations=100, parallel_iterations=5)
		x = opt.position.numpy().astype(np.float32)
		if not opt.converged:
			print("ERROR: LBFGS optimization did not converge")
		if np.any(np.isnan(x)):
			x, losses = compute_adam(minimizer, x0)
			x = x.numpy()
		elif not opt.converged:
			x, losses = compute_adam(minimizer, x)
			x = x.numpy()
	else:
		x, losses = compute_adam(minimizer, x0[:firstFitSize])
		x = x.numpy()
	x_zero = x
	if ARG.verbose:
		print("Initial fit=%s"%(AGM.reportMinimizerValue(x, x0)))
	ff_first = softplus(np.tensordot(x_zero, scaledBasisVec[:firstFitSize], axes=[0,0]))
	EXT = np.log(ff_first / b_avg)
	if ARG.report_first_extinction is not None:
		DEN.writeFrame(ARG.report_first_extinction, 0, EXT, force=True)
		DEN.writeFrame(ARG.report_first_extinction, 1, b_avg, force=True)
		DEN.writeFrame(ARG.report_first_extinction, 2, ff_first, force=True)
		for i in np.arange(len(scaledBasisVec)):
			DEN.writeFrame(ARG.report_first_extinction, 3+i, scaledBasisVec[i], force=True)
		DEN.writeFrame(ARG.report_first_extinction, 3+len(scaledBasisVec), gamma, force=True)
		DEN.writeFrame(ARG.report_first_extinction, 3+len(scaledBasisVec)+1, alpha, force=True)

	#Finally actual fitting
	b = DEN.getFrame(ARG.inputImg, 0)#b shall be itself softplus no need to do it here
	AGM = AlphaGammaConsecutiveMinimizer(b, scaledBasisVec, ff_first, b_avg, initScale=None, alpha=alpha, gamma=gamma, weight_l1=1, weight_tv=0, weight_neg=0, weight_cns_l1=1, weight_cns_tv=0, dtype=tf.float32)
	x0 = np.zeros(len(scaledBasisVec), dtype=np.float32)
	for i in np.arange(len(x_zero)):
		x0[i] = x_zero[i]
	for k in np.arange(frameCount):
		#print("Computing k=%d"%k)
		start_time_k = time.time()
		b = DEN.getFrame(ARG.inputImg, k)#b shall be itself softplus no need to do it here
		AGM.updateB(b)
		minimizer = AGM.getMinimizer()
		converged=True
		if ARG.lbfgs:
			opt = compute_lbfgs(minimizer, x0, max_iterations=100)
			x = opt.position.numpy().astype(np.float32)
			if not opt.converged:
				print(termcolor.colored("ERROR: k=%d LBFGS optimization did not converge"%k, "red"))
				print("ERROR: k=%d LBFGS optimization did not converge"%k, file=sys.stderr)
				converged=False
			if np.any(np.isnan(x)):
				x, losses = compute_adam(minimizer, x0)
				x = x.numpy()
		else:
			x, losses = compute_adam(minimizer, x0)
			x = x.numpy()
		if ARG.verbose:
			print("Frame k=%d, x=%s"%(k, AGM.reportMinimizerValue(x, x0)), flush=True)
		else:
			if k % 100 == 0:
				print("Frame k=%d, x=%s"%(k, x), flush=True)
		#softplus(-37)==0.0, so we need to avoid this
		#softplus(softplus(33))=33 still good regularization
		ff_now = softplus(softplus(np.tensordot(x, scaledBasisVec, axes=[0,0])))
		DEN.writeFrame(ARG.outputExt, k,  ff_now, force=True)
		if ARG.report_transmission is not None:
			tra = b / ff_now
			DEN.writeFrame(ARG.report_transmission, k, tra, force=True)
		if ARG.report_extinction is not None:
			ext = np.log(ff_now / b)
			DEN.writeFrame(ARG.report_extinction, k, ext, force=True)
		end_time_k = time.time()
		report_time(start_time, start_time_k, end_time_k, k, frameCount)
