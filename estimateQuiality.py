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
from skimage import feature
from skimage import filters
from skimage.filters import meijering, sato, frangi, hessian
from matplotlib import pyplot as plt
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


class JsonCache:
	def __init__(self, path=None):
		self.entries = {}
		if path is not None:
			self.path = path
		if os.path.exists(path):
			self.load(path)

	def _make_key(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		return (
			int(sinogram_ind),
			round(float(cor), 6),
			str(reconstruction),
			bool(apply_log),
		)

	def load(self, path):
		path = Path(path)
		with open(path, "r") as f:
			raw = json.load(f)
		self.entries = {}
		for entry in raw:
			key = self._make_key(
				sinogram_ind=entry["sinogram_ind"],
				cor=entry["cor"],
				reconstruction=entry["reconstruction"],
				apply_log=entry["apply_log"],
			)
			self.entries[key] = entry

	def save(self, path = None):
		if path is None:
			path = self.path
		if path is None:
			log.warning("No path specified for saving cache.")
			return
		path = Path(path)
		with open(path, "w") as f:
			json.dump(
				list(self.entries.values()),
				f,
				indent=2,
			)

	def get_entry(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
		create=False,
	):
		key = self._make_key(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		entry = self.entries.get(key)
		if entry is None and create:
			entry = {
				"sinogram_ind": int(sinogram_ind),
				"cor": float(cor),
				"reconstruction": reconstruction,
				"apply_log": apply_log,
				"scores": {},
			}
			self.entries[key] = entry
		return entry

	def get_json(self):
		return list(self.entries.values())

	def get_score(
		self,
		method,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		if entry is None:
			return None
		return entry["scores"].get(method)

	def set_score(
		self,
		method,
		value,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		entry["scores"][method] = value

	def get_scores(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		return entry["scores"]

parser = argparse.ArgumentParser()
parser.add_argument("inputSinograms")
parser.add_argument("--frame-index", type=int, default=None)
parser.add_argument("--refinement-steps", type=int, default=10)
parser.add_argument("--point-count", type=int, default=5)
parser.add_argument("--pixel-window", type=int, default=None)
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
cacheFile = ARG.inputSinograms + ".cor_estimation_cache.json"
cache = JsonCache(cacheFile)

#We use method from paper https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6492/1/The-blur-effect--perception-and-estimation-with-a-new/10.1117/12.702790.full
#Consider using other metrics https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry
#Also https://ieeexplore.ieee.org/abstract/document/8326697/figures#figures
def compute_scores(f, angles, cor_points, cache=None, apply_log=False, verbose=True):
	results = []
	center = f.shape[1] / 2 - 0.5
	offsets = cor_points - center
	print(f"Testing {len(offsets)} in range {offsets[0]} → {offsets[-1]}")
	for cor in cor_points:
		# Get/create cache entry
		scores_cache = (
			cache.get_scores(sinogram_ind=ARG.frame_index, cor=cor, reconstruction="dfi", apply_log=apply_log)
			if cache is not None else {}
		)
		score3 = scores_cache.get("blur_effect_3")
		score11 = scores_cache.get("blur_effect_11")
		score30 = scores_cache.get("blur_effect_30")
		cannyPixCount = scores_cache.get("canny_pix_count")
		shanonEntropy = scores_cache.get("shanon_entropy")
		laplacianVar = scores_cache.get("laplacian_var")
		ridgeMeijering = scores_cache.get("ridge_meijering")
		ridgeSato = scores_cache.get("ridge_sato")
		# Reconstruct only if needed
		if (score3 is None or score11 is None or score30 is None or cannyPixCount is None or shanonEntropy is None or laplacianVar is None or ridgeMeijering is None or ridgeSato is None):
			x = dfi_reconstruction(f, cor, angles=angles, apply_log=apply_log)
			h, w = x.shape
			sizefraction = 0.2  # keep 20% of width/height
			crop_h = int(h * sizefraction)
			crop_w = int(w * sizefraction)
			x0 = (w - crop_w) // 2
			y0 = (h - crop_h) // 2
			x = x[y0:y0 + crop_h, x0:x0 + crop_w]
			q01, q99 = np.quantile(x, [0.01, 0.99])
			im = (x - q01) / (q99 - q01 + 1e-8)
			im = np.clip(im, 0, 1)
			im = (im * 255).astype(np.uint8)
			if score3 is None:
				score3 = measure.blur_effect(
					im,
					h_size=3,
				)
				scores_cache["blur_effect_3"] = score3
			if score11 is None:
				score11 = measure.blur_effect(
					im,
					h_size=11,
				)
				scores_cache["blur_effect_11"] = score11
			if score30 is None:
				score30 = measure.blur_effect(
					im,
					h_size=30,
				)
				scores_cache["blur_effect_30"] = score30
			if cannyPixCount is None:
				canny = feature.canny(im)
				#plt.imshow(canny)
				#plt.title(f"COR={cor:.3f} canny edges")
				#plt.show()
				cannyPixCount = np.sum(canny)
				scores_cache["canny_pix_count"] = "%d"%cannyPixCount
			if shanonEntropy is None:
				shanonEntropy = measure.shannon_entropy(im)
				scores_cache["shanon_entropy"] = shanonEntropy
			if laplacianVar is None:
				laplacian = filters.laplace(im)
				laplacianVar = np.sqrt(np.sum(laplacian**2))
				scores_cache["laplacian_var"] = laplacianVar
			if ridgeMeijering is None:
				ridgeMeijering = meijering(im, black_ridges=False, sigmas=(1, 2, 3))
				#compute l2 norm
				ridgeMeijering = np.sqrt(np.sum(ridgeMeijering**2))
				scores_cache["ridge_meijering"] = ridgeMeijering
			if ridgeSato is None:
				ridgeSato = sato(im, black_ridges=False, sigmas=(1, 2, 3))
				ridgeSato = np.sqrt(np.sum(ridgeSato**2))
				scores_cache["ridge_sato"] = ridgeSato
		score_entry = {
			"blur_effect_3": score3,
			"blur_effect_11": score11,
			"blur_effect_30": score30,
			"canny_pix_count": cannyPixCount,
			"shanon_entropy": shanonEntropy,
			"laplacian_var": laplacianVar,
			"ridge_meijering": ridgeMeijering,
			"ridge_sato": ridgeSato,
		}
		results.append(score_entry)
		if verbose:
			print( f"COR={cor:10.3f} offset{cor-center:10.3f} b3={score3:.6f} b11={score11:.6f} b30={score30:.6f} canny_pix_count={cannyPixCount} shanon_entropy={shanonEntropy:.6f} laplacian_var={laplacianVar:.6f} ridge_meijering={ridgeMeijering:.6f} ridge_sato={ridgeSato:.6f}" )
	return results


def refine_cor(f, angles, refinement_steps=3, pointCount=5, apply_log=False):
	cache={}
	offsets=np.linspace(0,f.shape[1]-1,pointCount)
	for step in range(refinement_steps):
		scores=compute_scores(f,angles,offsets,cache=cache,apply_log=apply_log)
		idx=np.argsort(scores)[:3]
		best_cor=offsets[idx[0]]
		print(f"step {step}: best_cor={best_cor:.4f} score={scores[idx[0]]:.6f}")
		left=np.min(offsets[idx])
		right=np.max(offsets[idx])
		eps=(offsets[1]-offsets[0])/5
		best_idx=idx[0]
		sorted_idx=idx[np.argsort(offsets[idx])]
		left=np.min(offsets[sorted_idx])
		right=np.max(offsets[sorted_idx])
		center_idx=sorted_idx[1]
		is_center=(best_idx==center_idx)
		eps=(offsets[1]-offsets[0])/5
		if not is_center:
			left-=eps
			right+=eps
		offsets=np.linspace(left,right,pointCount)
	return best_cor


def refine_cor(f, angles, refinement_steps=3, pointCount=5, apply_log=False):
	cache={}
	offsets=np.linspace(0,f.shape[1]-1,pointCount)
	for step in range(refinement_steps):
		scores=compute_scores(f,angles,offsets,cache=cache,apply_log=apply_log)
		idx=np.argsort(scores)[:3]
		sorted_idx=idx[np.argsort(offsets[idx])]
		best_idx=idx[0]
		best_cor=offsets[best_idx]
		print(f"step {step}: best_cor={best_cor:.4f} score={scores[best_idx]:.6f}")
		left=offsets[sorted_idx[0]]
		right=offsets[sorted_idx[-1]]
		center_idx=sorted_idx[1]
		is_center=(best_idx==center_idx)
		eps=(offsets[1]-offsets[0])/5
		if not is_center:
			left-=eps
			right+=eps
		offsets=np.linspace(left,right,pointCount)
	return best_cor

def refine_cor(f, angles, refinement_steps=3, pointCount=5, apply_log=False, window=None):
	global cache
	if window is None:
		window = (0, f.shape[1] - 1)
	offsets = np.linspace(window[0], window[1], pointCount)
	for step in range(refinement_steps):
		eps = (offsets[1] - offsets[0]) / 10
		entries = compute_scores(f, angles, offsets, cache=cache, apply_log=apply_log)
		#scores = np.array([e["blur_effect_11"] for e in entries])
		#scores = np.array([-e["shanon_entropy"] for e in entries])
		scores = np.array([e["ridge_sato"] for e in entries])
		idx = np.argsort(scores)
		best_idx = idx[0]
		best_cor = offsets[best_idx]
		#Indices to the left and right of the best index
		left_idx = idx[idx < best_idx]
		right_idx = idx[idx > best_idx]
		print(f"step {step}: best_idx={best_idx} left_idx={left_idx} right_idx={right_idx}")
		rightOffset = None
		leftOffset = None
		if len(right_idx) > 0:
			right_idx = right_idx[np.argsort(scores[right_idx])]
			right = right_idx[0]
		else:
			right = best_idx
		if right != best_idx + 1:
			rightOffset = offsets[right] + eps
			print(f"step {step}: adjusting right offset to {rightOffset:.4f} because right idx is {right} and best idx is {best_idx}")
		else:
			rightOffset = offsets[right]
		if len(left_idx) > 0:
			left_idx = left_idx[np.argsort(scores[left_idx])]
			left = left_idx[0]
			leftOffset = offsets[left]
		else:
			left = best_idx
		if left != best_idx - 1:
			leftOffset = offsets[left] - eps
			print(f"step {step}: adjusting left offset to {leftOffset:.4f} because left idx is {left} and best idx is {best_idx}")
		else:
			leftOffset = offsets[left]
		print(f"step {step}: best_cor={best_cor:.4f} score={scores[best_idx]:.6f}")
		leftOffset = max(leftOffset, window[0])
		rightOffset = min(rightOffset, window[1])
		offsets = np.linspace(leftOffset, rightOffset, pointCount)
		cache.save()
	return best_cor

window = None
if ARG.pixel_window is not None:
	center = sinogram.shape[1] / 2 - 0.5
	window = (center - ARG.pixel_window, center + ARG.pixel_window)
refine_cor(sinogram, angles, refinement_steps=ARG.refinement_steps, pointCount=ARG.point_count, apply_log=False, window=window)

