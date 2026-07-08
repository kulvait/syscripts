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
parser.add_argument("ref_avg", help="Reference average of the sample region, used for normalization.")
parser.add_argument("img_avg", help="Average intensity of the sample region in the current reconstruction, used for normalization.")
parser.add_argument("img_sd", help="Standard deviation of the sample region in the current reconstruction, used for normalization.")
parser.add_argument("--illum-factor", type=float, default=0.5, help="Threshold factor for illumination (ref_profile > factor * mean), the higher the factor, more conservative the estimation of the illuminated region.")
parser.add_argument("--buckets", type=int, default=5, help="Number of vertical buckets inside illuminated region")
parser.add_argument("--out-pdf", default=None, help="Path to save plot as PDF")
parser.add_argument("--out-json", default=None, help="Path to save COR results as JSON")
ARG = parser.parse_args()

ref_avg = DEN.getFrame(ARG.ref_avg, 0)
img_avg = DEN.getFrame(ARG.img_avg, 0)
img_sd = DEN.getFrame(ARG.img_sd, 0)

# Find illuminated region in the reference average profile ... vertically (y-axis)
ref_profile = ref_avg.mean(axis=1)
ref_mean = np.mean(ref_profile)
ref_rel = ref_profile / ref_mean
mask_ref = ref_profile > ref_mean*ARG.illum_factor
idx = np.where(mask_ref)[0]
if len(idx) < 2:
	raise RuntimeError("No illuminated region found")
illuminated_top = idx[0]
illuminated_bottom = idx[-1]
#Do the same in horizontal direction (x-axis) if needed, but for now we focus on vertical region
ref_profile_h = ref_avg.mean(axis=0)
ref_mean_h = np.mean(ref_profile_h)
mask_ref_h = ref_profile_h > ref_mean_h*ARG.illum_factor
idx_h = np.where(mask_ref_h)[0]
if len(idx_h) < 2:
	raise RuntimeError("No illuminated region found in horizontal direction")
illuminated_left = idx_h[0]
illuminated_right = idx_h[-1]

illuminated_region = np.arange(illuminated_top, illuminated_bottom + 1)
iluminated_region_msg = f"Vertical illuminated region: [{illuminated_top}, {illuminated_bottom}], width = {illuminated_bottom - illuminated_top}, factor = {ARG.illum_factor}"
print(iluminated_region_msg)
iluminated_region_h_msg = f"Horizontal illuminated region: [{illuminated_left}, {illuminated_right}], width = {illuminated_right - illuminated_left}, factor = {ARG.illum_factor}"
print(iluminated_region_h_msg)

img_profile = img_avg.mean(axis=1)
sd_profile= img_sd.mean(axis=1)
sd_profile_median = np.median(sd_profile)
sd_profile = np.clip(sd_profile, sd_profile_median*0.5, sd_profile_median*100)  # Clip to avoid extreme values
sd_div_img = np.divide(sd_profile, np.sqrt(img_profile), out=np.zeros_like(sd_profile), where=img_profile!=0)
sd_div_img = np.divide(np.sqrt(img_profile), sd_profile, out=np.zeros_like(sd_profile), where=img_profile!=0) * ref_rel  # Normalize by reference profile

# Define vertical buckets within the illuminated region
def center_weight(n):
	# Create a weight array that peaks at the center and tapers towards the edges
	x = np.linspace(-1, 1, n)
	return 0.75 + 0.25 * np.cos(0.5*np.pi * x)  # Add a cosine component for smoother tapering

bucket_edges = np.linspace(illuminated_top, illuminated_bottom, ARG.buckets + 1, dtype=int)
bucket_maxima = []
regions = []
for i in range(ARG.buckets):
	bucket_start = bucket_edges[i]
	bucket_end = bucket_edges[i + 1]
	bucket_profile = sd_div_img[bucket_start:bucket_end].copy()  # Copy to avoid modifying the original array
	if len(bucket_profile) == 0:
		bucket_maxima.append(None)
		continue
	bucket_profile *= center_weight(len(bucket_profile))  # Apply center weighting
	max_idx = np.argmax(bucket_profile)
	global_max_idx = bucket_start + max_idx
	bucket_maxima.append(global_max_idx)
	regions.append({
		"id": i,
		"start": int(bucket_start),
		"end": int(bucket_end),
		"max_row": int(global_max_idx),
		"values": {
			"ref_avg": float(ref_profile[global_max_idx]),
			"tra_avg": float(img_profile[global_max_idx]),
			"img_sd": float(sd_profile[global_max_idx]),
			"sd_div_img": float(sd_div_img[global_max_idx]),
		},
	})
	print(f"Bucket {i+1}: [{bucket_start}, {bucket_end}), max at row {global_max_idx} with value {bucket_profile[max_idx]:.3g}")


results = {
    "illuminated_region": {
        "x_min": int(illuminated_left),
        "x_max": int(illuminated_right),
        "y_min": int(illuminated_top),
        "y_max": int(illuminated_bottom),
        "illuminated_width": int(illuminated_right - illuminated_left),
        "illuminated_height": int(illuminated_bottom - illuminated_top),
        "mean_ref_intensity": float(ref_mean),
        "threshold_factor": float(ARG.illum_factor),
        "ref_intensity_threshold": float(ref_mean * ARG.illum_factor)
    },
    "subregion_count": ARG.buckets,
    "subregions": regions
}

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

profiles = [
	("ref_avg", ref_profile),
	("tra_avg", img_profile),
#	 ("img_sd", sd_profile),
	("RQP", sd_div_img), #Reconstruction Quality Predicator
]

for i, (ax, (name, profile)) in enumerate(zip(axes, profiles)):
	mean_val = np.mean(profile)
	median_val = np.median(profile)
	ax.plot(profile, label="row profile")
	# illumination boundaries (NO legend clutter)
	ax.axvline(illuminated_top, color="blue", linewidth=2, alpha=0.8)
	ax.axvline(illuminated_bottom, color="blue", linewidth=2, alpha=0.8)
	ax.set_ylabel(name)
	ax.grid(True, alpha=0.3)
	ax.legend()
	# title ONLY on first plot
	if i == 0:
		thr_val = ref_mean * ARG.illum_factor
		ax.axhline(thr_val, color="magenta", linestyle="--", label=f"threshold = {int(thr_val)}")
		ax.axhline(mean_val, color="red", linestyle="--", label=f"mean = {int(mean_val)}")
		ax.axhline(median_val, color="green", linestyle=":", label=f"median = {int(median_val)}")
		ax.set_title(iluminated_region_msg)
	else:
		ax.axhline(mean_val, color="red", linestyle="--", label=f"mean = {mean_val:.3g}")
		ax.axhline(median_val, color="green", linestyle=":", label=f"median = {median_val:.3g}")
	ax.legend()


# -----------------------------
# LAST plot: add bucket info
# -----------------------------
last_ax = axes[-1]
last_ax.set_xlabel("Row index (y)")
# bucket boundaries
for b in bucket_edges:
	last_ax.axvline(b, color="orange", linestyle="--", alpha=0.4)
# bucket maxima
for i, max_idx in enumerate(bucket_maxima):
	if max_idx is not None:
		last_ax.plot(max_idx, sd_div_img[max_idx], 'ro', label=f'Bucket {i+1} max')

plt.tight_layout()

if ARG.out_pdf is not None:
	fig.savefig(ARG.out_pdf)
	log.info(f"Plot saved to {ARG.out_pdf}")
else:
	plt.show()

if ARG.out_json is not None:
	with open(ARG.out_json, "w") as f:
		json.dump(results, f, indent=2)
	log.info(f"Results saved to {ARG.out_json}")
