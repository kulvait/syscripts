#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2023
Current correction is optimal after dark field correction, thus separate script
@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
import statistics
from PIL import Image
from PIL.TiffTags import TAGS
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
from denpy import PETRA
from denpy import UTILS
import glob
import numpy as np
from termcolor import colored
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from bisect import bisect
from bisect import bisect_left
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Lock
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import time
import traceback
from skimage.restoration import denoise_tv_chambolle

parser = argparse.ArgumentParser()
parser.add_argument(
	'h5file',
	type=str,
	help=
	"H5 file to read data for exposure correction and for current correction.")
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument(
	"--type",
	default="IMG",
	const="IMG",
	nargs="?",
	choices=["IMG", "REF"],
	help=
	"Process just one category of IMG ... actual images, REF ... flat fields.")
parser.add_argument("--target-current-value",
					type=float,
					help="Current to correct to default 100mA was substituted by None, which means first measurement in the shifted sequence [default None]",
					default=None)
parser.add_argument("--saveinfo",
					type=str,
					help="Create info object from those data and save it as...",
					default=None)
parser.add_argument("--savefig",
					type=str,
					help="Create figure from those data and save it as...",
					default=None)
parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files, just move DEN.")
parser.add_argument(
	"--params-file",
	default="params",
	type=str,
	help=
	"Provide params file. By default it is params in current working directory."
)
parser.add_argument("--correct-time-offset",
					action="store_true",
					help="Use parameter time_offset_sec from params file to estimate offset computed by cross corelation.")
parser.add_argument("--correct-insertion-offset",
					action="store_true",
					help="Use parameter time_insertion_offset_sec from params file to estimate offset computed by first derivative peak distances.")
parser.add_argument("--tv-denoise", action="store_true", help="Use total variation denoising on the mean series after current correction, before noise removal. Default is False.")
parser.add_argument("--remove-short-term-noise", type=float, default=None, help="Remove short term noise. Specify sigma in degrees to perform convolution based correction, convolved mean is used for correction. Solid default is 0.1 degrees. Default is None (no correction).")
parser.add_argument("--remove-long-term-trend", type=float, default=None, help="Remove long term trend. Specify sigma in degrees to perform convolution based correction, trend is subtracted. Solid default is 10 degrees. Default is None (no correction).")
# Create a mutually exclusive group
parser.add_argument("--connect-gaps", action="store_true", help="Connect gaps in the mean corrected data. Default is False.")
detrendg = parser.add_mutually_exclusive_group()
detrendg.add_argument("--detrend-data", action="store_true", help="Remove linear trend from the mean corrected data. Default is False.")
detrendg.add_argument("--detrend-quadratic", action="store_true", help="Remove quadratic trend from the mean corrected data. Default is False.")
detrendg.add_argument("--detrend-exponential", action="store_true", help="Remove exponential trend from the mean corrected data. Default is False.")
parser.add_argument("--perform-trigonometric-fit", type=int, default=None, help="Fit K components from trigonometric series to the mean corrected data. Specify number of components K, default is None (no fit). When the value is 0, it will use K//2, where K is the number of insertion events as specified by insertion_event_count in params file.")
parser.add_argument("--detrend-trigonometric-fit", action="store_true", help="Detrend the data before fitting trigonometric series. Default is False.")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("-j", type=int, default=-1, help="Number of threads, 0 for no threading -1 for automatic estimate.")

#ARG = parser.parse_args([])
ARG = parser.parse_args()

def correctFrame(k, ARG, frame_mean, frame_corrected_mean, lck=None):
	try:
		if lck is not None:
			lck.acquire()
		f = DEN.getFrame(ARG.inputDen, k)
		if lck is not None:
			lck.release()
		correct_factor = frame_corrected_mean / frame_mean
		f = f * correct_factor
		if lck is not None:
			lck.acquire()
		DEN.writeFrame(ARG.outputDen, k, f, force=True)
		if ARG.verbose:
			print("Frame %d: %d pixels corrected, fraction: %.2f%%"%(k, corrected_pixels, corrected_fraction*100))
		if lck is not None:
			lck.release()
		return correct_factor
	except Exception as e:
		print("Error correcting %d th frame in %s: %s" % (k, ARG.inputDen, e))
		traceback.print_exc()
		raise e
	finally:
		if lck is not None:
			lck.release()

# Means were precomputed so use info object for correction
# info[0,:] time in the format usual in synchrotron description in ms
# info[1,:] angle in degrees
# info[2,:] current in mA
# info[3,:] mean of frame
# info[4,:] median of frame
# info[5,:] mean of frame after current correction
# info[6,:] median of frame after current correction
# info[7,:] mean of ARG.lower_quantile of the info indexed frame
# info[8,:] median of ARG.lower_quantile of the frame
# info[9,:] ARG.lower_quantile
# info[10,:] mean of the frame after correction
def meanCorrect(ARG, info):
	if not os.path.exists(ARG.inputDen):
		raise IOError("File %s does not exist." % (ARG.inputDen))
	inf = DEN.readHeader(ARG.inputDen)
	if len(inf["dimspec"]) != 3:
		raise ValueError("File %s has %d dimensions, shall be 3" %
						 (ARG.inputDen, len(inf["dimspec"])))
	imageCount = inf["dimspec"][2]
	dimx = inf["dimspec"][0]
	dimy = inf["dimspec"][1]
	if info.shape[0] != 11:
		raise ValueError("info object has %d rows, shall be 11" %
						 (info.shape[0]))
	if info.shape[1] != imageCount:
		raise ValueError("info object has %d columns, shall be %d" %
						 (info.shape[1], imageCount))
	dimspec = (dimx, dimy, imageCount)
	DEN.writeEmptyDEN(ARG.outputDen, dimspec, force=True)
	if True: #ARG.j == 0:
		for i in range(imageCount):
			correctFrame(i, ARG, info[3, i], info[10, i])
	else:
		if ARG.j < 0:
			ARG.j = multiprocessing.cpu_count()
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()" %
				  (ARG.j))
		else:
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d" %
				  (ARG.j, multiprocessing.cpu_count()))
		lock = Lock()
		tp = Pool(processes=ARG.j)
		tp_start = time.time()
		results = []
		for i in range(imageCount):
			res = tp.apply_async(correctFrame,
						   args=(i, ARG, info[3, i], info[10, i], lock))
			results.append(res)
		while True:
			time.sleep(5)
			try:
				readyCount = sum([result.ready() for result in results])
				elapsed_seconds = int(time.time() - tp_start)
				elapsed_str = ""
				#Time in hours, minutes, seconds
				hours = elapsed_seconds // 3600
				minutes = (elapsed_seconds % 3600) // 60
				seconds = elapsed_seconds % 60
				#Format time
				if hours > 0:
					elapsed_str = "%d:%02d:%02ds" % (hours, minutes, seconds)
				elif minutes > 0:
					elapsed_str = "%02d:%02ds" % (minutes, seconds)
				else:
					elapsed_str = "%02ds" % (seconds)
				# Print the adjusted message
				print(f"After {elapsed_str}, {readyCount}/{imageCount} projections are corrected, {100 * readyCount / imageCount:0.2f}% of total frames.")
				if readyCount == imageCount:
					break
			except Exception as e:
				print(f"Error processing: {e}")
				continue
		# exit loop if all tasks returned success
		tp.close()
		tp.join()
		successful = [result.successful() for result in results]
		if all(successful):
			print("All tasks in current correction finished successfully.")
		else:
			failed_count = len(successful) - sum(successful)
			print("From %d tasks, %d failed." % (len(results), failed_count))
			raise Exception(f"{failed_count}/{len(results)} tasks failed.")
	return info

#Multiprocessing function to process each frame in createInfoObject
def processFrame(i, ARG, zerotime, df, dimx, dimy, lower_quantile=0.9):
	try:
		img = DEN.getFrame(ARG.inputDen, i)
		if img.shape[0] != dimy or img.shape[1] != dimx:
			raise IOError(
				"Frame %d shape (%d, %d) does not agree with expected (%d, %d)"
				% (i, img.shape[0], img.shape[0], dimx, dimy))
		#Time, angle, beam current, raw mean, raw median, corrected mean, corrected median
		img = img.flatten()
		frameCurrent = np.float32(df["current"].iloc[i])
		mean = np.mean(img)
		median = np.median(img)
		inf = np.zeros([10], dtype=np.float64)
		if ARG.target_current_value is None:
			target_current_value = np.float32(df["current"].iloc[0])
		else:
			target_current_value = ARG.target_current_value
		inf[0] = np.float64(
			(df["time"].iloc[i] - zerotime).to_pytimedelta().total_seconds())
		inf[1] = np.float64(df["s_rot"].iloc[i])
		inf[2] = frameCurrent
		inf[3] = mean
		inf[4] = median
		factor = target_current_value / frameCurrent
		inf[5] = mean * factor
		inf[6] = median * factor
		k = int(lower_quantile * len(img))
		indices = np.argpartition(img, k)
		flat = img[indices[:k]]
		inf[7] = np.mean(flat)
		inf[8] = np.median(flat)
		inf[9] = lower_quantile
		return inf
	except Exception as e:
		print("Error processing %d th frame in %s: %s" % (i, ARG.inputDen, e))
		raise e

#To write dataframe to den
def createInfoObject(ARG, timeOffsetSec=None):
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True, timeOffsetSec=timeOffsetSec)
	if ARG.type == "REF":
		image_key = 1
	else:  # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	imageCount = len(df)
	dimx = 0
	dimy = 0
	if not os.path.exists(ARG.inputDen):
		raise IOError("File %s does not exist." % (ARG.inputDen))
	inf = DEN.readHeader(ARG.inputDen)
	if len(inf["dimspec"]) != 3:
		raise ValueError("File %s has %d dimensions, shall be 3" %
						 (ARG.inputDen, len(inf["dimspec"])))
	if inf["dimspec"][2] != imageCount:
		raise ValueError(
			"File %s contains %d images but the df contains %d descriptions"
			% (ARG.inputDen, inf["dimspec"][2], imageCount))
	dimx = inf["dimspec"][0]
	dimy = inf["dimspec"][1]
	zerotime = df["time"].iloc[0]
	info = np.zeros([10, imageCount], dtype=np.float64)
	# info[0,:] time in s from the first frame
	# info[1,:] angle in degrees
	# info[2,:] current in mA
	# info[3,:] mean of frame
	# info[4,:] median of frame
	# info[5,:] mean of frame after current correction
	# info[6,:] median of frame after current correction
	# info[7,:] mean of ARG.lower_quantile of the info indexed frame
	# info[8,:] median of ARG.lower_quantile of the frame
	# info[9,:] ARG.lower_quantile
	if ARG.j == 0:
		for i in range(imageCount):
			info[:, i] = processFrame(i, ARG, zerotime, df, dimx, dimy)
	else:
		if ARG.j < 0:
			ARG.j = multiprocessing.cpu_count()
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()" %
				  (ARG.j))
		else:
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d" %
				  (ARG.j, multiprocessing.cpu_count()))
		tp = Pool(processes=ARG.j)
		tp_start = time.time()
		results = []
		for i in range(imageCount):
			res = tp.apply_async(processFrame,
						   args=(i, ARG, zerotime, df, dimx, dimy))
			results.append(res)
		while True:
			time.sleep(5)
			try:
				readyCount = sum([result.ready() for result in results])
				elapsed_seconds = int(time.time() - tp_start)
				elapsed_str = ""
				#Time in hours, minutes, seconds
				hours = elapsed_seconds // 3600
				minutes = (elapsed_seconds % 3600) // 60
				seconds = elapsed_seconds % 60
				#Format time
				if hours > 0:
					elapsed_str = "%d:%02d:%02ds" % (hours, minutes, seconds)
				elif minutes > 0:
					elapsed_str = "%02d:%02ds" % (minutes, seconds)
				else:
					elapsed_str = "%02ds" % (seconds)
				# Print the adjusted message
				print(f"After {elapsed_str}, {readyCount}/{imageCount} frames are processed, {100 * readyCount / imageCount:0.2f}% of all frames.")
				if readyCount == imageCount:
					break
			except Exception as e:
				print(f"Error processing: {e}")
				continue
		# exit loop if all tasks returned success
		tp.close()
		tp.join()
		successful = [result.successful() for result in results]
		if all(successful):
			print("All tasks finished successfully.")
		else:
			failed_count = len(successful) - sum(successful)
			print("From %d tasks, %d failed." % (len(results), failed_count))
			raise Exception(f"{failed_count}/{len(results)} tasks failed.")
		for i in range(imageCount):
			info[:, i] = results[i].get()
	return info

def quadratic_detrend_constrained(signal, sampling_axis):
	"""
	Fits a quadratic trend constrained to match signal[0] and signal[-1],
	and returns the detrended signal with the same mean as the original.

	Parameters:
		signal (np.ndarray): 1D array of the signal to detrend.
		sampling_axis (np.ndarray): 1D array (e.g. time) with same shape as signal.

	Returns:
		detrended (np.ndarray): Signal with quadratic trend removed.
		trend (np.ndarray): The fitted quadratic trend.
	"""
	signal = np.asarray(signal)
	sampling_axis = np.asarray(sampling_axis)

	if signal.shape != sampling_axis.shape:
		raise ValueError("signal and sampling_axis must have the same shape")

	t = sampling_axis
	s = signal

	# Fix endpoints of the quadratic: trend(t[0]) = s[0], trend(t[-1]) = s[-1]
	# So: trend(t) = a*t^2 + b*t + c, with c and a dependent on b
	t0, t1 = t[0], t[-1]
	s0, s1 = s[0], s[-1]

	# Define constrained quadratic where a and c are expressed in terms of b
	def constrained_quad(t, b):
		# Solve for a and c so that trend(t0) = s0, trend(t1) = s1
		A = np.array([
			[t0**2, t0, 1],
			[t1**2, t1, 1]
		])
		y = np.array([s0, s1])

		# Solve for a and c given b
		def solve_ac(b_val):
			# System: A @ [a, b, c] = y --> solve for a, c
			rhs = y - b_val * np.array([t0, t1])
			coeffs = np.linalg.lstsq(A[:, [0, 2]], rhs, rcond=None)[0]
			return coeffs  # a, c

		a, c = solve_ac(b)
		return a * t**2 + b * t + c

	# Fit b only (a and c are derived inside)
	def fit_func(t, b):
		return constrained_quad(t, b)

	popt, _ = curve_fit(fit_func, t, s, p0=[0.0])
	b_opt = popt[0]
	trend = constrained_quad(t, b_opt)

	# Subtract the trend
	detrended = s - trend

	# Adjust mean to match original signal
	detrended += np.mean(s) - np.mean(detrended)

	return detrended, trend

def exponential_detrend_constrained(signal, sampling_axis):
	"""
	Fits a monotonic exponential trend constrained to match signal[0] and signal[-1],
	and returns the detrended signal with the same mean as the original.
	
	Assumes a trend of the form: A * exp(-k * t) + C

	Parameters:
		signal (np.ndarray): 1D array of the signal to detrend.
		sampling_axis (np.ndarray): 1D array (e.g. time) with same shape as signal.

	Returns:
		detrended (np.ndarray): Signal with exponential trend removed.
		trend (np.ndarray): The fitted exponential trend.
	"""
	signal = np.asarray(signal)
	sampling_axis = np.asarray(sampling_axis)
	if signal.shape != sampling_axis.shape:
		raise ValueError("signal and sampling_axis must have the same shape")
	t = sampling_axis
	s = signal
	# Normalize time axis to avoid ill-conditioning in exponential fitting
	t_norm = (t - t[0]) / (t[-1] - t[0])
	# Define exponential with constrained endpoints
	# We'll fit only 'k' (rate); A and C are derived so that exp(t=0) = s0 and exp(t=1) = s1
	s0, s1 = s[0], s[-1]
	def constrained_exp(t_norm, k):
		# Avoid numerical overflow/underflow for large k
		k = np.clip(k, -100, 100)
		exp0 = 1.0
		exp1 = np.exp(-k)
		if np.isclose(exp0, exp1):
			A = 0.0
			C = s0	# flat line
		else:
			A = (s0 - s1) / (exp0 - exp1)
			C = s0 - A * exp0
		return A * np.exp(-k * t_norm) + C
	def fit_func(t_norm, k):
		return constrained_exp(t_norm, k)
	# Initial guess for decay rate
	popt, _ = curve_fit(fit_func, t_norm, s, p0=[1.0])
	k_opt = popt[0]
	# Compute trend and detrended signal
	trend = constrained_exp(t_norm, k_opt)
	detrended = s - trend
	# Adjust mean to match original signal
	detrended += np.mean(s) - np.mean(detrended)
	return detrended, trend

def getCurrentData(ARG, zerotime, endtime, timeOffsetSec):
	# Apply negative time offset to current to leave frame timestamps unchanged
	beamCurrentData = PETRA.beamCurrentDataset(ARG.h5file, timeOffsetSec)
	startind = bisect(list(beamCurrentData["time"]), zerotime)
	endind = bisect_left(list(beamCurrentData["time"]), endtime)
	startind = max(0, startind - 1)
	endind = min(endind + 1, len(beamCurrentData))
	beamCurrentData = beamCurrentData.iloc[startind:endind]
	return beamCurrentData, startind, endind

def adjust_insertion_events_to_slice(insertion_events, startind, endind):
	"""
	Filters and adjusts insertion events to match a restricted beamCurrentData slice.

	Parameters:
		insertion_events (list): Original events from full data.
		startind (int): Start index of the restricted slice.
		endind (int): End index (exclusive) of the restricted slice.

	Returns:
		list: Filtered and adjusted insertion events.
	"""
	adjusted_events = []
	for evt in insertion_events:
		if startind <= evt["start_index"] < endind and startind <= evt["end_index"] < endind:
			new_evt = evt.copy()
			new_evt["start_index"] -= startind
			new_evt["end_index"] -= startind
			adjusted_events.append(new_evt)
	return adjusted_events

def plotInfoOverview(info, beamCurrentData, zerotime, insertion_events=None, mainLabel=None, pdf=None):
	"""
	Plot overview of the scan information including beam current, angles, means, medians, and corrected values.
	Parameters:
		info (np.ndarray): Info object containing scan statistics.
		beamCurrentData (pd.DataFrame): DataFrame with beam current measurements.
		zerotime (pd.Timestamp): Timestamp of the first frame.
		insertion_events (list): List of insertion events with start and end indices.
		mainLabel (str): Main title for the plot.
		pdf (PdfPages): Optional PdfPages object to save the figure.
	"""
	figure, axis = plt.subplots(3, 4)
	if mainLabel is not None:
		figure.suptitle(mainLabel, fontsize=16)
	beamCurrentTimes = [(x - zerotime).to_pytimedelta().total_seconds()
						for x in beamCurrentData["time"]]
	beamCurrentValues = beamCurrentData["current"].values
	#Plot current vs. time, mean intensity vs. time
	# Scatter beam current measurements, smaller dots
	axis[0, 0].plot(beamCurrentTimes, beamCurrentValues, color="#DDCC77", linewidth=10, label="Measured Beam Current", zorder=1)
	axis[0, 0].scatter(info[0, :], info[2, :], color="#332288", s=5, label="Current Acquisition Events", zorder=2, alpha=0.7)
	# Add insertion events with distinct markers and colors
	if insertion_events is not None:
		for ev in insertion_events:
			axis[0, 0].scatter(beamCurrentTimes[ev["start_index"]], beamCurrentValues[ev["start_index"]],
				color='green', s=100, marker='D', label='Insertion Start (Min)', edgecolor='black', zorder=3)
			axis[0, 0].scatter(beamCurrentTimes[ev["end_index"]], beamCurrentValues[ev["end_index"]],
				color='red', s=100, marker='D', label='Insertion End (Max)', edgecolor='black', zorder=3)
	# Set labels and title
	axis[0, 0].set_title("Current vs time")
	axis[0, 0].set_xlabel("Time [s]")
	axis[0, 0].set_ylabel("Beam current [mA]")
	handles, labels = axis[0, 0].get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	axis[0, 0].legend(by_label.values(), by_label.keys(), loc="center right")
	
	# --- Beam Current vs Time with Mean Intensity ---
	axis[0, 1].scatter(beamCurrentTimes, beamCurrentData["current"], color="red", s=2, zorder=2)
	axis[0, 1].scatter(info[0, :], info[2, :], color="green", s=2, zorder=3)
	axis[0, 1].set_xlim([0.0, np.max(beamCurrentTimes)])  # Ensure time axis covers full range
	axis[0, 1].set_title("Beam Current Measurements vs Time")
	axis[0, 1].set_xlabel("Time [s]")
	axis[0, 1].set_ylabel("Beam Current [mA]")
	# Adding intensity on secondary y-axis
	ax2 = axis[0, 1].twinx()
	ax2.set_ylabel("Mean Intensity")
	ax2.plot(info[0, :], info[3, :], color="blue", zorder=1)
	
	axis[0, 2].plot(info[0, :], info[1, :])
	axis[0, 2].set_title("Angle vs time")
	axis[0, 3].scatter(info[6, :], info[5, :])
	axis[0, 3].set_title("Corrected mean vs corrected median")
	
	# --- Uncorrected Mean vs Time ---
	axis[1, 0].scatter(info[0, :], info[3, :], color="blue", s=2) 
	axis[1, 0].set_title("Uncorrected Mean vs Time")
	axis[1, 0].set_xlabel("Time [s]")
	axis[1, 0].set_ylabel("Uncorrected Mean")
	# --- Uncorrected Median vs Time ---
	axis[1, 1].scatter(info[0, :], info[4, :], color="green", s=1) 
	axis[1, 1].set_title("Uncorrected Median vs Time")
	axis[1, 1].set_xlabel("Time [s]")
	axis[1, 1].set_ylabel("Uncorrected Median")
	# --- Corrected Mean vs Time ---
	axis[1, 2].scatter(info[0, :], info[5, :], color="red", s=1) 
	axis[1, 2].set_title("Corrected Mean vs Time")
	axis[1, 2].set_xlabel("Time [s]")
	axis[1, 2].set_ylabel("Corrected Mean")
	axis[1, 3].plot(info[0, :], info[6, :])
	axis[1, 3].set_title("Corrected median vs time")
	
	axis[2, 0].scatter(info[0, :], info[7, :])
	axis[2, 0].set_title("Uncorrected mean of bottom 90 % vs time")
	axis[2, 1].scatter(info[0, :], info[8, :])
	axis[2, 1].set_title("Uncorrected median of bottom 90% vs time")
	axis[2, 2].scatter(info[2, :], info[5, :])
	axis[2, 2].set_title("Corrected mean vs current")
	axis[2, 3].scatter(info[2, :], info[6, :])
	axis[2, 3].set_title("Corrected median vs current")
	if pdf is not None:
		fig = plt.gcf()
		fig.set_size_inches(3 * 11.75, 3 * 8.25)
		pdf.savefig(fig, bbox_inches='tight')
		plt.close(fig)
	else:
		plt.show()

def align_segments(y_signal, time_array, gap=1.0, verbose=False):
	"""
	Applies continuity-preserving offsets to segments of a signal, based on time gaps.
	
	Each segment following a gap is offset so its first value matches the last value
	of the preceding segment.

	Parameters:
		y_signal (np.ndarray): The 1D signal to align.
		time_array (np.ndarray): Time array of same length as y_signal.
		gap (float): Threshold for detecting segment breaks.
		verbose (bool): Print debug info.

	Returns:
		np.ndarray: Offset-aligned signal.
	"""
	y_signal = np.asarray(y_signal)
	time_array = np.asarray(time_array)
	if y_signal.shape != time_array.shape:
		raise ValueError("y_signal and time_array must have the same shape.")
	aligned_signal = np.copy(y_signal)
	time_diffs = np.diff(time_array)
	gap_indices = np.where(time_diffs > gap)[0]
	segment_starts = np.insert(gap_indices + 1, 0, 0)
	segment_ends = np.append(gap_indices, len(y_signal) - 1)
	prev_end_value = None
	for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
		segment = aligned_signal[start:end + 1]
		if verbose:
			print(f"Segment {i+1}: {start} to {end}, len={len(segment)}")
		if prev_end_value is not None:
			offset = prev_end_value - segment[0]
			aligned_signal[start:end + 1] += offset
			if verbose:
				print(f"  Offset applied: {offset:.4f} to segment {i+1}")
		prev_end_value = aligned_signal[end]
	return aligned_signal

def smooth_intensity(x_angle, y_intensity, sigma_degrees, time_array=None, time_gap=1.0, verbose=False):
	"""
	Smooths intensity using a Gaussian kernel, optionally applied to segments
	separated by large time gaps.

	Parameters:
		x_angle (np.ndarray): 1D array of angle values.
		y_intensity (np.ndarray): 1D array of intensity values (same shape as x_angle).
		sigma_degrees (float): Gaussian smoothing width in degrees.
		time_array (np.ndarray or None): Optional 1D time array. If provided, large time gaps will segment the data.
		time_gap (float): Time gap threshold to split segments (used only if time_array is given).
		verbose (bool): Print segment info and kernel stats.

	Returns:
		np.ndarray: Smoothed intensity signal.
	"""
	x_angle = np.asarray(x_angle)
	y_intensity = np.asarray(y_intensity)
	if x_angle.shape != y_intensity.shape:
		raise ValueError("x_angle and y_intensity must have the same shape.")
	smoothed_intensity = np.copy(y_intensity)
	# Determine segments
	if time_array is not None:
		time_array = np.asarray(time_array)
		if time_array.shape != x_angle.shape:
			raise ValueError("time_array must have the same shape as x_angle.")
		time_diffs = np.diff(time_array)
		gap_indices = np.where(time_diffs > time_gap)[0]
		segment_starts = np.insert(gap_indices + 1, 0, 0)
		segment_ends = np.append(gap_indices, len(time_array) - 1)
	else:
		segment_starts = [0]
		segment_ends = [len(x_angle) - 1]
	# Smooth each segment individually
	for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
		seg_x = x_angle[start:end + 1]
		seg_y = y_intensity[start:end + 1]
		if verbose:
			print(f"Segment {i+1}: indices {start} to {end}, length {end - start + 1}")
		if len(seg_y) > 1:
			pixel_size = np.mean(np.diff(seg_x))
			sigma = sigma_degrees / pixel_size
			kernel_size = int(6 * sigma) + 1
			if kernel_size % 2 == 0:
				kernel_size += 1
			if verbose:
				print(f"  pixel_size = {pixel_size:.4f}, sigma = {sigma:.2f}, kernel_size = {kernel_size}")
			x_kernel = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
			gaussian_kernel = np.exp(-0.5 * (x_kernel / sigma) ** 2)
			gaussian_kernel /= np.sum(gaussian_kernel)
			smoothed_segment = convolve1d(seg_y, gaussian_kernel, mode='reflect')
			smoothed_intensity[start:end + 1] = smoothed_segment
		else:
			if verbose:
				print(f"  Skipping segment {i+1} (too short for smoothing)")
	return smoothed_intensity

def denoise_chambolle_segment(y_signal, time_array, weight=100, gap=1.0, verbose=False):
	"""
	Denoises a 1D signal using total variation denoising (TV Chambolle),
	applied to segments separated by time gaps larger than a specified threshold.

	Parameters:
		y_signal (np.ndarray): The noisy signal to denoise.
		time_array (np.ndarray): Corresponding time values (must be same length as y_signal).
		weight (float): Weight for TV denoising. Higher = smoother.
		gap (float): Time gap threshold (in same units as time_array) to separate segments.
		verbose (bool): Whether to print debug output.

	Returns:
		np.ndarray: The denoised signal.
	"""
	y_signal = np.asarray(y_signal)
	time_array = np.asarray(time_array)
	if len(y_signal) != len(time_array):
		raise ValueError("y_signal and time_array must have the same length.")
	time_diffs = np.diff(time_array)
	gap_indices = np.where(time_diffs > gap)[0]
	segment_starts = np.insert(gap_indices + 1, 0, 0)
	segment_ends = np.append(gap_indices, len(time_array) - 1)
	denoised_signal = np.copy(y_signal)
	for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
		segment = y_signal[start:end + 1]
		if verbose:
			print(f"Segment {i+1}: index {start} to {end}, length {end - start + 1}")
		if len(segment) > 1:
			denoised_segment = denoise_tv_chambolle(segment, weight=weight)
			denoised_signal[start:end + 1] = denoised_segment
		else:
			if verbose:
				print(f"Skipping segment {i+1} (too short for denoising)")
	return denoised_signal

def fit_trig_basis(time_axis, signal, K, detrend=False, enforce_even_K=True):
	"""
	Fit signal with first K terms of trigonometric (Fourier) basis.
	
	Parameters:
	- time_axis: 1D array of time points
	- signal: 1D array of signal values to fit
	- K: Number of trigonometric basis functions
	- detrend: bool, if True, remove linear trend before fitting
	- enforce_even_K: bool, if True, enforce K >= 2 and even; else keep K as is
	
	Returns:
	- fitted_signal: Reconstruction using K basis terms
	- coeffs: Tuple of (a0, a_k array, b_k array)
	"""
	original_K = K
	if enforce_even_K:
		if K < 2:
			K = 2
		if K % 2 != 0:
			K += 1
		if K != original_K:
			print(f"Modified K from {original_K} to {K} for balance of trigonometric functions.")
	# Detrend if requested
	if detrend:
		start_val = signal[0]
		end_val = signal[-1]
		duration = time_axis[-1] - time_axis[0]
		slope = (end_val - start_val) / duration
		trend_line = start_val + slope * (time_axis - time_axis[0])
		
		signal_detrended = signal - trend_line
		mean_signal = np.mean(signal)
		signal = signal_detrended + mean_signal
	T = time_axis[-1] - time_axis[0]  # Total duration
	t = (time_axis - time_axis[0]) / T# Normalize to [0,1]

	N = len(t)
	A = np.ones((N, 2*K + 1))
	for k in range(1, K + 1):
		A[:, 2*k - 1] = np.cos(2 * np.pi * k * t)
		A[:, 2*k]	  = np.sin(2 * np.pi * k * t)

	# Solve least squares: Ax ≈ y
	coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
	fitted_signal = A @ coeffs

	# Split coefficients for optional inspection
	a0 = coeffs[0]
	a_k = coeffs[1::2]
	b_k = coeffs[2::2]

	return fitted_signal, (a0, a_k, b_k)

def unwrap_make_monotone(info_angle, wrap_increase=360, threshold=None):
	info_angle = np.asarray(info_angle, dtype=float)
	if threshold is None:
		threshold = 0.5 * wrap_increase  # default: half wrap
	corrected = [info_angle[0]]
	unwrap_factor = 0
	for i in range(1, len(info_angle)):
		prev_raw = info_angle[i - 1]
		curr_raw = info_angle[i]
		delta = curr_raw - prev_raw
		# If the jump looks like a wrap (e.g., 350 → 10), increase unwrap factor
		if delta < -threshold:
			unwrap_factor += wrap_increase
		# Unwrap the current angle
		unwrapped = curr_raw + unwrap_factor
		# If still less than the last corrected (small decrease or noise), repeat
		if unwrapped < corrected[-1]:
			unwrapped = corrected[-1]
		corrected.append(unwrapped)
	return np.array(corrected)

if ARG.verbose:
	print("Start currentCorrect.py h5=%s" % (ARG.h5file))

try:
	#Preprocessing of the input arguments including time offset
	timeOffsetSec = 0.0
	if ARG.params_file is not None:
		params = UTILS.readParamsFile(ARG.params_file)
	else:
		params = {}
	if ARG.correct_time_offset:
		if params.get("time_offset_sec") is not None:
			timeOffsetSec = float(params["time_offset_sec"])
			print("Using time offset from params file: %fs" % (timeOffsetSec))
	elif ARG.correct_insertion_offset:
		if params.get("time_insertion_offset_sec") is not None:
			timeOffsetSec = float(params["time_insertion_offset_sec"])
			print("Using time insertion offset from params file: %fs" % (timeOffsetSec))
	if ARG.type == "REF":
		image_key = 1
	else:  # IMG
		image_key = 0
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True, timeOffsetSec=timeOffsetSec)
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	zerotime = df["time"].iloc[0]
	endtime = df["time"].iloc[-1]
	totaltime = (endtime - zerotime).to_pytimedelta().total_seconds()
	beamcurrentData, startind, endind = getCurrentData(ARG, zerotime, endtime, timeOffsetSec)
	insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec, zerotime)
	insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
	#insertion_events_valid = [
	#	event for event in insertion_events
	#	if 0 <= event["mid_time"] <= totaltime	
	# Ensure mid_time is within valid range
	#]
	intertion_event_count = len(insertion_events_valid)
	imageCount = len(df)
	#First create info object with statistics of ARG.inputDen and corrected mean
	info = createInfoObject(ARG, timeOffsetSec)
	info_time = info[0, :] - info[0, 0]  # Time in seconds from the first frame
	info_angle = info[1, :].copy()  # Angle in degrees
	info_angle_unwrapped = unwrap_make_monotone(info_angle, wrap_increase=360, threshold=180)
	if ARG.type == "IMG":
		info_angle = info[1, :]
	else:
		info_angle = np.arange(info.shape[1])#Here I do not have any angular information so I use frame index
	if ARG.savefig is not None:
		matplotlib.use('Agg')
		pdf = PdfPages(ARG.savefig)
		plotInfoOverview(info, beamcurrentData, zerotime, insertion_events_valid, mainLabel="Characteristisc of the scan for timeSecOffeset=%f" %
					(timeOffsetSec), pdf=pdf)
	mean_corrected_noisy = info[5, :].copy()  # This is the mean after current correction
	mean_corrected_original = mean_corrected_noisy.copy()  # This is the original mean before any further processing
	if ARG.savefig is not None:
		figure, ax = plt.subplots(figsize=(15, 5))
		ax.set_title("Preprocessing of the current corrected mean data")
		ax.scatter(info_time, mean_corrected_noisy, label="Original mean data", s=5)
	if ARG.tv_denoise:
		mean_corrected_noisy = denoise_chambolle_segment(mean_corrected_noisy, info_time, weight=100, gap=1.0, verbose=ARG.verbose)
		if ARG.savefig is not None:
			ax.scatter(info_time, mean_corrected_noisy, label="TV denoising of mean data", s=3, alpha=0.7, color='orange')
	if ARG.connect_gaps:
		mean_corrected_noisy = align_segments(mean_corrected_noisy, info_time, gap=1.0, verbose=ARG.verbose)
		if ARG.savefig is not None:
			ax.scatter(info_time, mean_corrected_noisy, label="Aligned gaps in mean data", s=3, alpha=0.7, color='purple')
	if ARG.savefig is not None:
		ax.set_xlabel("Time [s]")
		ax.set_ylabel("Mean intensity")
		ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Horizontal lines only
		ax.legend()
		figure.tight_layout()
		if pdf is not None:
			pdf.savefig(figure, bbox_inches='tight')
			plt.close(figure)
	#mean_corrected_noisy contains the current corrected mean data, possibly TV denoised and aligned
	if ARG.remove_short_term_noise is not None or ARG.remove_long_term_trend is not None:
		if ARG.verbose:
			if ARG.remove_short_term_noise is not None and ARG.remove_long_term_trend is not None:
				print("Removing short term noise with sigma=%f and long term trend with sigma=%f" %
					  (ARG.remove_short_term_noise, ARG.remove_long_term_trend))
			elif ARG.remove_short_term_noise is not None:
				print("Removing short term noise with sigma=%f" %
					  (ARG.remove_short_term_noise))
			elif ARG.remove_long_term_trend is not None:
				print("Removing long term trend with sigma=%f" %
					  (ARG.remove_long_term_trend))
		if ARG.remove_long_term_trend is not None:
			mean_adjust = smooth_intensity(info_angle, mean_corrected_noisy, ARG.remove_long_term_trend, time_array=info_time, time_gap=1.0, verbose=ARG.verbose)
		else:
			mean_adjust = np.zeros_like(mean_corrected_noisy)
		if ARG.remove_short_term_noise is not None:
			mean_corrected = smooth_intensity(info_angle, mean_corrected_noisy, ARG.remove_short_term_noise, time_array=info_time, time_gap=1.0, verbose=ARG.verbose)
		else:
			mean_corrected = mean_corrected_noisy.copy()
		#Plot smoothed data
		if ARG.savefig is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Smoothed mean data")
			ax.scatter(info_time, mean_corrected_original, label="Adjustment mean data", s=10)
			if ARG.tv_denoise or ARG.connect_gaps:
				ax.scatter(info_time, mean_corrected_noisy, label="TV denoised and/or gap connected mean data", s=5, alpha=0.7)
			if ARG.remove_short_term_noise is not None:
				ax.scatter(info_time, mean_corrected, label="Smoothed high frequency oscilations with sigma=%0.2f" % (ARG.remove_short_term_noise), s=5, alpha=0.7, color='red')
			if ARG.remove_long_term_trend is not None:
				ax.scatter(info_time, mean_adjust, label="Smoothed low frequency oscilations with sigma=%0.2f" % (ARG.remove_long_term_trend), s=5, alpha=0.7, color='green')
			ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Horizontal lines only
			ax.legend()
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Mean intensity")
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
		#Plot adjusted data
		mean_adjust = mean_adjust - np.mean(mean_adjust)
		mean_corrected = mean_corrected - mean_adjust
		if ARG.savefig is not None and ARG.remove_long_term_trend is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Adjusted mean data")
			ax.scatter(info_time, mean_corrected_noisy - mean_adjust, label="Adjusted mean data")
			if ARG.remove_short_term_noise is not None:
				ax.scatter(info_time, mean_corrected, label="Smoothed high frequency oscilations with sigma=%0.2f adjusted to the trendline with sigma=%0.2f" % (ARG.remove_short_term_noise, ARG.remove_long_term_trend))
			ax.legend()
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
	else:
		mean_corrected = mean_corrected_noisy
	if ARG.detrend_data:
		if ARG.verbose:
			print("Detrending mean corrected data")
		#Remove linear trend
		start_val = mean_corrected[0]
		end_val = mean_corrected[-1]
		duration = info_time[-1] - info_time[0]
		slope = (end_val - start_val) / duration
		trend_line = start_val + slope * (info_time - info_time[0])
		trend_line = trend_line - np.mean(trend_line)  # Center the trend line
		#Adjust mean to the original mean
		mean_corrected_detrend = mean_corrected - trend_line
		if ARG.savefig is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Detrended mean data by linear fit")
			ax.scatter(info_time, mean_corrected, label="Mean corrected data")
			ax.scatter(info_time, mean_corrected_detrend, label="Linear detrended")
			ax.legend()
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
		#Set mean_corrected to the detrended data
		mean_corrected = mean_corrected_detrend
	elif ARG.detrend_quadratic:
		if ARG.connect_gaps:
			#Use rather unwrapped angles to avoid gaps in the time axis
			mean_corrected_detrend, trend_line = quadratic_detrend_constrained(mean_corrected, info_angle_unwrapped)
		else:
			mean_corrected_detrend, trend_line = quadratic_detrend_constrained(mean_corrected, info_time)
		if ARG.savefig is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Quadratic detrended mean data")
			ax.scatter(info_time, mean_corrected, label="Mean corrected data")
			ax.scatter(info_time, trend_line, label="Fitted quadratic trend")
			ax.scatter(info_time, mean_corrected_detrend, label="Quadratic detrended mean corrected data")
			ax.legend()
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Mean intensity")
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
		mean_corrected = mean_corrected_detrend
	elif ARG.detrend_exponential:
		if ARG.connect_gaps:
			#Use rather unwrapped angles to avoid gaps in the time axis
			mean_corrected_detrend, trend_line = exponential_detrend_constrained(mean_corrected, info_angle_unwrapped)
		else:
			mean_corrected_detrend, trend_line = exponential_detrend_constrained(mean_corrected, info_time)
		if ARG.savefig is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Exponential detrended mean data")
			ax.scatter(info_time, mean_corrected, label="Mean corrected data", s=10)
			ax.scatter(info_time, trend_line, label="Fitted exponential trend", s=5, alpha=0.7)
			ax.scatter(info_time, mean_corrected_detrend, label="Exponential detrended mean corrected data", s=5, alpha=0.5)
			ax.axhline(mean_corrected_detrend[0], color='black', linestyle='--', linewidth=0.5, alpha=0.5)
			ax.legend()
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Mean intensity")
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
		mean_corrected = mean_corrected_detrend
	if ARG.perform_trigonometric_fit is not None:
		K = ARG.perform_trigonometric_fit
		if K == 0:
			if "insertion_event_count" in params:
				K = int(params["insertion_event_count"]) // 2
			else:
				raise ValueError("Parameter insertion_event_count is not defined in params file, cannot estimate K for trigonometric fit.")
		if ARG.verbose:
			print("Performing trigonometric fit with %d components" % (K))
		fitted_signal, (a0, a_k, b_k) = fit_trig_basis(info_time, mean_corrected, K, detrend=ARG.detrend_trigonometric_fit, enforce_even_K=True)
		if ARG.savefig is not None:
			figure, ax = plt.subplots(figsize=(15, 5))
			ax.set_title("Trigonometric fit of the mean data")
			ax.scatter(info_time, mean_corrected, label="Mean corrected data")
			ax.scatter(info_time, fitted_signal, label="Fitted signal with %d components" % (K))
			ax.legend()
			ax.set_xlabel("Time [s]")
			ax.set_ylabel("Mean intensity")
			figure.tight_layout()
			if pdf is not None:
				pdf.savefig(figure, bbox_inches='tight')
				plt.close(figure)
		mean_corrected = fitted_signal
	info = np.vstack([info, mean_corrected])
	#Perform mean adjustment
	meanCorrect(ARG, info)
	#Save info
	if ARG.saveinfo is not None:
		DEN.storeNdarrayAsDEN(ARG.saveinfo, info, force=True)
	#Close the PDF file if it was opened
	if pdf is not None:
		pdf.close()
except Exception as e:
	print(colored("Error in currentCorrect.py: %s" % (e), "red"))
	traceback.print_exc()
	if ARG.fix_corrupted_h5:
		#Copy ARG.inputDen to ARG.outputDen
		print(colored("Error in currentCorrect.py fixing by movind %s to %s, do not write log images." % (ARG.inputDen, ARG.outputDen), "red"))
		os.system("mv %s %s" % (ARG.inputDen, ARG.outputDen))
		sys.exit(1)
	else:
		print(colored("Error in currentCorrect.py try running with --fix-corrupted-h5", "red"))
		sys.exit(1)

	

if ARG.verbose:
	print("End currentCorrect.py h5=%s" % (ARG.h5file))
