#!/usr/bin/env python
"""
Created on Wed Jul	12 2023

@author: Vojtech Kulvait

This script produces camera matrices for given dataset based on H5 file and provided dimensions of pixels

"""

from termcolor import colored
import argparse
import glob
import os
import sys
import errno
import sys
import h5py
import math
import re
from denpy import DEN
from denpy import UTILS
from denpy import PETRA
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("inputGeometryInfo", help="H5 file with dataset information, or DEN tick file with dataset information")
parser.add_argument("outputProjectionMatrices", help="DEN file to store projection matrices")

#Position of the rotation center can be specified in three mutually exclusive ways, if there is no specification, it is assumed to be 0.0
groupOffset = parser.add_mutually_exclusive_group()
#Rotation center offset convention: offset of the rotation axis image from the center of the unbinned camera frame (ROI),
#positive towards higher detector x index. The axis is imaged at binned pixel index
#(N-1)/2 + (rotation_center_offset + binning_center_adjustment + detector_center_offsetvx)/pixel_sizex,
#see binning_center_adjustment below.
groupOffset.add_argument("--rotation-center-offset", default=None, help="Offset of the rotation axis from the center of the unbinned camera frame (ROI) in physical units of the pixel size, positive towards higher detector x index, defaults to 0.0.", type=float)
groupOffset.add_argument("--rotation-center-offset-pix", default=None, help="Offset of the rotation axis from the center of the unbinned camera frame (ROI) in pixels after binning, i.e. in units of bin_x camera pixels; rotation_center_offset = value * bin_x * camera pixel size. Note that rotation_center_offset_pix* keys in --rotation-center-file are in unbinned camera pixels. Defaults to 0.0.", type=float)
groupOffset.add_argument("--rotation-center-file", default=None, help="Log file of detectRotationCenter*.py with rotation_center_offset*=value lines, values relative to the center of the unbinned camera frame (ROI), *_pix* values in unbinned camera pixels. Priority: linear fit evaluated at --rotation-center-file-fit-y, rotation_center_offset_pix_interp, rotation_center_offset_pix, rotation_center_offset.")
parser.add_argument("--rotation-center-file-fit-y", default=None, type=float, help="Binned detector row (bin_y camera rows per row, row 0 at the top of the camera) at which the linear fit rotation_center_offset_interpfit_a + rotation_center_offset_interpfit_b * y_unbinned from --rotation-center-file is evaluated. Use the binned camera row of the center of the (possibly cropped) detector. With --rotation-center-file-fitted-matrix it is the binned camera row of the center row (M-1)/2 of the detector.")
parser.add_argument("--rotation-center-file-fitted-matrix", action="store_true", help="Create skewed projection matrix based on the linear fit of the rotation center offset in --rotation-center-file. If the detector is cropped in y, provide --rotation-center-file-fit-y with the binned camera row of its center row.")


parser.add_argument("--detector-center-offsetvx", type=float, default=0., help="Additional offset in physical units, added to the rotation center offset, i.e. the rotation axis is imaged detector_center_offsetvx/pixel_sizex binned pixels further towards higher x index (opposite sign to the same option of createCameraMatricesForCircularScanTrajectoryParallelRay3D.py), defaults to 0.0.")
parser.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset in physical units, the plane z=0 is imaged detector_center_offsetvy/pixel_sizey binned rows below the center row (M-1)/2 in the material CT convention, defaults to 0.0.")
parser.add_argument("--pixel-sizex", default=None, type=float, help="Pixel size x to use for projection geometry.")
parser.add_argument("--pixel-sizey", default=None, type=float, help="Pixel size y to use for projection geometry.")
parser.add_argument("--detector-sizex", default=None, type=int, help="Detector x size of the matrices in binned pixels, takes precedence over dimx of --input-file and the h5 camera size; a smaller value than the binned camera width is a crop that shall be symmetric (or compensated by --detector-center-offsetvx).")
parser.add_argument("--detector-sizey", default=None, type=int, help="Detector y size (row count) of the matrices in binned pixels, takes precedence over dimy of --input-file and the h5 camera size, e.g. the row count of a slab. The plane z=0 is imaged at its center row (M-1)/2, combine with --rotation-center-file-fit-y set to the binned row of the slab center.")
parser.add_argument("--bin-x", type=int, default=1, help="X dimension of binning box, pixel size will be multiplied by this factor and default detector size is adjusted accordingly.")
parser.add_argument("--bin-y", type=int, default=1, help="Y dimension of binning box, pixel size will be multiplied by this factor and default detector size is adjusted accordingly.")
parser.add_argument("--input-file", default=None, help="DEN file with projections to create corresponding projection matrices, its projection count is checked and its dimx, dimy are used unless --detector-sizex, --detector-sizey are given; without it the (ROI) camera dimensions from the h5 file are used.")
parser.add_argument("--material-ct-convention", action="store_true", help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
parser.add_argument("--flat-matrices", help="Create 1D projection matrices, rotation direction follows --material-ct-convention as for 2D matrices.", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--petra-compatibility-transform", action="store_true", help="Make geometry compatible with that used by JM in his reconstruction scripts")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
parser.add_argument("--override-magnification", default=None)
parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files, provide --input-file.")
parser.add_argument("--ignore-pixel-shifts", action="store_true", help="Ignore pixel shifts as they appear in h5 file")
parser.add_argument("--ignore-exact-angles", action="store_true", help="Ignore exact angles as they appear in h5 file")
parser.add_argument("--force", action="store_true")
parser.add_argument("--write-params-file", action="store_true")
parser.add_argument('--_json-message', default="Created using KCT script createCameraMatricesForCircularScanTrajectoryParallelRay3D.py", help=argparse.SUPPRESS)
ARG = parser.parse_args()

if ARG.flat_matrices and ARG.rotation_center_file_fitted_matrix:
	parser.error("--flat-matrices can not be combined with --rotation-center-file-fitted-matrix")

zdim = None
inputIsDEN = False
h5 = None
tickArray = None
df = None

if not  os.path.isfile(ARG.inputGeometryInfo):
	print(f"Error: file not found: {ARG.inputGeometryInfo}", file=sys.stderr)
	sys.exit(-1)

try:
	DEN.readHeader(ARG.inputGeometryInfo)
	tickArray = DEN.getNumpyArray(ARG.inputGeometryInfo)
	inputIsDEN = True
	zdim = tickArray.shape[1]
except TypeError as e:
	h5 = h5py.File(ARG.inputGeometryInfo, 'r')
	inputIsDEN = False

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

if ARG.fix_corrupted_h5 or inputIsDEN:
	if ARG.input_file is not None:
		header = DEN.readHeader(ARG.input_file)
		if len(header["dimspec"]) != 3:
			raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!"%(ARG.input_file, len(header["dimspec"])))
		dimspec = header["dimspec"]
		detector_sizex = int(dimspec[0]) if ARG.detector_sizex is None else ARG.detector_sizex
		detector_sizey = int(dimspec[1]) if ARG.detector_sizey is None else ARG.detector_sizey
		if zdim is not None and zdim != dimspec[2]:
			raise TypeError("Number of projections in %s is %d but according to TICK file it is %d!"%(ARG.input_file, dimspec[2], zdim))
	else:
		raise ValueError("Input file must be specified to fix corrupted h5 file.")

try:
	if inputIsDEN:
		dat = {
			"frame_ind":np.arange(zdim, dtype=np.int64),
			"s_rot":tickArray[0],
			"pixel_shift":tickArray[1]
		}
		df = pd.DataFrame(dat)
	elif ARG.override_magnification is not None:
		df = PETRA.imageDataset(ARG.inputGeometryInfo, includeCurrent=False, includePixelShift=True, overrideMagnification=float(ARG.override_magnification))
	else:
		df = PETRA.imageDataset(ARG.inputGeometryInfo, includeCurrent=False, includePixelShift=True)
except Exception as e:
	print(colored("Error: %s"%e, "red"))
	if ARG.fix_corrupted_h5:
		print("Corrupted h5 file fix by np.linspace.", file=sys.stderr)
		data = {"frame_ind":np.arange(zdim),"s_rot":np.linspace(0, 180, num=zdim, endpoint=False),"pixel_shift":np.zeros(zdim)}
		df = pd.DataFrame(data)
	else:
		sys.exit(1)


#Solution from https://stackoverflow.com/a/55114771
def checkFileExistence(f):
	if os.path.exists(f):
		if os.path.isfile(f):
			if ARG.force:
				os.remove(f)
				return
			else:
				print("File %s exist, add --force to proceed."%f)
				sys.exit(1)
		else:
			print("Path %s exist remove to proceed.", f)
			sys.exit(1)

checkFileExistence(ARG.outputProjectionMatrices)
if ARG.write_params_file:
	paramsFile="%s.params"%(ARG.outputProjectionMatrices)
	checkFileExistence(paramsFile)

if h5 is not None:
	if "/entry/hardware/camera1" in h5: 
		cam = "camera1"
	elif "/entry/hardware/camera" in h5: 
		cam = "camera"
	else:
		raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."%h5)
	pix_sizecam = float(h5["entry/hardware/%s/pixelsize"%cam][0])
	overrideMagnification = ARG.override_magnification
	if overrideMagnification is not None:
		pix_sizemag = float(overrideMagnification)
	else:
		pix_sizemag = float(h5["entry/hardware/%s/magnification"%cam][0])
	default_pixel_size = float(pix_sizecam/pix_sizemag)
	#Size of the recorded projections: camera ROI if stored (as in denpy PETRA.getExperimentInfo), else the sensor.
	#Rotation center offsets and the rows of the rotation center fit refer to this unbinned frame.
	def cameraSize(keys):
		for k in keys:
			if "entry/hardware/%s/%s"%(cam, k) in h5:
				return int(h5["entry/hardware/%s/%s"%(cam, k)][0])
		raise ValueError("None of %s found in entry/hardware/%s."%(keys, cam))
	default_detector_sizex = cameraSize(["roi_width", "roi_xsize", "sensorsize_x"])
	default_detector_sizey = cameraSize(["roi_height", "roi_ysize", "sensorsize_y"])
else:
	#Default values if there is no h5 file, or it is corrupted and we cannot read it
	default_pixel_size = 1.0
	default_detector_sizex = detector_sizex * ARG.bin_x
	default_detector_sizey = detector_sizey * ARG.bin_y


#If there is nothing else specified
pixel_sizex = default_pixel_size
pixel_sizey = default_pixel_size
if ARG.pixel_sizex is not None:
	pixel_sizex = ARG.pixel_sizex
if ARG.pixel_sizey is not None:
	pixel_sizey = ARG.pixel_sizey
pixel_sizex *= ARG.bin_x
pixel_sizey *= ARG.bin_y
print("default_pixel_size=%f pixel_sizex=%f pixel_sizey=%f"%(default_pixel_size, pixel_sizex, pixel_sizey))

detector_sizex = default_detector_sizex
detector_sizey = default_detector_sizey
xdim_red = default_detector_sizex - detector_sizex % ARG.bin_x
ydim_red = default_detector_sizey - detector_sizey % ARG.bin_y
detector_sizex = xdim_red // ARG.bin_x
detector_sizey = ydim_red // ARG.bin_y
if ARG.detector_sizex is not None:
	detector_sizex = ARG.detector_sizex
if ARG.detector_sizey is not None:
	detector_sizey = ARG.detector_sizey
if ARG.input_file is not None:
	header = DEN.readHeader(ARG.input_file)
	if len(header["dimspec"]) != 3:
		raise TypeError("Dimension of dimspec for file %s shall be 3 but is %d!"%(ARG.input_file, len(header["dimspec"])))
	dimspec = header["dimspec"]
	#Explicit --detector-sizex/--detector-sizey take precedence, e.g. slab matrices for a full projection file
	if ARG.detector_sizex is None:
		detector_sizex = int(dimspec[0])
	elif ARG.detector_sizex != dimspec[0]:
		print("Using --detector-sizex=%d instead of dimx=%d of %s."%(ARG.detector_sizex, dimspec[0], ARG.input_file))
	if ARG.detector_sizey is None:
		detector_sizey = int(dimspec[1])
	elif ARG.detector_sizey != dimspec[1]:
		print("Using --detector-sizey=%d instead of dimy=%d of %s."%(ARG.detector_sizey, dimspec[1], ARG.input_file))
	zdim = int(dimspec[2])
	if zdim != len(df):
		raise TypeError("Number of projections in %s shall be %d according to H5 file but is %d!"%(ARG.input_file, len(df), zdim ))

#Now we compute if there is some offset e.g. due to the binning and correct for it
#We expect that top left corner of the detector matches top left of the camera
#Rotation center offsets (h5/rotation center file) are relative to the center of the unbinned camera frame (ROI)
#of W=default_detector_sizex pixels. Binning by bin_x from the top left corner drops the last W % bin_x
#pixels, so the center of the binned detector lies (W % bin_x)/2 unbinned pixels left of the camera center
#and the rotation axis is (W % bin_x)/2 unbinned pixels further right with respect to it:
#axis_binned_pix = (N-1)/2 + (rotation_center_offset + binning_center_adjustment)/pixel_sizex
#Symmetric crop of the binned detector (--detector-sizex) does not move the center, so it does not enter.
expected_binned_sizex = default_detector_sizex // ARG.bin_x
binning_center_adjustment = 0.0
if default_detector_sizex % ARG.bin_x != 0:
	binning_center_adjustment = (float(default_detector_sizex % ARG.bin_x)*0.5)*default_pixel_size
	if default_detector_sizex//ARG.bin_x >= detector_sizex:
		print("Binning center adjustment due to binning in x by %d is %f"%(ARG.bin_x, binning_center_adjustment))

if detector_sizex < expected_binned_sizex:
	print(colored("Warning: detector_sizex=%d < expected_binned_sizex=%d, any crop shall be symmetric or adequate ARG.detector_center_offsetvx needs to be specified!"%(detector_sizex, expected_binned_sizex), "yellow"))
elif detector_sizex > expected_binned_sizex:
	print(colored("Warning: detector_sizex=%d > expected_binned_sizex=%d, can be due to error!"%(detector_sizex, expected_binned_sizex), "red"))
	raise ValueError("detector_sizex=%d is larger than expected_binned_sizex=%d, can be due to error!"%(detector_sizex, expected_binned_sizex))


def readRotationCenterFile(fileName, camera_pixel_size):
	"""
	Parse rotation_center_offset*=value lines of a detectRotationCenter*.py log.
	Values are relative to the center of the unbinned camera frame (ROI), *_pix* values are in unbinned camera pixels,
	others in physical units. Returns (offset, fit_a, fit_b) in physical units, the fit is
	offset(y_unbinned) = fit_a + fit_b * y_unbinned where y_unbinned is the unbinned camera row.
	"""
	keys = {}
	with open(fileName, "r") as rcf:
		for line in rcf:
			m = re.match(r"^(rotation_center_offset\w*)=\s*(\S+)", line.strip())
			if m is not None:
				try:
					keys[m.group(1)] = float(m.group(2))
				except ValueError:
					pass
	offset = None
	for key, scale in (("rotation_center_offset_pix_interp", camera_pixel_size), ("rotation_center_offset_pix", camera_pixel_size), ("rotation_center_offset", 1.0)):
		if key in keys:
			offset = keys[key] * scale
			break
	fit_a = keys.get("rotation_center_offset_interpfit_a")
	fit_b = keys.get("rotation_center_offset_interpfit_b")
	if (fit_a is None or fit_b is None) and "rotation_center_offset_pix_interpfit_a" in keys and "rotation_center_offset_pix_interpfit_b" in keys:
		fit_a = keys["rotation_center_offset_pix_interpfit_a"] * camera_pixel_size
		fit_b = keys["rotation_center_offset_pix_interpfit_b"] * camera_pixel_size
	return (offset, fit_a, fit_b)

def unbinnedRow(y_binned):
	#Center of binned row y_binned in unbinned camera row coordinates, binning from the top of the camera
	return y_binned * ARG.bin_y + 0.5 * (ARG.bin_y - 1)

rotation_center_offset_interpfit_a = None
rotation_center_offset_interpfit_b = None
if ARG.rotation_center_offset is None and ARG.rotation_center_offset_pix is None and ARG.rotation_center_file is None:
	rotation_center_offset = 0.0
elif ARG.rotation_center_offset is not None:
	rotation_center_offset = ARG.rotation_center_offset
elif ARG.rotation_center_offset_pix is not None:
	#Pixels after binning, relative to the center of the unbinned camera frame (ROI)
	rotation_center_offset = ARG.rotation_center_offset_pix * ARG.bin_x * default_pixel_size
else:
	(rotation_center_offset, rotation_center_offset_interpfit_a, rotation_center_offset_interpfit_b) = readRotationCenterFile(ARG.rotation_center_file, default_pixel_size)
	if rotation_center_offset_interpfit_a is not None and rotation_center_offset_interpfit_b is not None and ARG.rotation_center_file_fit_y is not None:
		a = rotation_center_offset_interpfit_a
		b = rotation_center_offset_interpfit_b
		y = ARG.rotation_center_file_fit_y
		y_unb = unbinnedRow(y)
		rotation_center_offset = a + b * y_unb
		print("For y=%f y_unb=%f rotation_center_offset=%f + %f y_unb = %f"%(y, y_unb, a, b, rotation_center_offset))
	if rotation_center_offset is None:
		raise ValueError("No rotation_center_offset* value found in %s."%(ARG.rotation_center_file))

print("rotation_center_offset=%f rotation_center_offset_pix=%f"%(rotation_center_offset, rotation_center_offset/pixel_sizex))

totalCenterOffsetx = binning_center_adjustment + rotation_center_offset + ARG.detector_center_offsetvx
totalCenterOffsety = ARG.detector_center_offsetvy

print("totalCenterOffsetx=%f rotation_center_offset=%f ARG.detector_center_offsetvx=%f"%(totalCenterOffsetx, rotation_center_offset,  ARG.detector_center_offsetvx))

#Now code analogous to createCameraMatricesForCircularScanTrajectoryParallelRay3D.py

def degToRad(angle):
	return np.pi*angle/180


#PETRA compatibility transform
def JTransform(angle):
	return -angle - 0.5*np.pi

M=float(detector_sizey)
N=float(detector_sizex)
PX=pixel_sizex
PY=pixel_sizey
print("M=%d N=%d PX=%f PY=%f"%(M, N, PX, PY))
directionAngles = df["s_rot"]
directionAngles = [degToRad(x) for x in directionAngles]

def is_nondecreasing(seq):
	return all(x <= y for x, y in zip(seq, seq[1:]))

if ARG.ignore_exact_angles:
	if is_nondecreasing(directionAngles):
		angle_count = len(directionAngles)
		min_angle = directionAngles[0]
		max_angle = directionAngles[-1]
		directionAngles = np.linspace(min_angle, max_angle, angle_count)
		print("Using directionAngles np.linspace(%f, %f, %d)"%(min_angle, max_angle, angle_count))
	else:
		raise ValueError("directionAngles is not a nondecreasing sequence")


if ARG.petra_compatibility_transform:
	directionAngles = [JTransform(x) for x in directionAngles]

#Rows a, b of the projection matrix (detector x, y index per unit of volume coordinates), same for all branches
def detectorRows(theta):
	if ARG.material_ct_convention:
		a = np.array([np.cos(theta)/PX, np.sin(theta)/PX, 0.0], dtype=np.float64)
		b = np.array([0.0, 0.0, 1.0/PY], dtype=np.float64)
	else:
		a = np.array([np.cos(theta)/PX, -np.sin(theta)/PX, 0.0], dtype=np.float64)
		b = np.array([0.0, 0.0, -1.0/PY], dtype=np.float64)
	return (a, b)

fittedSlope = 0.0
if ARG.rotation_center_file_fitted_matrix:
	#Encode slope information into the projection matrix
	if rotation_center_offset_interpfit_a is None or rotation_center_offset_interpfit_b is None:
		print("There is not relevant information about fit.")
		sys.exit(-1)
	#rotation_center_offset(y_unbinned) = a + b * y_unbinned, evaluated at the row imaging the plane z=0,
	#which is the center row (M-1)/2 shifted by totalCenterOffsety/PY rows
	if ARG.rotation_center_file_fit_y is not None:
		center_row = ARG.rotation_center_file_fit_y
	else:
		center_row = 0.5 * (default_detector_sizey // ARG.bin_y - 1)
		if detector_sizey != default_detector_sizey // ARG.bin_y:
			print(colored("Warning: detector_sizey=%d differs from binned camera height %d, assuming centered crop, provide --rotation-center-file-fit-y with the binned camera row of the detector center!"%(detector_sizey, default_detector_sizey // ARG.bin_y), "red"))
	zero_row = center_row + totalCenterOffsety / PY
	offset_zero_row = rotation_center_offset_interpfit_a + rotation_center_offset_interpfit_b * unbinnedRow(zero_row)
	totalCenterOffsetx = binning_center_adjustment + offset_zero_row + ARG.detector_center_offsetvx
	#Axis image moves by fit_b * bin_y physical units per binned detector row
	fittedSlope = rotation_center_offset_interpfit_b * ARG.bin_y / PX
	print("Fitted matrix: center_row=%f rotation_center_offset=%f at the row of z=0, slope %f pixels per detector row"%(center_row, offset_zero_row, fittedSlope))

CameraMatrices = []
for i in range(len(directionAngles)):
	theta = float(directionAngles[i])
	(a, b) = detectorRows(theta)
	px0 = N * 0.5 - 0.5 + totalCenterOffsetx / PX
	if not ARG.ignore_pixel_shifts:
		px0 -= df["pixel_shift"].iloc[i]/ARG.bin_x
	py0 = M * 0.5 - 0.5 + totalCenterOffsety / PY
	if ARG.rotation_center_file_fitted_matrix:
		#Detector row changes by b[2] per unit z, the axis image by fittedSlope per row
		a[2] = fittedSlope * b[2]
	if ARG.flat_matrices:
		CameraMatrices.append([np.append(a, px0)])
	else:
		CameraMatrices.append([np.append(a, px0), np.append(b, py0)])
CameraMatrices = np.array(CameraMatrices, dtype=np.float64)

DEN.storeNdarrayAsDEN(ARG.outputProjectionMatrices, CameraMatrices, force=ARG.force)

if ARG.write_params_file:
	with open(paramsFile, 'w') as f:
		json.dump(ARG.__dict__, f, indent=2, sort_keys=True)
