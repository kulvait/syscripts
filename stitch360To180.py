#!/usr/bin/env python
"""
Created on 26.6.2024

@author: Vojtech Kulvait

This script takes 360 deg scan and produces 180 deg scan by stitching the images together.

Rotation center convention is the one of createProjectionMatrices.py: rotation_center_offset is the offset of
the rotation axis image from the center of the unbinned camera frame (ROI) in physical units, positive towards higher
detector x index. On the binned detector of N pixels the axis is imaged at binned pixel index
(N-1)/2 + (rotation_center_offset + binning_center_adjustment + detector_center_offsetvx)/pixel_sizex.
"""

from termcolor import colored
import argparse
import os
import errno
import sys
import json
import h5py
import re
from denpy import DEN
from denpy import PETRA
from algotom.prep.conversion import convert_sinogram_360_to_180_enh
import numpy as np
import multiprocessing as mp

def degToRad(angle):
	return np.pi*angle/180


#PETRA compatibility transform
def JTransform(angle):
	return -angle - 0.5*np.pi

parser = argparse.ArgumentParser()
parser.add_argument("inputH5", help="H5 file with dataset information")
parser.add_argument("inputSIN", help="DEN file with 360 deg sinograms of dimensions [dimx, angles, rows]")
parser.add_argument("outputSIN", help="DEN file to store stitched 180 deg sinograms")
parser.add_argument("--output-projection-matrices", default=None, help="Output file with projection matrices")
parser.add_argument("--rotation-center-offset", default=0.0, type=float, help="Offset of the rotation axis from the center of the unbinned camera frame (ROI) in physical units of the pixel size, positive towards higher detector x index, defaults to 0. Overridden by --rotation-center-file.")
parser.add_argument("--detector-center-offsetvx", type=float, default=0., help="Additional offset in physical units, added to the rotation center offset as in createProjectionMatrices.py, defaults to 0.0.")
parser.add_argument("--detector-center-offsetvy", type=float, default=0., help="Unused, the stitched projection matrices place the plane z=0 at the center row (M-1)/2.")
parser.add_argument("--rotation-center-file", default=None, help="Log file of detectRotationCenter*.py with rotation_center_offset*=value lines, values relative to the center of the unbinned camera frame (ROI), *_pix* values in unbinned camera pixels. Priority: linear fit evaluated at --rotation-center-file-fit-y, rotation_center_offset_pix_interp, rotation_center_offset_pix, rotation_center_offset.")
parser.add_argument("--rotation-center-file-fit-y", default=None, type=float, help="Binned detector row (bin_y camera rows per row, row 0 at the top of the camera) at which the linear fit rotation_center_offset_interpfit_a + rotation_center_offset_interpfit_b * y_unbinned from --rotation-center-file is evaluated for all rows.")
parser.add_argument("--rotation-center-file-fitted-matrix", action="store_true", help="Stitch every sinogram row with its own rotation center from the linear fit in --rotation-center-file and encode the resulting slope into skewed projection matrices.")
parser.add_argument("--sinogram-first-row", default=0, type=int, help="Binned camera row of the first row of the input sinogram, needed with --rotation-center-file-fitted-matrix when the sinograms are cropped in y, defaults to 0.")
parser.add_argument("--pixel-sizex", default=None, type=float, help="Pixel size x to use for projection geometry.")
parser.add_argument("--pixel-sizey", default=None, type=float, help="Pixel size y to use for projection geometry.")
parser.add_argument("--detector-sizex", default=None, help="If there is input file, this is ignored and dimx used instead")
parser.add_argument("--detector-sizey", default=None, help="If there is input file, this is ignored and dimy used instead")
parser.add_argument("--bin-x",
                    type=int,
                    default=1,
                    help="X dimension of binning box, pixel size will be multiplied by this factor and default detector size is adjusted accordingly.")
parser.add_argument("--bin-y",
                    type=int,
                    default=1,
                    help="Y dimension of binning box, pixel size will be multiplied by this factor and default detector size is adjusted accordingly.")
parser.add_argument("--material-ct-convention", action=argparse.BooleanOptionalAction, default=True, help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis, use --no-material-ct-convention to disable. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--petra-compatibility-transform", action="store_true", help="Make geometry compatible with that used by JM in his reconstruction scripts")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
parser.add_argument("--override-magnification", default=None)
parser.add_argument("--normalize", action="store_true", help="Use normalize=True in convert_sinogram_360_to_180 function.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--suggested-sinogram-width", default=None, type=str, help="Params file with stitched_sinogram_width=..., if not specified, it is computed from the input sinogram.")
parser.add_argument("--write-params-file", action="store_true", help="Write arguments as JSON to <output>.params, output is --output-projection-matrices if given, else outputSIN.")
parser.add_argument('--_json-message', default="Created using KCT script stitch360To180.py", help=argparse.SUPPRESS)
parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")
ARG = parser.parse_args()

if ARG.log_file:
	sys.stdout = open(ARG.log_file, "wt")

if ARG.threads == -1:
	ARG.threads = mp.cpu_count()
	if ARG.verbose:
		print("Stitching 360 sinogram to 180 from %s writing to %s with threads=mp.cpu_count()=%d."%(ARG.inputSIN, ARG.outputSIN, ARG.threads))


if ARG.override_magnification is not None:
	df = PETRA.imageDataset(ARG.inputH5, includePixelShift=True, overrideMagnification=float(ARG.override_magnification))
else:
	df = PETRA.imageDataset(ARG.inputH5, includePixelShift=True)

#Absolute angles are used in the projection matrices so that the geometry matches createProjectionMatrices.py,
#angles relative to the first one are used to select the rows of the stitched sinogram
directionAngles = np.array([degToRad(x) for x in df["s_rot"].to_numpy()], dtype=np.float64)
relativeAngles = directionAngles - directionAngles[0]

if "pixel_shift" in df and np.any(df["pixel_shift"].to_numpy() != 0):
	print(colored("Warning: nonzero pixel shifts in %s are not taken into account, stitching assumes constant rotation center!"%(ARG.inputH5), "red"))

h5 = h5py.File(ARG.inputH5, 'r')

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
			print("Path %s exist remove to proceed."%f)
			sys.exit(1)

def sigmoid(x, scale = 3):
	return 1 / (1 + np.exp(-x*scale))


def scaled_sigmoid(x, scale = 5):
	minval = sigmoid(-1.0, scale)
	sigval = sigmoid(x, scale)
	return (sigval - minval) / (1.0 - 2.0*minval)

#Variable d is relative distance from the COR and shall be in the range -1, 1 for weighting to take effect
def weighting_scheme(d):
	if d < -1.0:
		return 0.0
	elif d > 1.0:
		return 1.0
	else:
		return scaled_sigmoid(d)

if ARG.output_projection_matrices is not None:
	checkFileExistence(ARG.output_projection_matrices)
if ARG.write_params_file:
	if ARG.output_projection_matrices is not None:
		paramsFile = "%s.params"%(ARG.output_projection_matrices)
	else:
		paramsFile = "%s.params"%(ARG.outputSIN)
	checkFileExistence(paramsFile)

if "/entry/hardware/camera1" in h5:
	cam = "camera1"
elif "/entry/hardware/camera" in h5:
	cam = "camera"
else:
	raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."%ARG.inputH5)
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

#If there is nothing else specified
pixel_sizex = default_pixel_size
pixel_sizey = default_pixel_size
if ARG.pixel_sizex is not None:
	pixel_sizex = ARG.pixel_sizex
if ARG.pixel_sizey is not None:
	pixel_sizey = ARG.pixel_sizey
pixel_sizex *= ARG.bin_x
pixel_sizey *= ARG.bin_y

detector_sizex = default_detector_sizex // ARG.bin_x
detector_sizey = default_detector_sizey // ARG.bin_y
if ARG.detector_sizex is not None:
	detector_sizex = int(ARG.detector_sizex)
if ARG.detector_sizey is not None:
	detector_sizey = int(ARG.detector_sizey)

if not os.path.isfile(ARG.inputSIN):
	raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ARG.inputSIN)

#Now we compute if there is some offset e.g. due to the binning and correct for it, same as createProjectionMatrices.py
#We expect that top left corner of the detector matches top left of the camera
#Rotation center offsets are relative to the center of the unbinned camera frame (ROI) of W=default_detector_sizex pixels.
#Binning by bin_x from the top left corner drops the last W % bin_x pixels, so the center of the binned detector
#lies (W % bin_x)/2 unbinned pixels left of the camera center and the rotation axis (W % bin_x)/2 unbinned
#pixels further right with respect to it. Symmetric crop of the binned detector does not move the center.
expected_binned_sizex = default_detector_sizex // ARG.bin_x
binning_center_adjustment = (float(default_detector_sizex % ARG.bin_x)*0.5)*default_pixel_size
if binning_center_adjustment != 0.0:
	print("Binning center adjustment due to binning in x by %d is %f"%(ARG.bin_x, binning_center_adjustment))
if detector_sizex < expected_binned_sizex:
	print(colored("Warning: detector_sizex=%d < expected_binned_sizex=%d, any crop shall be symmetric or adequate ARG.detector_center_offsetvx needs to be specified!"%(detector_sizex, expected_binned_sizex), "yellow"))
elif detector_sizex > expected_binned_sizex:
	raise ValueError("detector_sizex=%d is larger than expected_binned_sizex=%d, can be due to error!"%(detector_sizex, expected_binned_sizex))

def readRotationCenterFile(fileName, camera_pixel_size):
	"""
	Parse rotation_center_offset*=value lines of a detectRotationCenter*.py log, same as createProjectionMatrices.py.
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

rotation_center_offset = ARG.rotation_center_offset
rotation_center_offset_interpfit_a = None
rotation_center_offset_interpfit_b = None
if ARG.rotation_center_file is not None:
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
if ARG.rotation_center_file_fitted_matrix and (rotation_center_offset_interpfit_a is None or rotation_center_offset_interpfit_b is None):
	print("There is not relevant information about fit, --rotation-center-file-fitted-matrix requires rotation_center_offset_interpfit_a/b in --rotation-center-file.")
	sys.exit(-1)

print("rotation_center_offset=%f rotation_center_offset_pix=%f"%(rotation_center_offset, rotation_center_offset/pixel_sizex))

totalCenterOffsetx = binning_center_adjustment + rotation_center_offset + ARG.detector_center_offsetvx

print("totalCenterOffsetx=%f rotation_center_offset=%f ARG.detector_center_offsetvx=%f"%(totalCenterOffsetx, rotation_center_offset,  ARG.detector_center_offsetvx))

checkFileExistence(ARG.outputSIN)

INF = DEN.readHeader(ARG.inputSIN)
dimspec = INF["dimspec"]
if len(dimspec) < 2 or len(dimspec) > 3:
	raise TypeError("Dimension of dimspec for file %s shall be 2 or 3 but is %d!"%(ARG.inputSIN, len(dimspec)))

dimx = dimspec[0]
angleCount = dimspec[1]
if len(dimspec) == 3:
	dimy = dimspec[2]
else:
	dimy = 1

if angleCount != len(directionAngles):
	raise ValueError("Number of angles %d in %s does not match %d angles in %s!"%(angleCount, ARG.inputSIN, len(directionAngles), ARG.inputH5))

if dimy != detector_sizey:
	print(colored("Y dimension of the input sinogram %s %dx%d does not match the y dimension of the detector %dx%d, using from sinogram!"%(ARG.inputSIN, dimx, dimy, detector_sizex, detector_sizey), "red"))
	detector_sizey = dimy

if dimx != detector_sizex:
	raise ValueError("X dimension of the input sinogram %s %d does not match the dimension of the detector %d!"%(ARG.inputSIN, dimx, detector_sizex))


#Create sequence of center_of_rotation_before_stitiching[y] to represent the center of rotation position in pixels for each y
center_of_rotation_before_stitiching = np.zeros(dimy, dtype=np.float64)
center_of_rotation_after_stitching = np.zeros(dimy, dtype=np.float64)

CODPIX = (detector_sizex-1)*0.5
for k in range(dimy):
	if ARG.rotation_center_file_fitted_matrix:
		COROFFSET = binning_center_adjustment + ARG.detector_center_offsetvx + rotation_center_offset_interpfit_a + rotation_center_offset_interpfit_b * unbinnedRow(ARG.sinogram_first_row + k)
	else:
		COROFFSET = totalCenterOffsetx
	COROFFSETPIX = COROFFSET / pixel_sizex
	center_of_rotation_before_stitiching[k] = CODPIX + COROFFSETPIX

def estimateStitchedSinogramWidth(dimx_orig, center_of_rotation_before_stitiching):
	maxWidth = dimx_orig
	minWidth = dimx_orig*2
	for k in range(len(center_of_rotation_before_stitiching)):
		#Compute algotom final size of the sinogram
		COR = center_of_rotation_before_stitiching[k]
		overlap = min(2*COR + 1, 2 * (dimx_orig - COR) - 1)
		if overlap - np.floor(overlap) > 0.5:
			outWidth = 2 * dimx_orig - np.floor(overlap) - 1
		else:
			outWidth = 2 * dimx_orig - np.floor(overlap)
		if outWidth > maxWidth:
			maxWidth = outWidth
		if outWidth < minWidth:
			minWidth = outWidth
	return (int(minWidth), int(maxWidth))

dimx_stitched = estimateStitchedSinogramWidth(dimx, center_of_rotation_before_stitiching)
if ARG.suggested_sinogram_width is not None:
	with open(ARG.suggested_sinogram_width, 'r') as f:
		for x in f.readlines():
			if re.search("^stitched_sinogram_width=", x):
				dimx_stitched = int(x.rsplit("=",1)[1])
				break
	if isinstance(dimx_stitched, tuple):
		raise ValueError("No stitched_sinogram_width= found in %s."%(ARG.suggested_sinogram_width))
	print("Based on params file for given dimx=%d  dimx_stitched=%d"%(dimx, dimx_stitched))
else:
	if dimx_stitched[0] != dimx_stitched[1]:
		#Rows with a smaller overlap need a wider output, narrower rows are padded to the common width
		print(colored("Based on the center of rotation and original size dimx=%d the estimate for dimx_stitched=%d-%d, taking maximum"%(dimx, dimx_stitched[0], dimx_stitched[1]), "red"))
	dimx_stitched = dimx_stitched[1]
	print("Based on the center of rotation and original size dimx=%d the estimate for dimx_stitched=%d"%(dimx, dimx_stitched))
angleCountStitched = angleCount // 2 + 1
angleCountCorrected = int(sum(relativeAngles<np.pi))

stitchedAngleDifferences = np.pi +  relativeAngles[0:angleCountStitched] - relativeAngles[-angleCountStitched:]


if angleCountStitched != angleCountCorrected + 1:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while angleCountStitched=%d int(sum(relativeAngles<np.pi))=angleCountCorrected=%d"%(angleCountStitched, angleCountCorrected), "red"))
elif np.abs(relativeAngles[angleCountCorrected]-np.pi) > 2*np.pi/angleCount:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while abs(relativeAngles[angleCountCorrected]-np.pi)=%f > 2*np.pi/angleCount=%f"%(np.abs(relativeAngles[angleCountCorrected]-np.pi), 2*np.pi/angleCount), "red"))
elif angleCount % 2 == 0:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while angleCount=%d is even"%(angleCount), "red"))
elif not np.all(np.abs(stitchedAngleDifferences) < 2*np.pi/angleCount):
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while not np.all(np.abs(stitchedAngleDifferences) < 2*np.pi/angleCount)", "red"))
else:
	print("Data are optimally aligned to use algotom stitching procedure")


DEN.writeEmptyDEN(ARG.outputSIN, [dimx_stitched, angleCountCorrected, dimy], force=ARG.force)

def process_frame(k, inputFile, outputFile, center_of_rotation_before_stitiching, dimx_stitched, dimang_correct, verbose=False, normalize=False):
	if verbose and k % 100 == 0:
		print("proces_frame(k=%d, inputFile=%s, outputFile=%s, center_of_rotation_before_stitiching, dimx_stitched=%d, dimang_correct=%d, verbose=%s, normalize=%s)"%(k, inputFile, outputFile, dimx_stitched, dimang_correct, verbose, normalize))
	sin = DEN.getFrame(inputFile, k)
	(dimy_sin, dimx_sin) = sin.shape
	CORPIX = center_of_rotation_before_stitiching[k]
	#Apply sigmoid weigting to sin
	weight = np.ones(dimx_sin)
	CODPIX = (dimx_sin-1)*0.5
	if CORPIX < CODPIX:
		radius = CORPIX + 0.5
		maxindex = min(dimx_sin, int(2*CORPIX + 1))
		for i in range(maxindex):
			weight[i] = scaled_sigmoid((i - CORPIX)/radius)
	else:
		radius = dimx_sin - CORPIX - 0.5
		minindex = min(dimx_sin - 1, max(0, int(dimx_sin - 0.5 - 2 * radius)))
		for i in range(minindex, dimx_sin):
			weight[i] = scaled_sigmoid((CORPIX-i)/radius)
	sin = sin * weight #This works https://stackoverflow.com/questions/22934219/numpy-multiply-arrays-rowwise/78682115#78682115
	#I want to stop algothom from any weighting attempts
	(stsin, NEWCORPIX) = convert_sinogram_360_to_180_enh(sin, CORPIX, total_width=dimx_stitched, norm=normalize, noweighting=True)
	stsin = stsin[:dimang_correct]
	DEN.writeFrame(outputFile, k, stsin, force=True)
	if verbose and k % 100 == 0:
		print("Processed frame %d center_of_rotation_after_stitching=%f"%(k, NEWCORPIX))
	return (k, NEWCORPIX)

frameArgs = [(k, ARG.inputSIN, ARG.outputSIN, center_of_rotation_before_stitiching, dimx_stitched, angleCountCorrected, ARG.verbose, ARG.normalize) for k in range(dimy)]
if ARG.threads == 0:
	results = [process_frame(*x) for x in frameArgs]
else:
	with mp.Pool(processes=ARG.threads) as pool:
		#get() propagates exceptions from the workers
		results = [r.get() for r in [pool.apply_async(process_frame, args=x) for x in frameArgs]]
for (k, v) in results:
	center_of_rotation_after_stitching[k] = v

if ARG.verbose:
	print("All frames stitched")

stitched_maxangle = relativeAngles[angleCountCorrected-1]

if stitched_maxangle > np.pi:
	print(colored("Stitched max angle is %fpi, which is larger than pi, this is suspicious!"%(stitched_maxangle/np.pi), "red"))
else:
	print("Stitched max angle is %fpi"%(stitched_maxangle/np.pi))

#Rotation axis image in the stitched sinogram NEWCORPIX(k) = NEWCORPIX_center + NEWCORSLOPE * (k - (dimy-1)/2)
rows = np.arange(dimy, dtype=np.float64) - 0.5 * (dimy - 1)
if ARG.rotation_center_file_fitted_matrix and dimy > 1:
	(NEWCORSLOPE, NEWCORPIX) = np.polyfit(rows, center_of_rotation_after_stitching, 1)
else:
	NEWCORSLOPE = 0.0
	NEWCORPIX = float(np.mean(center_of_rotation_after_stitching))
NEWCORRESIDUAL = np.abs(center_of_rotation_after_stitching - (NEWCORPIX + NEWCORSLOPE * rows)).max()
if NEWCORRESIDUAL > 1e-3:
	message = "Center of rotation after stitching %s with max deviation %f pixels from the %s, values in center_of_rotation_after_stitching=%s!"%("is not linear in rows" if ARG.rotation_center_file_fitted_matrix else "is not constant", NEWCORRESIDUAL, "linear fit" if ARG.rotation_center_file_fitted_matrix else "mean", list(center_of_rotation_after_stitching))
	print(colored(message, "red"))
	if not ARG.rotation_center_file_fitted_matrix:
		raise ValueError(message)
print("Center of rotation after stitching NEWCORPIX=%f at the center row, slope %f pixels per row"%(NEWCORPIX, NEWCORSLOPE))

if ARG.output_projection_matrices is not None:
	M=float(detector_sizey)
	N=float(dimx_stitched)
	PX=pixel_sizex
	PY=pixel_sizey
	print("M=%d N=%d PX=%f PY=%f"%(M, N, PX, PY))
	if ARG.petra_compatibility_transform:
		directionAngles = [JTransform(x) for x in directionAngles]
	CameraMatrices = []
	for i in range(angleCountCorrected):
		theta = float(directionAngles[i])
		#Same rows as createProjectionMatrices.py
		if ARG.material_ct_convention:
			a = np.array([np.cos(theta)/PX, np.sin(theta)/PX, 0.0], dtype=np.float64)
			b = np.array([0.0, 0.0, 1.0/PY], dtype=np.float64)
		else:
			a = np.array([np.cos(theta)/PX, -np.sin(theta)/PX, 0.0], dtype=np.float64)
			b = np.array([0.0, 0.0, -1.0/PY], dtype=np.float64)
		#Detector row changes by b[2] per unit z, the axis image by NEWCORSLOPE per row
		a[2] = NEWCORSLOPE * b[2]
		px0 = NEWCORPIX
		py0 = M * 0.5 - 0.5
		CameraMatrices.append([np.append(a, px0), np.append(b, py0)])
	CameraMatrices = np.array(CameraMatrices, dtype=np.float64)
	print("Storing projection matrices to %s"%(os.path.abspath(ARG.output_projection_matrices)))
	DEN.storeNdarrayAsDEN(ARG.output_projection_matrices, CameraMatrices, ARG.force)
	print("Stored projection matrices to %s"%(os.path.abspath(ARG.output_projection_matrices)))

if ARG.write_params_file:
	with open(paramsFile, 'w') as f:
		json.dump(ARG.__dict__, f, indent=2, sort_keys=True)
