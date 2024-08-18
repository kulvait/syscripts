#!/usr/bin/env python
"""
Created on 26.6.2024

@author: Vojtech Kulvait

This script takes 360 deg scan and produces 180 deg scan by stitching the images together.

"""

from termcolor import colored
import argparse
import glob
import os
import errno
import sys
import h5py
import math
import re
from denpy import DEN
from denpy import UTILS
from denpy import PETRA
from algotom.prep.conversion import convert_sinogram_360_to_180_enh
import numpy as np
import multiprocessing as mp

#Direction so that theta is ccw to the X axis, returns unit vector
def rayDirection(theta):
	return(np.array([np.sin(theta), -np.cos(theta), 0.], dtype=np.float64))

def degToRad(angle):
	return np.pi*angle/180


#PETRA compatibility transform
def JTransform(angle):
	return -angle - 0.5*np.pi

parser = argparse.ArgumentParser()
parser.add_argument("inputH5", help="H5 file with dataset information")
parser.add_argument("inputSIN", help="H5 file with dataset information")
parser.add_argument("outputSIN", help="H5 file with dataset information")
parser.add_argument("--output-projection-matrices", default=None, help="Output file with projection matrices")
parser.add_argument("--rotation-center-offset", default=0.0, help="Offset of the rotation center, defaults to 0")
parser.add_argument("--detector-center-offsetvx", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0.")
parser.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0.")
parser.add_argument("--rotation-center-file", default=None, help="Use value stored in file to determine rotation center")
parser.add_argument("--rotation-center-file-fit-y", default=None, type=float, help="Perform interpolation fit")
parser.add_argument("--rotation-center-file-fitted-matrix", action="store_true", help="Create skewed projection matrix based on fit data")
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
parser.add_argument("--material-ct-convention", action="store_true", default=True, help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
parser.add_argument("--flat-matrices", help="Create 1D projection matrices.", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--petra-compatibility-transform", action="store_true", help="Make geometry compatible with that used by JM in his reconstruction scripts")
parser.add_argument("--log-file", default=None, help="Output to log file insted of stdout")
parser.add_argument("--override-magnification", default=None)
parser.add_argument("--normalize", action="store_true", help="Use normalize=True in convert_sinogram_360_to_180 function.")
parser.add_argument("--force", action="store_true")
parser.add_argument("--suggested-sinogram-width", default=None, type=str, help="Params file with stitched_sinogram_width=..., if not specified, it is computed from the input sinogram.")
parser.add_argument("--write-params-file", action="store_true")
parser.add_argument('--_json-message', default="Created using KCT script createCameraMatricesForCircularScanTrajectoryParallelRay3D.py", help=argparse.SUPPRESS)
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

directionAngles = df["s_rot"].to_numpy()
directionAnges = directionAngles - directionAngles[0]
directionAngles = np.array([degToRad(x) for x in directionAngles], dtype=np.float64)

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
			print("Path %s exist remove to proceed.", f)
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
	checkFileExistence(paramsFile)
	paramsFile="%s.params"%(ARG.output_projection_matrices)

if "/entry/hardware/camera1" in h5: 
	cam = "camera1"
elif "/entry/hardware/camera" in h5: 
	cam = "camera"
else:
	raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s."%info["h5"])
pix_sizecam = float(h5["entry/hardware/%s/pixelsize"%cam][0])
overrideMagnification = ARG.override_magnification
if overrideMagnification is not None:
	pix_sizemag = float(overrideMagnification)
else:
	pix_sizemag = float(h5["entry/hardware/%s/magnification"%cam][0])
default_pixel_size = float(pix_sizecam/pix_sizemag)
default_detector_sizex = int(h5["entry/hardware/%s/sensorsize_x"%cam][0])
default_detector_sizey = int(h5["entry/hardware/%s/sensorsize_y"%cam][0])

#If there is nothing else specified
pixel_sizex = default_pixel_size
pixel_sizey = default_pixel_size
if ARG.pixel_sizex is not None:
	pixel_sizex = ARG.pixel_sizex
if ARG.pixel_sizey is not None:
	pixel_sizey = ARG.pixel_sizey
pixel_sizex *= ARG.bin_x
pixel_sizey *= ARG.bin_y

detector_sizex = default_detector_sizex
detector_sizey = default_detector_sizey
xdim_red = default_detector_sizex - detector_sizex % ARG.bin_x
ydim_red = default_detector_sizey - detector_sizey % ARG.bin_y
detector_sizex = xdim_red // ARG.bin_x
detector_sizey = ydim_red // ARG.bin_y
if ARG.detector_sizex is not None:
	detector_sizex = int(ARG.detector_sizex)
if ARG.detector_sizey is not None:
	detector_sizey = int(ARG.detector_sizey)

if not os.path.isfile(ARG.inputSIN):
	raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ARG.inputSIN)

#Now we compute if there is some offset e.g. due to the binning and correct for it
#We expect that top left corner of the detector matches top left of the camera

default_detector_halflength = (default_detector_sizex*0.5)*default_pixel_size
detector_halflength = (float(detector_sizex)*0.5)*pixel_sizex
detector_halflength_adjustment = detector_halflength - default_detector_halflength
if np.abs(detector_halflength_adjustment)/default_pixel_size > 0.01:
	print(colored("Halflenght adjustment due to detector_halflength=%f and default_detector_halflength=%f does not match, they are %f of default_pixel_size, can be due to error!"%(detector_halflength, default_detector_halflength, detector_halflength_adjustment/default_pixel_size), "red"))

rotation_center_offset = ARG.rotation_center_offset
rotation_center_offset_interpfit_a = None
rotation_center_offset_interpfit_b = None
if ARG.rotation_center_file is not None:
	rcf = open(ARG.rotation_center_file, 'r')
	for x in rcf.readlines():
		if re.search("^rotation_center_offset=", x):
			rotation_center_offset = float(x.rsplit("=",1)[1])
		if re.search("^rotation_center_offset_interpfit_a=", x):
			rotation_center_offset_interpfit_a = float(x.rsplit("=",1)[1])
		if re.search("^rotation_center_offset_interpfit_b=", x):
			rotation_center_offset_interpfit_b = float(x.rsplit("=",1)[1])
	if rotation_center_offset_interpfit_b is not None and ARG.rotation_center_file_fit_y is not None:
		a = rotation_center_offset_interpfit_a
		b = rotation_center_offset_interpfit_b
		y = ARG.rotation_center_file_fit_y
		rotation_center_offset = a + b * y * ARG.bin_y
		print("For y=%f y_unb=%f rotation_center_offset=%f + %f y_unb = %f"%(y, y*ARG.bin_y, a, b, rotation_center_offset))

print("rotation_center_offset=%f rotation_center_offset_pix=%f"%(rotation_center_offset, rotation_center_offset/pixel_sizex))

totalCenterOffsetx = detector_halflength_adjustment + rotation_center_offset + ARG.detector_center_offsetvx
totalCenterOffsety = ARG.detector_center_offsetvy

print("totalCenterOffsetx=%f rotation_center_offset=%f ARG.detector_center_offsetvx=%f"%(totalCenterOffsetx, rotation_center_offset,  ARG.detector_center_offsetvx))

#Now code analogous to createCameraMatricesForCircularScanTrajectoryParallelRay3D.py
checkFileExistence(ARG.outputSIN)

INF = DEN.readHeader(ARG.inputSIN)
dimspec = INF["dimspec"]
if len(dimspec) < 2 or len(dimspec) > 3:
	raise TypeError("Dimension of dimspec for file %s shall be 2 or 3 but is %d!"%(arg.inputSIN, len(dimspec)))

dimx = dimspec[0]
angleCount = dimspec[1]
if len(dimspec) == 3:
	dimy = dimspec[2]
else:
	dimy = 1

if dimy != detector_sizey:
	print(colored("Y dimension of the input sinogram %s %dx%d does not match the y dimension of the detector %dx%d, using from sinogram!"%(ARG.inputSIN, dimx, dimy, detector_sizex, detector_sizey), "red"))
	detector_sizey = dimy

if dimx != detector_sizex:
	raise ValueError("X dimension of the input sinogram %s %d does not match the dimension of the detector %d!"%(ARG.inputSIN, dimx, detector_sizex))


#Create sequence of center_of_rotation_before_stitiching[y] to represent the center of rotation position in pixels for each y
center_of_rotation_before_stitiching = np.zeros(dimy, dtype=np.float32)
center_of_rotation_after_stitching = np.zeros(dimy, dtype=np.float32)

for k in range(dimy):
	CODPIX = (detector_sizex-1)*0.5
	if rotation_center_offset_interpfit_a is not None and rotation_center_offset_interpfit_b is not None:
		COROFFSET = detector_halflength_adjustment + ARG.detector_center_offsetvx + rotation_center_offset_interpfit_a + rotation_center_offset_interpfit_b * ARG.bin_x * k
	else:
		COROFFSET = detector_halflength_adjustment + ARG.detector_center_offsetvx + rotation_center_offset
	COROFFSETPIX = COROFFSET / pixel_sizex
	center_of_rotation_before_stitiching[k] = CODPIX + COROFFSETPIX

def estimateStitchedSinogramWidth(dimx_orig, center_of_rotation_before_stitiching):
	maxWidth = dimx_orig
	minWidth = dimx_orig*2
	CODPIX = (dimx_orig-1)*0.5
	for k in range(len(center_of_rotation_before_stitiching)):
		#Compute algotom final size of the sinogram
		COR = center_of_rotation_before_stitiching[0]
		overlap = min(2*COR + 1, 2 * (dimx_orig - COR) - 1)
		if overlap - np.floor(overlap) > 0.5:
			outWidth = 2 * dimx_orig - np.floor(overlap) - 1
		else:
			outWidth = 2 * dimx_orig - np.floor(overlap)
		if outWidth > maxWidth:
			maxWidth = outWidth
		if outWidth < minWidth:
			minWidth = outWidth
	return (minWidth, maxWidth)

dimx_stitched = estimateStitchedSinogramWidth(dimx, center_of_rotation_before_stitiching)
if ARG.suggested_sinogram_width is not None:
	with open(ARG.suggested_sinogram_width, 'r') as f:
		for x in f.readlines():
			if re.search("^stitched_sinogram_width=", x):
				dimx_stitched = int(x.rsplit("=",1)[1])
				break
	print("Based on params file for given dimx=%d  dimx_stitched=%d"%(dimx, dimx_stitched))
else:
	if dimx_stitched[0] != dimx_stitched[1]:
		print(colored("Based on the center of rotation and original size dimx=%d the estimate for dimx_stitched=%d-%d, taking minimum"%(dimx, dimx_stitched[0], dimx_stitched[1]), "red"))
	dimx_stitched = dimx_stitched[0]
	print("Based on the center of rotation and original size dimx=%d the estimate for dimx_stitched=%d"%(dimx, dimx_stitched))
angleCountStitched = angleCount // 2 + 1
angleCountCorrected = int(sum(directionAngles<np.pi))

stitchedAngleDifferences = np.pi +  directionAngles[0:angleCountStitched] - directionAngles[-angleCountStitched:]


if angleCountStitched != angleCountCorrected + 1:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while angleCountStitched=%d int(sum(directionAngles<np.pi))=angleCountCorrected=%d"%(angleCountStitched, angleCountCorrected), "red"))
elif np.abs(directionAngles[angleCountCorrected]-np.pi) > 2*np.pi/angleCount:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while abs(angleCountStitched[angleCountCorrected]-np.pi)=%f > 2*np.pi/angleCount=%f"%(np.abs(angleCountStitched[angleCountCorrected]-np.pi), 2*np.pi/angleCount), "red"))
elif angleCount % 2 == 0:
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while angleCount=%d is even"%(angleCount), "red"))
elif not np.all(np.abs(stitchedAngleDifferences) < 2*np.pi/angleCount):
	print(colored("Data are not optimally aligned to use algotom stitching procedure, while not np.all(np.abs(stitchedAnleDifferences) < 2*np.pi/angleCount)", "red"))
else:
	print("Data are optimally aligned to use algotom stitching procedure")


DEN.writeEmptyDEN(ARG.outputSIN, [dimx_stitched, angleCountCorrected, dimy], force=ARG.force)

#Mechanism to write to center_of_rotation_after_stitching and not its copy in individual threads
def insert_npx_value(x):
	(k, v) = x
	center_of_rotation_after_stitching[k] = v

def process_frame(k, inputFile, outputFile, center_of_rotation_before_stitiching, center_of_rotation_after_stitching, dimx_stitched, dimang_correct, verbose=False, normalize=False):
	if verbose and k % 100 == 0:
		print("proces_frame(k=%d, inputFile=%s, outputFile=%s, center_of_rotation_before_stitiching, center_of_rotation_after_stitching, dimx_stitched=%d, dimang_correct=%d, verbose=%s, normalize=%s)"%(k, inputFile, outputFile, dimx_stitched, dimang_correct, verbose, normalize))
	sin = DEN.getFrame(inputFile, k)
	(dimy_sin, dimx_sin) = sin.shape
	CORPIX = center_of_rotation_before_stitiching[k]
	#Apply sigmoid weigting to sin
	weight = np.ones(dimx_sin)
	dimy_stitch = dimy_sin // 2 + 1
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
#START DEBUG
#	sin[:dimang_correct, :] = 0.0
#END DEBUG
	#I want to stop algothom from any weighting attempts
	#w = np.ones((dimy_stitch, dimx_sin))
	#(stsin, NEWCORPIX) = convert_sinogram_360_to_180(sin, CORPIX, total_width=dimx_stitched, norm=normalize, wei_mat1 = w, wei_mat2 = w)
	(stsin, NEWCORPIX) = convert_sinogram_360_to_180_enh(sin, CORPIX, total_width=dimx_stitched, norm=normalize, noweighting=True)
	stsin = stsin[:dimang_correct]
	DEN.writeFrame(outputFile, k, stsin, force=True)
	if verbose and k % 100 == 0:
		print("Processed frame %d center_of_rotation_after_stitching=%f"%(k, NEWCORPIX))
	return (k, NEWCORPIX)

pool = mp.Pool(processes=ARG.threads)

#process_frame(0, ARG.inputSIN, ARG.outputSIN, center_of_rotation_before_stitiching, center_of_rotation_after_stitching, dimx_stitched, angleCountCorrected, ARG.verbose, ARG.normalize)
#exit(0)
for k in range(dimy):
	pool.apply_async(process_frame, args=(k, ARG.inputSIN, ARG.outputSIN, center_of_rotation_before_stitiching, center_of_rotation_after_stitching, dimx_stitched, angleCountCorrected, ARG.verbose, ARG.normalize), callback=insert_npx_value)

pool.close()
pool.join()

if ARG.verbose:
	print("All frames stitched")

#Test if all elements in center_of_rotation_after_stitching are the same
if not np.all(center_of_rotation_after_stitching == center_of_rotation_after_stitching[0]):
	print(colored("All elements in center_of_rotation_after_stitching are not the same, this is suspicious values in center_of_rotation_after_stitching=%s!"%(list(map(center_of_rotation_after_stitching))), "red"))
	raise ValueError("All elements in center_of_rotation_after_stitching are not the same!")

stitched_maxangle = directionAngles[angleCountCorrected-1]

if stitched_maxangle > np.pi:
	print(colored("Stitched max angle is %fpi, which is larger than pi, this is suspicious!"%(stitched_maxangle/np.pi), "red"))
else:
	print("Stitched max angle is %fpi"%(stitched_maxangle/np.pi))

NEWCORMIN = center_of_rotation_after_stitching.min()
NEWCORMAX = center_of_rotation_after_stitching.max()
if NEWCORMIN != NEWCORMAX:
	print(colored("Center of rotation after stitching is not constant, NEWCORMIN=%f NEWCORMAX=%f"%(NEWCORMIN, NEWCORMAX), "red"))

NEWCORPIX = center_of_rotation_after_stitching[0]

if ARG.output_projection_matrices is not None:
	M=float(detector_sizey)
	N=float(dimx_stitched)
	PX=pixel_sizex
	PY=pixel_sizey
	print("M=%d N=%d PX=%f PY=%f"%(M, N, PX, PY))
	if ARG.petra_compatibility_transform:
		directionAngles = [JTransform(x) for x in directionAngles]
	CameraMatrices = np.zeros((0,2,4), dtype=np.float64)
	for i in range(angleCountCorrected):
			theta = float(directionAngles[i])
			VR = rayDirection(theta)
			VX = np.array([np.cos(theta)*PX, np.sin(theta)*PX, 0.0], dtype=np.float64)
			a = np.array([np.cos(theta)/PX, np.sin(theta)/PX, 0.0], dtype=np.float64)
			if ARG.material_ct_convention:
				VY = np.array([0.0, 0.0, PY], dtype=np.float64)
				b = np.array([0.0, 0.0, 1.0/PY], dtype=np.float64)
			else:
				VY = np.array([0.0, 0.0, -PY], dtype=np.float64)
				b = np.array([0.0, 0.0, -1.0/PY], dtype=np.float64)
			px0 = NEWCORPIX
			py0 = M * 0.5 - 0.5
			CM = np.array([np.append(a, px0), np.append(b,py0)])
			CM.shape=(1,2,4)
			CameraMatrices = np.concatenate((CameraMatrices, CM))
	print("Storing projection matrices to %s"%(os.path.abspath(ARG.output_projection_matrices)))
	DEN.storeNdarrayAsDEN(ARG.output_projection_matrices, CameraMatrices, ARG.force)

print("Stored projection matrices to %s"%(os.path.abspath(ARG.output_projection_matrices)))

if ARG.write_params_file:
	with open(paramsFile, 'w') as f:
		json.dump(ARG.__dict__, f, indent=2, sort_keys=True)
