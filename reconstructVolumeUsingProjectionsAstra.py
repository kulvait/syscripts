#!/usr/bin/env python3
"""
Created on Fri Feb 11 11:05:28 2022

@author: Vojtech Kulvait
"""

import astra
#import mpi#Volume decomposition

import os
import argparse
from libtiff import TIFF
import numpy as np
from denpy import DEN
from PIL import Image
import re
import json
import glob
import time
import scipy.io

#folderPrefix = "/home/user/exp/DESY_global"
#Can be mountpoint on home computer
inputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03"
outputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_folder"
inputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03"

inputFolder = "/home/kulvaitv/exp/PtNiWire/scratch_cc/ivw0032_Referenz_blau_4_000/flat_corrected/rawBin2"
outputFolder = "/home/kulvaitv/exp/PtNiWire/scratch_cc/kulvait_scratch/ivw0032_Referenz_blau_4_000/astra"

parser = argparse.ArgumentParser()
parser.add_argument("inputProjections", help="Input projection data. In case of file it is assumed to be den file with projections. In case of directory it is assumed to contain tif projections.")
parser.add_argument("--cgls", action="store_true")
parser.add_argument("--sirt", action="store_true")
parser.add_argument("--sart", action="store_true")
parser.add_argument("--em", action="store_true")
parser.add_argument("--fbp", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--saveden", action="store_true")
parser.add_argument("--savetiff", action="store_true")
parser.add_argument("--neglog", action="store_true")
parser.add_argument("--suffix", default="")
parser.add_argument("--output-folder", default=".")
parser.add_argument("--store-projections", default=None)
parser.add_argument("--angles-mat", default=None)
parser.add_argument("--theta-zero", type=float, default=0, help="Initial angle theta from Radon transform in degrees, defaults to zero. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html")
parser.add_argument("--theta-angular-range", type=float, default=360, help="This is angular range in degrees, along which possitions are distributed.")
parser.add_argument("--material-ct-convention", action="store_true", default=False, help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
parser.add_argument("--itterations", type=int, default=100)
parser.add_argument("--platform-id", type=str, default="0:0")
#parser.add_argument('--box-sizex',
#                    help="Bounding box dimension.",
#                    type=float,
#                    default=5.0)
#parser.add_argument('--box-sizey',
#                    help="Bounding box dimension.",
#                    type=float,
#                    default=5.0)
#parser.add_argument('--box-sizez',
#                    help="Bounding box dimension.",
#                    type=float,
#                    default=5.0)
parser.add_argument('--voxel-sizex',
                    help="Volume dimension x, defaults to pixel-sizex.",
                    type=float,
                    default=0.0125)
parser.add_argument('--voxel-sizey',
                    help="Volume dimension y, defaults to pixel-sizex.",
                    type=float,
                    default=0.0125)
parser.add_argument('--voxel-sizez',
                    help="Voxel size z, defaults to pixel-sizey.",
                    type=float,
                    default=None)
parser.add_argument('--volume-sizex',
                    help="Volume dimension x.",
                    type=int,
                    default=1024)
parser.add_argument('--volume-sizey',
                    help="Volume dimension y.",
                    type=int,
                    default=1024)
parser.add_argument('--volume-sizez',
                    help="Voxel size z, defaults to row_count.",
                    type=int,
                    default=None)
parser.add_argument('--pixel-sizex',
                    help="Detector dimension x.",
                    type=float,
                    default=0.00255076)
parser.add_argument('--pixel-sizey',
                    help="Detector dimension y.",
                    type=float,
                    default=0.00255076)
parser.add_argument('--yrange-from', type=int, default=None)
parser.add_argument('--yrange-to', type=int, default=None)
parser.add_argument("--detector-center-offsetvx", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0.")
parser.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0.")

ARG = parser.parse_args()
#sin=sin[1::2]
#angles=angles[1::2]

if not (ARG.savetiff or ARG.saveden):
    parser.error('Provide either --saveden or --savetiff.')


def parsePlatformString(platformId):
	if ":" not in platformId:
		raise Exception("Invalid platform string")
	tk = platformId.rsplit(":")
	platform = tk[0]
	gpu = tk[1]
	return int(tk[1])

gpuid = parsePlatformString(ARG.platform_id)
astra.astra.set_gpu_index(gpuid)

def degToRad(angle):
	return np.pi*angle/180

def getFileNum(filePath):
	numberSearch = re.search(r"(\d*).tif", filePath)
	if numberSearch is None:
		return ""
	else:
		suf = numberSearch.group(1)
		if suf == "":
			return ""
		else:
			return "%05d" % int(suf)


#Try to get reconlog.txt from tiff file
def getReconLog(tifFile):
	logDir = os.path.dirname(os.path.dirname(tifFile))
	log = os.path.join(logDir, "reconlog.txt")
	if os.path.isfile(log):
		return log
	else:
		return None


def parseParamFile(logFile):
	with open(logFile, "r") as f:
		lines = f.readlines()
		dct = {}
		for l in lines:
			tokens = l.split(":", 1)
			if len(tokens) != 2:
				continue
			dct[tokens[0].strip()] = tokens[1].strip()
		return dct
	return None

def writeDenFile(volume, denFile, force=False):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	dimy = volume.shape[1]
	dimx = volume.shape[2]
	dimz = volume.shape[0]
	DEN.writeEmptyDEN(denFile, [dimx, dimy, dimz], force=force)
	for k in range(dimz):
		DEN.writeFrame(denFile, k, volume[k, :, :], force=force)

def writeTiffFiles(volume, tiffFilePattern, force=False):
	dimy = volume.shape[1]
	dimx = volume.shape[2]
	dimz = volume.shape[0]
	for k in range(dimz):
		if ARG.yrange_from is not None:
			tiffFile = "%s_%05d.tiff" % (tiffFilePattern, k + ARG.yrange_from)
		else:
			tiffFile = "%s_%05d.tiff" % (tiffFilePattern, k)
		if os.path.exists(tiffFile):
			if force:
				os.remove(tiffFile)
			else:
				raise IOError("File %s exists, add force to overwrite" % (tiffFile))
		im = Image.fromarray(volume[k,:,:], mode='F')  # float32
		im.save(tiffFile, "TIFF")

def createProjectorConfig(projectorName, projectionsID, volumeID, usegpu=True):
	#cfg = {}
	if usegpu:
		name = "%s_CUDA" % projectorName
		#cfg["type"]="%s_CUDA"%projectorName
	else:
		name = projectorName
		#cfg["type"]=projectorName
	cfg = astra.astra_dict(name)
	cfg["ProjectionDataId"] = projectionsID
	cfg["ReconstructionDataId"] = volumeID
	return cfg


def generateAstraParallel3d_vec(angles, det_width, det_height, offsetvx=None, offsetvy=None, material_ct_convention = None):
	if offsetvx is None:
		offsetvx = 0.
	if offsetvy is None:
		offsetvy = 0.
	if material_ct_convention is None:
		material_ct_convention = False
	vectors = np.zeros((len(angles), 12))
	for i in range(angles_count):
		theta = angles[i]
		#From https://www.astra-toolbox.com/docs/geom3d.html
		#ray direction
		vectors[i, 0] = np.sin(theta)
		vectors[i, 1] = -np.cos(theta)
		vectors[i, 2] = 0.
		#center of detector
		vectors[i, 3] = 0. + np.cos(theta)*offsetvx
		vectors[i, 4] = 0. + np.sin(theta)*offsetvx
		vectors[i, 5] = offsetvy
		# vector from detector pixel (0,0) to (0,1)
		vectors[i, 6] = np.cos(theta) * det_width
		vectors[i, 7] = np.sin(theta) * det_width
		vectors[i, 8] = 0
		#vector from detector pixel (0,0) to (1,0)
		vectors[i, 9] = 0
		vectors[i, 10] = 0
		if material_ct_convention:
			vectors[i, 11] = det_height
		else:
			vectors[i, 11] = -det_height
	return (vectors)

def transformToExtinction(invertedProjectionIntensities):
	#DEBUG ... another scaling
	#invertedProjectionIntensities = invertedProjectionIntensities/invertedProjectionIntensities.max()
	#END DEBUG
	invertedProjectionIntensities = np.log(
	    np.reciprocal(invertedProjectionIntensities))  #Compute extinction
	#offset = 1-1/invertedProjectionIntensities.max()
	#invertedProjectionIntensities = np.log(np.reciprocal(invertedProjectionIntensities)+offset)
	return (invertedProjectionIntensities)


#print("Log file is %s"%(logFile))
#print(dct)
#=================INPUT PROJECTION DATA===============
sec = time.time()
if os.path.isdir(ARG.inputProjections):
	pth=os.path.join(ARG.inputProjections, "*.tif")
	tifFiles = glob.glob(pth)
	tifFiles.sort()
	if len(tifFiles) == 0:
		raise(IOError("The path %s contains %d *.tif projections."%(pth, len(tifFiles))))
	print("The path %s contains %d *.tif projections."%(pth, len(tifFiles)))
	tif = TIFF.open(tifFiles[0])
	img = tif.read_image()
	row_count = img.shape[0]
	col_count = img.shape[1]
	angles_count = len(tifFiles)
	if ARG.verbose:
		print("First projection %s has dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."% (tifFiles[0], img.shape[0], img.shape[1], img.dtype, img.min(), img.max(), img.mean()))
	if ARG.yrange_from is not None:
		row_count = ARG.yrange_to - ARG.yrange_from
	projectionData = np.zeros(shape=(row_count, angles_count, col_count), dtype=np.float32)
	for i in range(len(tifFiles)):
		f = tifFiles[i]
		img = TIFF.open(f)
		img = img.read_image()
		if ARG.yrange_from is not None:
			img = img[ARG.yrange_from:ARG.yrange_to, :]
		if ARG.neglog:
			img = np.log(np.reciprocal(img))
		projectionData[:,i,:] = img 
		if ARG.verbose and i % 10 == 0:
			print("Read file %d of %d" % (i + 1, len(tifFiles)))
else:
	projectionData = DEN.getNumpyArray(ARG.inputProjections)
	projectionData = np.swapaxes(projectionData, 0, 1)
	angles_count = projectionData.shape[1]
	if ARG.yrange_from is not None:
		projectionData = projectionData[ARG.yrange_from:ARG.yrange_to, :]
	row_count=projectionData.shape[0]
	col_count=projectionData.shape[2]
	if ARG.neglog:
		projectionData = np.log(np.reciprocal(projectionData))
	if ARG.verbose:
		print("Projection data in the file %s has dimensions height=%d angles=%d width=%d dtype=%s min=%f, max=%f, mean=%f."%(ARG.inputProjections, projectionData.shape[0], projectionData.shape[1], projectionData.shape[2], projectionData.dtype, projectionData.min(), projectionData.max(), projectionData.mean()))
if ARG.store_projections is not None:
	DEN.storeNdarrayAsDEN(os.path.join(ARG.output_folder, ARG.store_projections), np.swapaxes(projectionData, 0, 1), force=ARG.force)
#Now I created structure with projections let's focus on angles
if ARG.angles_mat is not None:
	matlab_dic = scipy.io.loadmat(ARG.angles_mat)
	angles = matlab_dic["angles"]
else:
	theta_zero = degToRad(ARG.theta_zero)
	theta_angular_range = degToRad(ARG.theta_angular_range)
	angles = np.linspace(theta_zero, theta_zero + theta_angular_range, num=angles_count, endpoint=False)

#Geometry setup
if len(angles) != angles_count:
	print("INCOMPATIBLE ANGLES DIMENSIONS!")
	os.sys.exit(1)
vectors = generateAstraParallel3d_vec(angles, ARG.pixel_sizex,
                                                ARG.pixel_sizey, ARG.detector_center_offsetvx, ARG.detector_center_offsetvy, ARG.material_ct_convention)
proj_geom = astra.create_proj_geom('parallel3d_vec', row_count, col_count,
                                   vectors)
#logFile = getReconLog(tifFiles[0])
#print("Using logFile %s"%(logFile))
#dct = parseParamFile(logFile)
#det_width = float(dct["scan_pixelsize"])
#det_height = float(dct["scan_pixelsize"])

#The same as the following
#proj_geom = astra.create_proj_geom('parallel3d', det_width, det_height, row_count, col_count, angles)
#Projections read
#a matrix: the object is initialized with the contents of this matrix.
#The matrix must be of size (u,angles,v), where u is the number of columns of the detector and v the number
#of rows as defined in the projection geometry. It must be of class single, double or logical.
#2D: (angles,u), where u is the number of detector pixels as defined in the projection geometry
#projectionData = np.empty((col_count, angles_count, row_count))
#Contrary to what doc say it needs to be like this
#Coordinate order: row (v), angle, column (u)
#see https://github.com/astra-toolbox/astra-toolbox/blob/master/samples/python/s006_3d_data.py

#For 3D geometries there are no explicit projectors
#The consequence is 3D projectors use only
#raycasters that are configurable num

sin_id = astra.data3d.create('-proj3d', proj_geom, projectionData)

#distance between the centers of two adjacent detector pixels
if ARG.verbose:
	print(
	    "Creating projector with det_width=%f and det_height=%f for %d angles."
	    % (ARG.pixel_sizex, ARG.pixel_sizey, len(angles)))

#In 3D
#astra_create_proj_geom('parallel3d_vec',  det_row_count, det_col_count, vectors);
#First try something small although number of detectors is 3927
vx_count = ARG.volume_sizex
vy_count = ARG.volume_sizey
if ARG.volume_sizez is None:
	vz_count = row_count
else:
	vz_count = ARG.volume_sizez
if ARG.voxel_sizex is None:
	VOXELX = ARG.pixel_sizex
else:
	VOXELX = ARG.voxel_sizex
if ARG.voxel_sizey is None:
	VOXELY = ARG.pixel_sizex
else:
	VOXELY = ARG.voxel_sizey
if ARG.voxel_sizez is None:
	VOXELZ = ARG.pixel_sizey
else:
	VOXELZ=ARG.voxel_sizez
min_x = -0.5*vx_count*VOXELX
max_x = 0.5*vx_count*VOXELX
min_y = -0.5*vy_count*VOXELY
max_y = 0.5*vy_count*VOXELY
min_z = -0.5 * ARG.pixel_sizey * vz_count
max_z = 0.5 * ARG.pixel_sizey * vz_count

#=======================OUTPUT VOLUME=========
if not os.path.exists(ARG.output_folder):
	os.makedirs(ARG.output_folder, exist_ok=True)
outputName = "reconstructVolumeUsingProjectionsAstra_%s" % ARG.suffix
outputPattern = os.path.join(ARG.output_folder, outputName)
if ARG.saveden:
	outputFileName="%s.den" % (outputPattern)
	if os.path.exists(outputFileName) and not ARG.force:
		print("File %s exist, add --force to overwrite." % ARG.outputDEN)
		os.sys.exit(1)
vol_geom = astra.create_vol_geom(vy_count, vx_count, vz_count, min_x, max_x,
                                 min_y, max_y, min_z, max_z)
#To understand this order is useful to go https://www.astra-toolbox.com/apiref/creators.html
#See https://github.com/astra-toolbox/astra-toolbox/issues/206
#Basically volumes are defined by (y, x, z) while arrays used for initialization are defined as (z, y, x).

vol_id = astra.data3d.create('-vol', vol_geom, 0.0)
#Following functionality only in unmerged branch
#https://github.com/astra-toolbox/astra-toolbox/discussions/314
#https://people.compute.dtu.dk/pcha/HDtomo/SC/Week2Day4.pdf
#proj_geom, vol_geom = astra.mpi.create(proj_geom, vol_geom)
#In 3D
#vol_geom = astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z);
#vol_id = astra_mex_data3d('create', '-vol', vol_geom);

cfg = {}
if ARG.cgls:
	cfg = createProjectorConfig("CGLS3D", sin_id, vol_id, ARG.gpu)
elif ARG.sirt:
	cfg = createProjectorConfig("SIRT3D", sin_id, vol_id, ARG.gpu)
elif ARG.fbp:
	cfg = createProjectorConfig("BP3D", sin_id, vol_id, ARG.gpu)

print("Creating %s algorithm" % cfg["type"])
#cfg = {};
#cfg["type"] = 'CGLS'
#cfg["type"] = 'CGLS_GPU'
#cfg["ProjectorId"] = proj_id;
#cfg["ProjectionDataId"] = sin_id;
#cfg["ReconstructionDataId"] = vol_id;
cgls_id = astra.algorithm.create(cfg)
print("Created algorithm")
if ARG.fbp:
	astra.algorithm.run(cgls_id, 1)
else:
	astra.algorithm.run(cgls_id, ARG.itterations)
print("Getting volume")
volume = astra.data3d.get(vol_id)
print("Output volume has dimensions %dx%dx%d and type %s" %
      (volume.shape[0], volume.shape[1], volume.shape[2], volume.dtype))


fullOutputName = os.path.join(ARG.output_folder, outputName)

if ARG.saveden:
	writeDenFile(volume, "%s.den" % (outputPattern), ARG.force)
if ARG.savetiff:
	writeTiffFiles(volume, outputPattern, ARG.force)
sec = time.time() - sec
print("Time %0.2fs"%(sec))

with open("%s.log" % (fullOutputName), 'wt') as f:
	json.dump(vars(ARG), f, indent=4)
#garbage disposal
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)
astra.algorithm.delete(cgls_id)
