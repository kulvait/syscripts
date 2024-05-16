#!/usr/bin/env python3
"""
Created on Fri Feb 11 11:05:28 2022

@author: Vojtech Kulvait
"""
import pdb
from termcolor import colored
import odl
import numpy as np
#import mpi#Volume decomposition

import os
import sys
import argparse
from denpy import DEN
from PIL import Image
from PIL.TiffTags import TAGS
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
outputFolder = "/home/kulvaitv/exp/PtNiWire/scratch_cc/kulvait_scratch/ivw0032_Referenz_blau_4_000/odl"

parser = argparse.ArgumentParser()
parser.add_argument("inputProjections")
parser.add_argument("--cgls", action="store_true")
parser.add_argument("--sirt", action="store_true")
parser.add_argument("--sart", action="store_true")
parser.add_argument(
    "--btv",
    action="store_true",
    help=
    "Bregman-TV according to https://github.com/odlgroup/odl/blob/master/examples/solvers/bregman_tv_tomography.py"
)
parser.add_argument(
    "--pdhg",
    action="store_true",
    help=
    "Total variation tomography using PDHG according to https://github.com/odlgroup/odl/blob/master/examples/solvers/pdhg_tomography.py"
)
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
parser.add_argument("--angles-den", default=None)
parser.add_argument(
    "--theta-zero",
    type=float,
    default=0,
    help=
    "Initial angle theta from Radon transform in degrees, defaults to zero. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html"
)
parser.add_argument(
    "--theta-angular-range",
    type=float,
    default=360,
    help=
    "This is angular range in degrees, along which possitions are distributed.")
parser.add_argument("--theta-endpoint",
                    action="store_true",
                    help="If to include endpoint in the range.")
parser.add_argument(
    "--material-ct-convention",
    action="store_true",
    default=False,
    help=
    "The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html."
)
parser.add_argument("--offset-mat", default=None)
parser.add_argument("--offset-x", default=0.0, type=float)
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
                    help="Voxel size x, defaults to pixel-sizex.",
                    type=float,
                    default=None)
parser.add_argument('--voxel-sizey',
                    help="Voxel size y, defaults to pixel-sizex.",
                    type=float,
                    default=None)
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
parser.add_argument(
    '--volume-sizez',
    help=
    "Volume dimension z, defaults to number of rows in input projection data",
    type=int,
    default=None)
parser.add_argument('--pixel-sizex',
                    help="Detector dimension x.",
                    type=float,
                    default=0.00255076)
parser.add_argument('--pixel-sizey',
                    help="Volume dimension y.",
                    type=float,
                    default=0.00255076)
parser.add_argument(
    '--regularization-parameter',
    help="Regularization parameter for given method if it supports it",
    type=float,
    default=0.0)
parser.add_argument('--yrange-from', type=int, default=None)
parser.add_argument('--yrange-to', type=int, default=None)
parser.add_argument(
    "--detector-center-offsetvx",
    type=float,
    default=0.,
    help=
    "Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0."
)
parser.add_argument(
    "--detector-center-offsetvy",
    type=float,
    default=0.,
    help=
    "Offset of the center of the detector, detector_center_offsetvx * VX + detector_center_offsetvy * VY is added to the coordinates of the center of the detector for each angle, defaults to 0.0."
)

#ARG = parser.parse_args([
#    inputFolder, "--output-folder", outputFolder, "--force", "--gpu", "--cgls",
#    "--verbose", "--yrange-from", "500", "--yrange-to", "530", "--itterations", "40", "--saveden"
#])
#Previous do not work and the following works
recoFolder = "/home/kulvaitv/exp/PtNiWire/processed/ivw0032_Referenz_blau_4_000/reco"
ARG = parser.parse_args([
    inputFolder, "--output-folder", outputFolder, "--force", "--gpu", "--fbp",
    "--verbose", "--saveden", "--yrange-from", "450", "--yrange-to", "460",
    "--store-projections", "prj.den", "--angles-mat",
    os.path.join(recoFolder, "angles.mat"), "--offset-mat",
    os.path.join(recoFolder, "offset_shift.mat"), "--volume-sizex", "2048",
    "--volume-sizey", "2048", "--neglog"
])

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


def degToRad(angle):
	return np.pi * angle / 180


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


def writeTiffFiles(volume, tiffFilePattern, force=False):
	dimx = volume.shape[0]
	dimy = volume.shape[1]
	dimz = volume.shape[2]
	for k in range(dimz):
		if ARG.yrange_from is not None:
			tiffFile = "%s_%05d.tiff" % (tiffFilePattern, k + ARG.yrange_from)
		else:
			tiffFile = "%s_%05d.tiff" % (tiffFilePattern, k)
		if os.path.exists(tiffFile):
			if force:
				os.remove(tiffFile)
			else:
				raise IOError("File %s exists, add force to overwrite" %
				              (tiffFile))
		im = Image.fromarray(np.flip(np.transpose(volume[:, :, k])),
		                     mode='F')  # float32
		im.save(tiffFile, "TIFF")


def writeDenFile(volume, denFile, force=False):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	dimx = volume.shape[0]
	dimy = volume.shape[1]
	dimz = volume.shape[2]
	DEN.writeEmptyDEN(denFile, [dimx, dimy, dimz], force=force)
	for k in range(dimz):
		#DEN.writeFrame(denFile, k, np.flip(np.transpose(volume[:, :, k]), 0), force=force)
		DEN.writeFrame(denFile, k, np.transpose(volume[:, :, k]), force=True)


def writeDenProjections(com, denFile, force=False):
	if os.path.exists(denFile):
		if force:
			os.remove(denFile)
		else:
			raise IOError("File %s exists, add force to overwrite" % (denFile))
	dimy = com.shape[2]
	dimx = com.shape[1]
	dimz = com.shape[0]
	DEN.writeEmptyDEN(denFile, [dimx, dimy, dimz], force=force)
	for k in range(dimz):
		DEN.writeFrame(denFile, k, np.transpose(com[k, :, :]), force=force)


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


def generateAstraParallel3d_vec(angles,
                                det_width,
                                det_height,
                                offsetvx=None,
                                offsetvy=None):
	if offsetvx is None:
		offsetvx = 0.
	if offsetvy is None:
		offsetvy = 0.
	vectors = np.zeros((len(angles), 12))
	for i in range(angles_count):
		angle = angles[i]
		#From https://www.astra-toolbox.com/docs/geom3d.html
		#ray direction
		vectors[i, 0] = np.cos(angle)
		vectors[i, 1] = np.sin(angle)
		vectors[i, 2] = 0.
		#center of detector
		vectors[i, 3] = 0. + np.sin(angle) * offsetvx
		vectors[i, 4] = 0. - np.cos(angle) * offsetvx
		vectors[i, 5] = offsetvy
		# vector from detector pixel (0,0) to (0,1)
		vectors[i, 6] = np.sin(angle) * det_width
		vectors[i, 7] = -np.cos(angle) * det_width
		vectors[i, 8] = 0
		#vector from detector pixel (0,0) to (1,0)
		vectors[i, 9] = 0
		vectors[i, 10] = 0
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
#=================INPUT DIMENSIONS===============
sec = time.time()
if os.path.isdir(ARG.inputProjections):
	pth = os.path.join(ARG.inputProjections, "*.tif")
	tifFiles = glob.glob(pth)
	tifFiles.sort()
	if len(tifFiles) == 0:
		raise (IOError("The path %s contains %d *.tif projections." %
		               (pth, len(tifFiles))))
	print("The path %s contains %d *.tif projections." % (pth, len(tifFiles)))
	tif = Image.open(tifFiles[0])
	img = np.array(tif)
	row_count = img.shape[0]
	col_count = img.shape[1]
	angles_count = len(tifFiles)
	if ARG.verbose:
		print(
		    "First projection %s has dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."
		    % (tifFiles[0], img.shape[0], img.shape[1], img.dtype, img.min(),
		       img.max(), img.mean()))
	if ARG.yrange_from is not None:
		row_count = ARG.yrange_to - ARG.yrange_from
else:
	projectionDataInfo = DEN.readHeader(ARG.inputProjections)
	if projectionDataInfo["dimcount"] != 3:
		raise (IOError("Wrong dimcount %d in input projection file %s." %
		               (projectionDataInfo["dimcount"], ARG.inputProjections)))
	row_count = projectionDataInfo["dimspec"][1]
	col_count = projectionDataInfo["dimspec"][0]
	angles_count = projectionDataInfo["dimspec"][2]
	if ARG.yrange_from is not None:
		row_count = ARG.yrange_to - ARG.yrange_from
print(colored("Inputs are read", "green"), flush=True)

#First I need to specify vectors of discretization of detector and angles
#angles
#Angles are omega from https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html but read are thetas
if ARG.angles_mat is not None:
	matlab_dic = scipy.io.loadmat(ARG.angles_mat)
	angles = matlab_dic["angles"]
	angles = angles.reshape(
	    (angles.shape[0])
	)  #It is array of the shape (angles.shape[0], 1) and ODL does not like it
	if len(angles) != angles_count:
		print(
		    "INCOMPATIBLE ANGLES DIMENSIONS len(angles)=%d and angles_count=%d!"
		    % (len(angles), angles_count))
		os.sys.exit(1)
elif ARG.angles_den is not None:
	info = DEN.getFrame(ARG.angles_den, 0)
	angles = info[1]
	angles = list(degToRad(x) for x in angles)
	if len(angles) != angles_count:
		print("angles_count=%d is not equal len(angles)=%d!" %
		      (angles_count, len(angles)))
		os.sys.exit(1)
else:
	angles = np.linspace(
	    0, 2 * np.pi, angles_count,
	    endpoint=False)  #Equally spaced values which has sin.shape[0] voids
	angles = np.linspace(
	    0 - np.pi / 2, np.pi - np.pi / 2, angles_count,
	    endpoint=False)  #Equally spaced values which has sin.shape[0] voids
	theta_zero = degToRad(ARG.theta_zero)
	theta_angular_range = degToRad(ARG.theta_angular_range)
	angles = np.linspace(theta_zero,
	                     theta_zero + theta_angular_range,
	                     num=angles_count,
	                     endpoint=ARG.theta_endpoint)
#Detector dimensions from projection data
min_px = -0.5 * ARG.pixel_sizex * col_count
max_px = 0.5 * ARG.pixel_sizex * col_count
min_py = -0.5 * ARG.pixel_sizey * row_count
max_py = 0.5 * ARG.pixel_sizey * row_count
detector_centers_x = np.linspace(min_px + 0.5 * ARG.pixel_sizex,
                                 max_px - 0.5 * ARG.pixel_sizex, col_count)
detector_centers_y = np.linspace(min_py + 0.5 * ARG.pixel_sizey,
                                 max_py - 0.5 * ARG.pixel_sizey, row_count)
print("Pixel sizes = [%f, %f]" % (ARG.pixel_sizex, ARG.pixel_sizey))

# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
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
	VOXELZ = ARG.voxel_sizez
min_x = -0.5 * vx_count * VOXELX
max_x = 0.5 * vx_count * VOXELX
min_y = -0.5 * vy_count * VOXELY
max_y = 0.5 * vy_count * VOXELY
min_z = -0.5 * vz_count * VOXELZ
max_z = 0.5 * vz_count * VOXELZ
reco_space = odl.uniform_discr(min_pt=[min_x, min_y, min_z],
                               max_pt=[max_x, max_y, max_z],
                               shape=[vx_count, vy_count, vz_count],
                               dtype='float32')

#=======================OUTPUT VOLUME=========
if not os.path.exists(ARG.output_folder):
	os.makedirs(ARG.output_folder, exist_ok=True)
if ARG.yrange_from is not None:
	rangeString = "Yfrom%dto%d"%(ARG.yrange_from, ARG.yrange_from + row_count)
else:
	rangeString = "fullYdim%d"%(row_count)
outputName = "reconstructVolumeUsingProjectionsODL_%s" % (rangeString)
if ARG.suffix != "":
	outputName = "%s_%s" % (ARG.suffix)
fullOutputName = os.path.join(ARG.output_folder, outputName)
if ARG.saveden:
	outputFileName = "%s.den" % (fullOutputName)
if ARG.savetiff:
	outputFileName = "%s.tiff" % (fullOutputName)
if os.path.exists(outputFileName) and not ARG.force:
	print("File %s exist, add --force to overwrite." % fullOutputName)
	os.sys.exit(1)

#For now do no offsetting when defining geometry
angle_partition = odl.nonuniform_partition(angles)
detector_partition = odl.nonuniform_partition(detector_centers_x,
                                              detector_centers_y)
offset = [0, 0, 0]
if ARG.offset_mat is not None:
	print("Loading offset from %s" % (ARG.offset_mat))
	matlab_dic = scipy.io.loadmat(ARG.offset_mat)
	print(matlab_dic["offset_shift"][0][0])
	offset[0] = matlab_dic["offset_shift"][0][0] * ARG.pixel_sizex
else:
	offset[0] = ARG.offset_x * ARG.pixel_sizex
	#offset[0] = -ARG.offset_x
print("offset: %s" % (offset))
#If initial ray direction is along x axis detector is aligned with y axis
#VK geometry
initAngle = angles[0]
#For theta
if ARG.material_ct_convention:
	det_axis_y = (0.0, 0.0, 1.0)
else:
	det_axis_y = (0.0, 0.0, -1.0)
geometry = odl.tomo.Parallel3dAxisGeometry(
    angle_partition,
    detector_partition,
    axis=[0, 0, 1],
    det_axes_init=[(np.cos(initAngle), np.sin(initAngle), 0), det_axis_y],
    det_pos_init=[
        offset[0] * np.cos(initAngle), offset[0] * np.sin(initAngle), offset[2]
    ])
#For omega
#geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition, axis=[0 , 0, 1], det_axes_init=[(-np.sin(initAngle), np.cos(initAngle) , 0), (0, 0, -1)], det_pos_init=[offset[0]*-np.sin(initAngle),offset[0]*np.cos(initAngle),offset[2]])
#Default ODL geometry
#geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition, axis=[0 , 0, 1], det_axes_init=[(1, 0, 0), (0, 0, 1)], det_pos_init=[offset[0],offset[1],offset[2]])

ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

#Create projection data
partition = odl.nonuniform_partition(angles, detector_centers_x,
                                     detector_centers_y)
tspace = odl.space.space_utils.rn(partition.shape, dtype='float32')
ds = odl.discr.discr_space.DiscretizedSpace(partition, tspace)
projectionData_element = ds.element()
with odl.util.utility.writable_array(projectionData_element) as projectionData:
	if os.path.isdir(ARG.inputProjections):
		pth = os.path.join(ARG.inputProjections, "*.tif")
		tifFiles = glob.glob(pth)
		tifFiles.sort()
		for i in range(angles_count):
			f = tifFiles[i]
			im = Image.open(f)
			img = np.array(im)
			if ARG.yrange_from is not None:
				img = img[ARG.yrange_from:ARG.yrange_to, :]
				#img = img[:,ARG.yrange_from:ARG.yrange_to]
			if ARG.neglog:
				projectionData[i] = np.log(np.reciprocal(np.transpose(img)))
				#projectionData[i] = np.log(np.reciprocal(img))
			else:
				projectionData[i] = np.transpose(img)
			if ARG.verbose and i % 10 == 0:
				print("Read file %d of %d" % (i + 1, len(tifFiles)))
	else:
		for i in range(angles_count):
			img = DEN.getFrame(ARG.inputProjections, i)
			if ARG.yrange_from is not None:
				img = img[ARG.yrange_from:ARG.yrange_to, :]
			if ARG.neglog:
				projectionData[i] = np.log(np.reciprocal(np.transpose(img)))
			else:
				projectionData[i] = np.transpose(img)
	if ARG.verbose:
		print(
		    "Projection data in %s has dimensions angles=%d width=%d height=%d dtype=%s min=%f, max=%f, mean=%f."
		    % (ARG.inputProjections, projectionData.shape[0],
		       projectionData.shape[1], projectionData.shape[2],
		       projectionData.dtype, projectionData.min(), projectionData.max(),
		       projectionData.mean()))
	if ARG.store_projections is not None:
		denFile = os.path.join(ARG.output_folder, ARG.store_projections)
		writeDenProjections(projectionData, denFile, force=ARG.force)
		#DEN.storeNdarrayAsDEN(denFile, projectionData, force=ARG.force)
	if ARG.verbose:
		print(
		    "The projection data has the dimensions %dx%dx%d and dtype=%s with min=%f, max=%f, mean=%f."
		    %
		    (projectionData.shape[0], projectionData.shape[1],
		     projectionData.shape[2], projectionData.dtype,
		     projectionData.min(), projectionData.max(), projectionData.mean()))

#distance between the centers of two adjacent detector pixels
if ARG.verbose:
	print(
	    "Creating projector with det_width=%f and det_height=%f for %d angles."
	    % (ARG.pixel_sizex, ARG.pixel_sizey, len(angles)))

if ARG.cgls:
	print("Not yet implemented")
elif ARG.sirt:
	print("Not yet implemented")
elif ARG.fbp:
	fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)
	print(colored("Start FBP %s" % (fbp.operatorString()), "green"), flush=True)
	x = reco_space.zero()
	#from pudb.remote import set_trace;
	#set_trace(term_size=(201,52))#This is what is in $LINES and $COLUMNS see https://unix.stackexchange.com/questions/184009/how-do-i-find-number-of-vertical-lines-available-in-the-terminal
	#import pudb; pu.db
	#pdb.run("fbp(projectionData_element)")
	rec = fbp(projectionData_element, out=x)
	print(colored("End FBP", "green"), flush=True)
elif ARG.btv:
	print("Start Bregman-TV")
	# Components for variational problem: l2-squared data matching and isotropic
	# TV-regularization
	l2_norm = 0.5 * odl.solvers.L2NormSquared(
	    ray_trafo.range).translated(projectionData_element)
	gradient = odl.Gradient(reco_space)
	reg_param = 0.3  #Overregularization
	reg_param = 0.03
	reg_param = 0.003
	reg_param = 0.0003
	reg_param = 0.00003
	reg_param = 0.000003
	reg_param = 0.0000003
	reg_param = 0.00000003
	reg_param = 0.000000003
	l12_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)
	# Assemble functionals and operators for the optimization algorithm
	f = odl.solvers.ZeroFunctional(
	    reco_space)  # No f functional used, set to zero
	g = [l2_norm, l12_norm]
	L = [ray_trafo, gradient]
	# Estimated operator norms, which are used to ensure that we fulfill the
	# convergence criteria of the optimization algorithm
	ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=20)
	gradient_norm = odl.power_method_opnorm(gradient, maxiter=20)
	# Parameters for the optimization method; tuned in order to reduce the number
	# of inner iterations needed to solve the first step in the Bregman iterations
	niter_inner = 200
	tau = 0.01  # Step size for the primal variable
	sigma_ray_trafo = 45.0 / ray_trafo_norm**2  # Step size for dual variable
	sigma_gradient = 45.0 / gradient_norm**2  # Step size for dual variable
	sigma = [sigma_ray_trafo, sigma_gradient]
	# The reconstruction looks nice after about 5 outer iterations; set total
	# number of outer iterations to 7 to show what happens if one does to many
	niter_bregman = 7
	niter_bregman = 5  #Or not since that takes too long
	# Create initial guess and initial subgradient
	x = reco_space.zero()
	p = reco_space.zero()
	# This defines the outer Bregman iterations
	for breg_iter in range(niter_bregman):
		print('Outer Bregman Iteration: {}'.format(breg_iter))
		# Create the affine part of the Bregman functional
		constant = l12_norm(gradient(x))
		print("1")
		linear_part = reg_param * odl.solvers.QuadraticForm(vector=-p,
		                                                    constant=constant)
		print("2")
		callback_inner = odl.solvers.CallbackPrintIteration(step=50)
		print("3")
		# Inner iterations; x is updated in-place with the consecutive iterates
		odl.solvers.forward_backward_pd(x=x,
		                                f=f,
		                                g=g,
		                                L=L,
		                                h=linear_part,
		                                tau=tau,
		                                sigma=sigma,
		                                niter=niter_inner,
		                                callback=callback_inner)
		print("4")
		# Update the subgradient
		p -= (1 / reg_param) * ray_trafo.adjoint(l2_norm.gradient(ray_trafo(x)))
		print("5")
		# Display the result after this iteration
		#x.show(title='Outer Bregman Iteration {}'.format(breg_iter),force_show=True)
	rec = x
elif ARG.pdhg:
	gradient = odl.Gradient(reco_space)
	op = odl.BroadcastOperator(ray_trafo, gradient)
	f = odl.solvers.ZeroFunctional(op.domain)
	l2_norm = odl.solvers.L2NormSquared(
	    ray_trafo.range).translated(projectionData_element)
	if ARG.regularization_parameter == 0:
		reg = 0.015
	else:
		reg = ARG.regularization_parameter
	print("Start PDHG with REG=%e" % (reg))
	l1_norm = reg * odl.solvers.L1Norm(gradient.range)
	g = odl.solvers.SeparableSum(l2_norm, l1_norm)
	op_norm = 1.1 * odl.power_method_opnorm(op)
	#Config
	#	print("CALLING POWER METHOD", flush=True)
	#	op_norm = 1.01 * odl.power_method_opnorm(op)
	#	print("END POWER METHOD", flush=True)
	niter = 200
	tau = 1.0 / op_norm
	sigma = 1.0 / op_norm
	print("Tau=%e sigma=%e" % (tau, sigma), flush=True)
	#	print(tau)

	#callback = (odl.solvers.CallbackPrintIteration(step=10) &
	#        odl.solvers.CallbackShow(step=10))
	x = op.domain.zero()
	#odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)
	odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
	#odl.solvers.pdhg(x, f, g, op, niter=1)
	rec = x

print("End reconstruction, get and write volume data.", flush=True)
with odl.util.utility.writable_array(rec) as volume:
	fullOutputName = os.path.join(ARG.output_folder, outputName)
	print("Output volume has dimensions %dx%dx%d and type %s" %
	      (volume.shape[0], volume.shape[1], volume.shape[2], volume.dtype))
	if ARG.saveden:
		writeDenFile(volume, "%s.den" % (fullOutputName), ARG.force)
	if ARG.savetiff:
		writeTiffFiles(volume, fullOutputName, ARG.force)
sec = time.time() - sec
print("Time %0.2fs\n" % (sec))

with open(os.path.join(ARG.output_folder, "ARG_%s.log" % (outputName)),
          'wt') as f:
	json.dump(vars(ARG), f, indent=4)
