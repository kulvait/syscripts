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

#folderPrefix = "/home/user/exp/DESY_global"
#Can be mountpoint on home computer
inputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03"
outputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_folder"
inputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03"

parser = argparse.ArgumentParser()
parser.add_argument("inputFolder")
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
parser.add_argument("--itterations", default=100)
parser.add_argument('--box-sizex', help="Bounding box dimension.", type=float, default=5.0)
parser.add_argument('--box-sizey', help="Bounding box dimension.", type=float, default=5.0)
parser.add_argument('--box-sizez', help="Bounding box dimension.", type=float, default=5.0)
parser.add_argument('--volume-sizex', help="Volume dimension x.", type=int, default=1024)
parser.add_argument('--volume-sizey', help="Volume dimension y.", type=int, default=1024)
parser.add_argument('--volume-sizez', help="Volume dimension z.", type=int, default=1024)
parser.add_argument('--first-index', type=int, default=None)
parser.add_argument('--last-index', type=int, default=None)
parser.add_argument("--angles-mat", type=str, default=None, help="Sequence of angles stored in mat file.")


ARG = parser.parse_args([inputFolder, "--output-folder", outputFolder, "--force", "--gpu", "--sirt", "--verbose", "--saveden", "--first-index", "450", "--last-index", "460"])
#Previous do not work and the following works
#ARG = parser.parse_args([inputFolder, "--output-folder", outputFolder, "--force", "--gpu", "--fbp", "--verbose", "--saveden", "--first-index", "450", "--last-index", "460"])


#sin=sin[1::2]
#angles=angles[1::2]

def getFileNum(filePath):
    numberSearch = re.search(r"(\d*).tif", filePath)
    if numberSearch is None:
        return ""
    else:
        suf  = numberSearch.group(1)
        if suf == "":
            return ""
        else:
            return "%05d"%int(suf)

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
                continue;
            dct[tokens[0].strip()] = tokens[1].strip()
        return dct
    return None

def writeDenFile(volume, denFile, force=False):
    if os.path.exists(denFile):
        if force:
            os.remove(denFile)
        else:
            raise IOError("File %s exists, add force to overwrite"%(denFile))
    dimy=volume.shape[1]
    dimx=volume.shape[2]
    dimz=volume.shape[0]
    DEN.writeEmptyDEN(denFile, dimx, dimy, dimz, True)
    for k in range(dimz):
        DEN.writeFrame(denFile, k, volume[k,:,:], True)
    
def createProjectorConfig(projectorName, projectionsID, volumeID, usegpu = True):
    #cfg = {}
    if usegpu:
        name = "%s_CUDA"%projectorName
        #cfg["type"]="%s_CUDA"%projectorName
    else:
        name=projectorName
        #cfg["type"]=projectorName
    cfg = astra.astra_dict(name)
    cfg["ProjectionDataId"] = projectionsID;
    cfg["ReconstructionDataId"] = volumeID;
    return cfg

def generateEquispacedAstraParallel3d_vec(angles_count, det_width, det_height):
    increment = 2*np.pi/angles_count;
    vectors = np.zeros((angles_count, 12))
    for i in range(angles_count):
        angle = i*increment
        #From https://www.astra-toolbox.com/docs/geom3d.html
        #ray direction
        vectors[i,0] = np.sin(angle);
        vectors[i,1] = -np.cos(angle);
        vectors[i,2] = 0;
        #center of detector
        vectors[i,3] = 0;
        vectors[i,4] = 0;
        vectors[i,5] = 0;
        # vector from detector pixel (0,0) to (0,1)
        vectors[i,6] = np.cos(angle) * det_width;
        vectors[i,7] = np.sin(angle) * det_width;
        vectors[i,8] = 0;
        #vector from detector pixel (0,0) to (1,0)
        vectors[i,9] = 0;
        vectors[i,10] = 0;
        vectors[i,11] = det_height;
    return(vectors)
    
def transformToExtinction(invertedProjectionIntensities):
   #DEBUG ... another scaling
   #invertedProjectionIntensities = invertedProjectionIntensities/invertedProjectionIntensities.max()
   #END DEBUG
   invertedProjectionIntensities = np.log(np.reciprocal(invertedProjectionIntensities)) #Compute extinction
   #offset = 1-1/invertedProjectionIntensities.max()
   #invertedProjectionIntensities = np.log(np.reciprocal(invertedProjectionIntensities)+offset)
   return(invertedProjectionIntensities)
   


#print("Log file is %s"%(logFile))
#print(dct)
#=================INPUT PROJECTION DATA===============
tifFiles = glob.glob(os.path.join(ARG.inputFolder, "*.tif"))
tifFiles.sort()
if ARG.first_index is not None:
	tifFiles = tifFiles[ARG.first_index:ARG.last_index]
tif=TIFF.open(tifFiles[0])
sin=tif.read_image()
angles = np.linspace(0, 2*np.pi, sin.shape[0], endpoint=False)#Equally spaced values which has sin.shape[0] voids
angles_count = sin.shape[0]
col_count = sin.shape[1]
row_count = len(tifFiles)
if ARG.verbose:
    print("The file %s has dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."%(tifFiles[0], sin.shape[0], sin.shape[1], sin.dtype, sin.min(), sin.max(), sin.mean()))

#Geometry setup
logFile = getReconLog(tifFiles[0])
print("Using logFile %s"%(logFile))
dct = parseParamFile(logFile)
det_width = float(dct["scan_pixelsize"])
det_height = float(dct["scan_pixelsize"])
vectors = generateEquispacedAstraParallel3d_vec(angles_count, det_width, det_height)
proj_geom = astra.create_proj_geom('parallel3d_vec', row_count, col_count, vectors)
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
projectionData = np.zeros(shape=(row_count, angles_count, col_count), dtype="float32")
for i in range(row_count):
    tif=TIFF.open(tifFiles[i])
    sin=tif.read_image()
    sin=transformToExtinction(sin)
    #projectionData[:,:,i] = np.transpose(sin)
    projectionData[i,:,:] = sin
    if ARG.verbose and i % 10 == 0:
        print("Just read %d-th/%d projectionData of %s"%(i+1, row_count, tifFiles[i]))
if ARG.neglog:
	projectionData = np.log(np.reciprocal(projectionData))

if ARG.verbose:
    print("Transformed projections of dimensions %dx%dx%d and dtype=%s with min=%f, max=%f, mean=%f."%(projectionData.shape[0], projectionData.shape[1], projectionData.shape[2], projectionData.dtype, projectionData.min(), projectionData.max(), projectionData.mean()))


#For 3D geometries there are no explicit projectors
#The consequence is 3D projectors use only 
#raycasters that are configurable num

sin_id = astra.data3d.create('-proj3d', proj_geom, projectionData);


angles = np.linspace(0, 2*np.pi, sin.shape[0], endpoint=False)#Equally spaced values which has sin.shape[0] voids
#print(angles)
#distance between the centers of two adjacent detector pixels
if ARG.verbose:
    print("Creating projector with det_width=%f and det_height=%f for %d angles."%(det_width, det_height, len(angles)))




#In 3D
#astra_create_proj_geom('parallel3d_vec',  det_row_count, det_col_count, vectors);
#First try something small although number of detectors is 3927
vx_count = 1024
vy_count = 1024
vz_count = len(tifFiles)
min_x = -2.3
max_x = 2.3
min_y = -2.3
max_y = 2.3
min_z = -det_height*vz_count*0.5
max_z = det_height*vz_count*0.5

#=======================OUTPUT VOLUME=========
os.makedirs(ARG.output_folder, exist_ok=True)
outputName = "%s%s"%(ARG.suffix, getFileNum(tifFiles[0]))

vol_geom = astra.create_vol_geom(vy_count, vx_count, vz_count, min_x, max_x, min_y, max_y, min_z, max_z)
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

cfg={}
if ARG.cgls:
    cfg = createProjectorConfig("CGLS3D", sin_id, vol_id, ARG.gpu)
elif ARG.sirt:
    cfg = createProjectorConfig("SIRT3D", sin_id, vol_id, ARG.gpu)
elif ARG.fbp:
    cfg = createProjectorConfig("BP3D", sin_id, vol_id, ARG.gpu)
    
print("Creating %s algorithm"%cfg["type"])
#cfg = {};
#cfg["type"] = 'CGLS'
#cfg["type"] = 'CGLS_GPU'
#cfg["ProjectorId"] = proj_id;
#cfg["ProjectionDataId"] = sin_id;
#cfg["ReconstructionDataId"] = vol_id;
cgls_id = astra.algorithm.create(cfg);
print("Created algorithm")
if ARG.fbp:
    astra.algorithm.run(cgls_id, 1)
else:
    astra.algorithm.run(cgls_id, ARG.itterations);
print("Getting volume")
volume = astra.data3d.get(vol_id);
print("Output volume has dimensions %dx%dx%d and type %s"%(volume.shape[0], volume.shape[1], volume.shape[2], volume.dtype))

fullOutputName = os.path.join(ARG.output_folder, outputName)

if ARG.saveden:
    writeDenFile(volume, "%s.den"%(fullOutputName), ARG.force)
if ARG.savetiff:
    im = Image.fromarray(volume, mode='F') # float32
    im.save("%s.tiff"%(fullOutputName), "TIFF")
    
with open("%s.log"%(fullOutputName), 'wt') as f:
        json.dump(vars(ARG), f, indent=4)    
#garbage disposal
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)
astra.algorithm.delete(cgls_id);
