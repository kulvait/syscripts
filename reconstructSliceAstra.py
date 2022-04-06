#!/usr/bin/env python3
"""
Created on Fri Feb 11 11:05:28 2022

@author: Vojtech Kulvait
"""

import astra
import os
import argparse
from libtiff import TIFF
import numpy as np
from denpy import DEN
from PIL import Image
import re
import json

sinogram = "/asap3/petra3/gpfs/p07/2020/data/11009431/processed/bric022_369_a/trans03/bric022_369_a00374.tif"
outputFolder = "/asap3/petra3/gpfs/p07/2020/data/11009431/scratch_cc/VKREC/bric022_369_a_FBP"

parser = argparse.ArgumentParser()
parser.add_argument("inputSinogramTiff")
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
parser.add_argument("--suffix", default="")
parser.add_argument("--output-folder", default=".")
parser.add_argument("--itterations", default=100)
parser.add_argument('--box-sizex', help="Bounding box dimension.", type=float, default=5.0)
parser.add_argument('--box-sizey', help="Bounding box dimension.", type=float, default=5.0)
parser.add_argument('--volume-sizex', help="Volume dimension x.", type=int, default=1024)
parser.add_argument('--volume-sizey', help="Volume dimension y.", type=int, default=1024)

#ARG = parser.parse_args([sinogram, "--force", "--gpu", "--cgls", "--verbose", "--savetiff"])
ARG = parser.parse_args()

if ARG.verbose:
    print("Creating folder %s"%(ARG.output_folder))
os.makedirs(ARG.output_folder, exist_ok=True)


tif=TIFF.open(ARG.inputSinogramTiff)
sin=tif.read_image()
angles = np.linspace(0, 2*np.pi, sin.shape[0], endpoint=False)#Equally spaced values which has sin.shape[0] voids

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



if ARG.verbose:
    print("Object has dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."%(sin.shape[0], sin.shape[1], sin.dtype, sin.min(), sin.max(), sin.mean()))

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
    dimy=volume.shape[0]
    dimx=volume.shape[1]
    dimz=1
    DEN.writeEmptyDEN(denFile, dimx, dimy, dimz, True)
    DEN.writeFrame(denFile, 0, volume, True)
    
def createProjectorConfig(projectorName, projectorID, projectionsID, volumeID, usegpu = True):
    cfg = {}
    if usegpu:
        cfg["type"]="%s_CUDA"%projectorName
    else:
        cfg["type"]=projectorName
    cfg["ProjectorId"] = projectorID;
    cfg["ProjectionDataId"] = projectionsID;
    cfg["ReconstructionDataId"] = volumeID;
    return cfg

outputName = "SLURM%s%s"%(ARG.suffix, getFileNum(ARG.inputSinogramTiff))
logFile = getReconLog(ARG.inputSinogramTiff)
dct = parseParamFile(logFile)
#print("Log file is %s"%(logFile))
#print(dct)
det_width = float(dct["scan_pixelsize"])
det_count = sin.shape[1]

#DEBUG ... another scaling
#sin = sin/sin.max()
#END DEBUG

sin = np.log(np.reciprocal(sin)) #Compute extinction
#offset = 1-1/sin.max()
#sin = np.log(np.reciprocal(sin)+offset)

if ARG.verbose:
    print("Transformed sin of dimensions %dx%d and dtype=%s with min=%f, max=%f, mean=%f."%(sin.shape[0], sin.shape[1], sin.dtype, sin.min(), sin.max(), sin.mean()))


angles = np.linspace(0, 2*np.pi, sin.shape[0], endpoint=False)#Equally spaced values which has sin.shape[0] voids
#print(angles)
#distance between the centers of two adjacent detector pixels
print("Creating projector with det_width=%f and det_count=%f for %d angles."%(det_width, det_count, len(angles)))



proj_geom = astra.create_proj_geom('parallel', det_width, det_count, angles)
#In 3D
#astra_create_proj_geom('parallel3d_vec',  det_row_count, det_col_count, vectors);
#First try something small although number of detectors is 3927

col_count = ARG.volume_sizex
row_count = ARG.volume_sizey
min_x = -0.5*ARG.box_sizex
max_x = 0.5*ARG.box_sizex
min_y = -0.5*ARG.box_sizey
max_y = 0.5*ARG.box_sizey

vol_geom = astra.create_vol_geom(row_count, col_count, min_x, max_x, min_y, max_y)
vol_id = astra.data2d.create('-vol', vol_geom, 0.0)
#In 3D
#vol_geom = astra_create_vol_geom(row_count, col_count, slice_count, min_x, max_x, min_y, max_y, min_z, max_z);
#vol_id = astra_mex_data3d('create', '-vol', vol_geom);

sin_id = astra.data2d.create('-sino', proj_geom, sin);

#https://www.astra-toolbox.com/docs/proj2d.html
#Types are line|strip|linear
projType2D = "strip"
#projType2D = "line"
#projType2D = "cuda"
proj_id = astra.create_projector(projType2D, proj_geom, vol_geom);
prj_iscuda= astra.projector.is_cuda(proj_id)
print("Created projector %s with is_cuda=%s."%(projType2D, prj_iscuda))
usegpu=True
cfg={}
if ARG.cgls:
    cfg = createProjectorConfig("CGLS", proj_id, sin_id, vol_id, ARG.gpu)
    outputName = "CGLS_%s"%outputName
elif ARG.sirt:
    cfg = createProjectorConfig("SIRT", proj_id, sin_id, vol_id, ARG.gpu)
    outputName = "SIRT_%s"%outputName
elif ARG.sart:
    cfg = createProjectorConfig("SART", proj_id, sin_id, vol_id, ARG.gpu)
    outputName = "SART_%s"%outputName
elif ARG.em:
    if not ARG.gpu:
        raise("EM is implemented only for GPU")
    else:
        cfg = createProjectorConfig("EM", proj_id, sin_id, vol_id, ARG.gpu)
    outputName = "EM_%s"%outputName
elif ARG.fbp:
    cfg = createProjectorConfig("FBP", proj_id, sin_id, vol_id, ARG.gpu)
    outputName = "FBP_%s"%outputName
    
if ARG.verbose:    
    print("Creating %s algorithm"%cfg["type"])
#cfg = {};
#cfg["type"] = 'CGLS'
#cfg["type"] = 'CGLS_GPU'
#cfg["ProjectorId"] = proj_id;
#cfg["ProjectionDataId"] = sin_id;
#cfg["ReconstructionDataId"] = vol_id;
cgls_id = astra.algorithm.create(cfg);
if ARG.fbp:
    astra.algorithm.run(cgls_id, 1)
else:
    astra.algorithm.run(cgls_id, ARG.itterations);
volume = astra.data2d.get(vol_id);

if ARG.verbose:
    print("Output volume has dimensions %dx%d and type %s"%(volume.shape[0], volume.shape[1], volume.dtype))

fullOutputName = os.path.join(ARG.output_folder, outputName)

if ARG.saveden:
    writeDenFile(volume, "%s.den"%(fullOutputName), ARG.force)
if ARG.savetiff:
    im = Image.fromarray(volume, mode='F') # float32
    im.save("%s.tiff"%(fullOutputName), "TIFF")
    
with open("%s.log"%(fullOutputName), 'wt') as f:
        json.dump(vars(ARG), f, indent=4)    
#garbage disposal
astra.data2d.delete(sin_id)
astra.data2d.delete(vol_id)
astra.projector.delete(proj_id);
astra.algorithm.delete(cgls_id);
