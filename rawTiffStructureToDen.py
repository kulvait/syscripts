#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:15:23 2022

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
from libtiff import TIFF
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN


parser = argparse.ArgumentParser()
parser.add_argument("inputH5")
parser.add_argument("outputDir")
parser.add_argument("--force", action="store_true")

ARG = parser.parse_args()
#ARG = parser.parse_args(["/home/user/desy_example_data/syn0101_17L_Ti_12w_000_nexus.h5", "/tmp/desy", "--force"])

#To create dataframe with given columns
def insertToDf(df, dat, name):
    time=dat["%s/time"%(name)]
    value=dat["%s/value"%(name)]
    for i in range(len(value)):
        t=time[i]
        v=value[i]
        df.loc[t][name]=v
        
#To write dataframe to den
def writeDenFile(df, inputDir, denFile, force=False):
    if os.path.exists(denFile):
        if force:
            os.remove(denFile)
        else:
            raise IOError("File %s exists, add force to overwrite"%(denFile))
    df = df.sort_values("time", ascending=True)
    fileStr = df["image_file"].iloc[0] #In some versions this is string but in some bytes
    if isinstance(fileStr, bytes):
        fileStr = fileStr.decode("utf-8")
    file = os.path.join(inputDir, fileStr)
    img=TIFF.open(file)
    img = img.read_image()
    dimy=img.shape[0]
    dimx=img.shape[1]
    dimz=len(df)
    print("Creating %s %dx%dx%d"%(denFile, dimx, dimy, dimz))
    DEN.writeEmptyDEN(denFile, dimx, dimy, dimz, True)
    for i in range(len(df)):
        fileStr = df["image_file"].iloc[i] #In some versions this is string but in some bytes
        if isinstance(fileStr, bytes):
            fileStr = fileStr.decode("utf-8")
        img=TIFF.open(os.path.join(inputDir, fileStr))
        img = img.read_image()
        print("Writing %d-th file %s of shape %d,%d into %s"%(i, fileStr, img.shape[0], img.shape[1], denFile))
        DEN.writeFrame(denFile, i, img, True)

#Parse input data
inputDir = os.path.dirname(ARG.inputH5)
if not os.path.exists(ARG.outputDir):
    print("Create dir %s"%(ARG.outputDir))
    os.mkdir(ARG.outputDir)

f = h5py.File(ARG.inputH5, 'r')
data = f["entry/scan/data"]
#Now each subfolder has two datasets (time, value)
#Time is uint64 and has a representation of 'datetime64[ms]'
#Extract all the information into dtuple list
labels = list(data.keys())
if len(labels) < 1:
    sys.exit("Error: labels count is %d!"%(labels.count))
#print(data["image_file/time"].value)
#dataset.value attribute was removed after h5py 2.9.0
df = pd.DataFrame(columns=labels+["time"], index=list(data["%s/time"%(labels[0])][()]))

for ind in df.index:
    df.loc[ind]["time"] = pd.to_datetime(ind, unit="ms")
for lab in labels:
    insertToDf(df, data, lab)
#imageFiles=f["entry/scan/data/image_file/value"].value
#imageTimes=np.array(f["entry/scan/data/image_file/time"].value, dtype='datetime64[ms]')
#Most likely this is np.datetime64[ms]
#Dark fields, white fields, scan
dark = df.loc[df["image_key"]==2]
white = df.loc[df["image_key"]==1]    
scan = df.loc[df["image_key"]==0]

writeDenFile(dark, inputDir, os.path.join(ARG.outputDir, "dar.den"), ARG.force) 
writeDenFile(white, inputDir, os.path.join(ARG.outputDir, "ref.den"), ARG.force) 
writeDenFile(scan, inputDir, os.path.join(ARG.outputDir, "img.den"), ARG.force) 
#Now create reduced scan frame
#tst = scan.head(2)
#writeDenFile(tst, inputDir, os.path.join(ARG.outputDir, "x.den"), ARG.force)    


    
#        
#
#
#
#
#scan.sort_values("time", ascending=True)
#for f in scan["image_file"]:
#    print(f)


