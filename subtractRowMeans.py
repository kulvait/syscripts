#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subtract row means from each frame in a DEN file.
"""

import argparse
from denpy import DEN
import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import numpy as np
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("inputDen")
parser.add_argument("outputDen")
parser.add_argument("--force", action="store_true")

ARG = parser.parse_args()

# First read dimensions of the DEN file
header = DEN.readHeader(ARG.inputDen)
if header["dimcount"] not in [2, 3]:
    print("File %s shall have dimension two or three" %
          (os.path.basename(ARG.inputDen)))
    sys.exit(-1)

if header["dimcount"] == 2:
    zdim = 1
else:
    zdim = header["dimspec"][2]

xdim = header["dimspec"][0]
ydim = header["dimspec"][1]

DEN.writeEmptyDEN(ARG.outputDen, [xdim, ydim, zdim],
                  header["type"],
                  force=ARG.force)
print("Created file [dimx, dimy, dimz] = [%d, %d, %d]" %
      (xdim, ydim, zdim))

for k in range(zdim):
    print("Start for %d" % k)
    start = time.time()

    f = DEN.getFrame(ARG.inputDen, k)
    row_means = np.mean(f, axis=1, keepdims=True)
    g = f - row_means

    DEN.writeFrame(ARG.outputDen, k, g, force=True)
    print("Elapsed %0.2fs" % (time.time() - start))
