#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from denpy import DEN

parser = argparse.ArgumentParser(
    description="Create shifted projection-matrix files by adding COR offsets to an existing matrix file."
)
parser.add_argument("input_pm", help="Input DEN file with baseline projection matrices")
parser.add_argument("output_pm", help="Output DEN file prefix for shifted projection matrices")
parser.add_argument("--offset-pix", type=float, default=0.0, help="Number of pixels to shift the projection matrices (default: 0)")
parser.add_argument("--force", action="store_true", help="Overwrite existing files")
ARG = parser.parse_args()


def check_output(path):
    if os.path.exists(path):
        if ARG.force:
            os.remove(path)
        else:
            print(f"Output exists: {path}. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)


base = DEN.getNumpyArray(ARG.input_pm)

if base.ndim != 3 or base.shape[2] != 4 or base.shape[1] not in (1, 2):
    raise ValueError(f"Unexpected projection matrix shape: {base.shape}")

arr = np.array(base, copy=True)
arr[:, 0, 3] += ARG.offset_pix
DEN.storeNdarrayAsDEN(ARG.output_pm, arr, force=ARG.force)
