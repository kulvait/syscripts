#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize singular vectors in a DEN file to unit L2 norm and fix sign orientation.
If mean is negative, flips sign so mean is positive.

Usage:
    python normalizeSingularVectors.py input_vectors.den output_vectors.den
"""

import argparse
import numpy as np
from denpy import DEN

def parse_args():
    parser = argparse.ArgumentParser(description="Normalize singular vectors in DEN file.")
    parser.add_argument("input_den_file", type=str, help="Input DEN file containing singular vectors.")
    parser.add_argument("output_den_file", type=str, help="Output DEN file for normalized vectors.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output if exists.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Read header to get dimensions
    info = DEN.readHeader(args.input_den_file)
    dims = info["dimspec"]
    if len(dims) != 3:
        raise ValueError("Input DEN file must be 3D (dimx, dimy, num_vectors).")
    dimx, dimy, n_vectors = dims

    # Allocate output DEN
    DEN.writeEmptyDEN(args.output_den_file, dims, force=args.force)

    # Process vectors one by one
    for i in range(n_vectors):
        vec = DEN.getFrame(args.input_den_file, i).flatten()

        # Normalize L2 norm
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        # Flip sign if mean < 0
        if vec.mean() < 0:
            vec = -vec

        # Reshape and write frame
        DEN.writeFrame(args.output_den_file, i, vec.reshape((dimy, dimx)), force=True)

    print(f"Normalized {n_vectors} vectors and saved to {args.output_den_file}")

if __name__ == "__main__":
    main()
