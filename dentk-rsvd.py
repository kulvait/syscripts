#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perform Robust PCA (Robust SVD) on DEN file using PyRobustPCA.
@author: Vojtech Kulvait
"""

import argparse
import numpy as np
from denpy import DEN
from PyRobustPCA import RobustPCADetMCD
from PyRobustPCA import RobustPCAOGK
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform robust SVD on DEN file and extract singular values, optionally output U and V."
    )
    parser.add_argument("input_den_file", type=str,
                        help="File in a DEN format to process, with shape T x 1 x N.")
    parser.add_argument("output_S_values", type=str,
                        help="File in a DEN format to output singular values to.")
    parser.add_argument("--vectorsV", type=str,
                        help="File in a DEN format to output V vectors to.")
    parser.add_argument("--vectorsU", type=str,
                        help="File in a DEN format to output U vectors to.")
    parser.add_argument("--max-s-vals", type=int, default=None,
                        help="Report at most this number of singular values.")
    parser.add_argument("--explain-procent", type=float, default=None,
                        help="Stop reporting when cumulative variance exceeds this percentage.")
    parser.add_argument("--method", type=str, default="ogk", choices=["mcd", "ogk"],
                        help="Choose robust PCA method: 'mcd' (default) or 'ogk'.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite of output files if they exist.")
    return parser.parse_args()

def main():
    args = parse_args()

    # === Load DEN data ===
    info = DEN.readHeader(args.input_den_file)
    if len(info["dimspec"]) != 3:
        raise ValueError(f"{args.input_den_file} must have 3 dimensions (got {len(info['dimspec'])}).")
    
    dimx, dimy, imageCount = info["dimspec"]
    #    X = []
    
    #    for k in range(imageCount):
    #        frame = DEN.getFrame(args.input_den_file, k)
    #        X.append(frame.flatten())
    
    #    X = np.array(X)  # Shape: (T, N)
    
    # Previsous code was commented because X of the shape (frames, pixels) was alocating pixels * pixels memory, therefore we work with transposed matrix.
    # Prepare data matrix with shape (pixels, frames)
    # This matrix is effectively X_transposed compared to your other code's expected shape (frames, pixels)
    X_transposed = np.zeros((dimx * dimy, imageCount), dtype=np.float32)
    for k in range(imageCount):
        frame = DEN.getFrame(args.input_den_file, k)
        X_transposed[:, k] = frame.flatten()
    
    # Run Robust PCA on X_transposed (shape: pixels x frames)
    start = time.time()
    if args.method == "mcd":
        pca = RobustPCADetMCD()
        pca.fit(X_transposed, n_iter=2)
    elif args.method == "ogk":
        pca = RobustPCAOGK()
        pca.fit(X_transposed)
    else:
        raise ValueError("Unsupported method")
    end = time.time()
    print(f"Robust PCA ({args.method}) took {end - start:.2f} seconds")
    
    # Extract singular values (scores)
    S_vals = pca.get_scores().astype(np.float32)
    
    # Determine how many components to keep
    if args.explain_procent is not None:
        total = np.sum(S_vals)
        cumulative = np.cumsum(S_vals) / total * 100
        num_components = np.searchsorted(cumulative, args.explain_procent) + 1
    elif args.max_s_vals is not None:
        num_components = min(len(S_vals), args.max_s_vals)
    else:
        num_components = len(S_vals)
    
    # Write singular values as a DEN file (shape 1x1xnum_components)
    DEN.writeEmptyDEN(args.output_S_values, (1, 1, num_components), force=True)
    for i in range(num_components):
        DEN.writeFrame(args.output_S_values, i, np.array([[S_vals[i]]], dtype=np.float32), force=True)


    # === Extract U results of transposed X which needs to be computed anyway ===
    if args.vectorsV or args.vectorsU:
        X_transposed_V = pca.get_principal_components().astype(np.float32)  # V.T in standard SVD
        X_transposed_V = X_transposed_V[:, :num_components]
        if args.vectorsU:
            U = X_transposed_V
            DEN.writeEmptyDEN(args.vectorsU, (U.shape[0], 1, U.shape[1]), force=True)
            for i in range(U.shape[1]):
                DEN.writeFrame(args.vectorsU, i, U[:, i].reshape((U.shape[0], 1)), force=True)

        if args.vectorsV:
            V = X_transposed @ X_transposed_V  # V = X^T @ U
            DEN.writeEmptyDEN(args.vectorsV, (dimx, dimy, V.shape[1]), force=True)
            for i in range(V.shape[1]):
                vec = V[:, i]
                # Normalize to L2 norm 1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                # Flip sign if mean is negative
                if vec.mean() < 0:
                    vec = -vec
                DEN.writeFrame(args.vectorsV, i, vec.reshape((dimy, dimx)), force=True)
             
    print(f"Processed {imageCount} frames -> {num_components} robust SVD components.")
    print(f"Singular values: {S_vals}")

if __name__ == "__main__":
    main()
