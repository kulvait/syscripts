#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8.10.2025
Convert a series of tiff images to zarr format

@author: Vojtěch Kulvait
"""
import h5py
import pandas as pd
import statistics
from PIL import Image
from PIL.TiffTags import TAGS
#pd.set_option('display.max_columns', 100) to display untruncated columns
import sys
import os
import argparse
from denpy import DEN
from denpy import PETRA
import glob
import numpy as np
from termcolor import colored
from timeit import default_timer as timer
import zarr
import shutil
from pathlib import Path


from ome_zarr.format import FormatV04
from ome_zarr.format import FormatV05
from ome_zarr.io import parse_url as omezarr_parse_url
from ome_zarr.writer import write_image as omezarr_write_image
from ome_zarr.writer import add_metadata as omezarr_add_metadata



parser = argparse.ArgumentParser()
parser.add_argument('inputTifFiles', nargs='+', type=str)
parser.add_argument("outputZarr", type=str)
parser.add_argument("--force", action="store_true")
parser.add_argument(
    "--float32",
    help="Force output as float32 regardless of input dtype.",
    action="store_true"
)
parser.add_argument(
    "--float16",
    help="Force output as float16 regardless of input dtype.",
    action="store_true"
)
parser.add_argument('--compression', type=str,
                    choices=['none', 'zstd', 'lz4', 'gzip', 'blosc'],
                    default='zstd',
                    help="Compression type (default: zstd)")
parser.add_argument('--clevel', type=int, default=5,
                    help="Compression level (default: 5).")
parser.add_argument('--zarrv3', action='store_true', help="Write Zarr v3 store instead of v2.")
parser.add_argument('--ome', action='store_true', help="Write OME-Zarr instead of plain.")
parser.add_argument("--verbose", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()

def get_compressor(name, clevel=5):
    """Return a zarr-compatible compressor based on name."""
    if name == 'none':
        return None
    elif name == 'zstd':
        return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    elif name == 'lz4':
        return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    elif name == 'gzip':
        return GZip(level=clevel)
    elif name == 'blosc':
        return Blosc(cname='blosclz', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    else:
        raise ValueError(f"Unknown compression type: {name}")

def get_compressor(name, clevel=5, zarrv3=False):
    """Return a zarr-compatible compressor/codec based on name and zarr format version."""
    if not zarrv3:
        from numcodecs import Blosc, GZip as NcGZip
        # Old style compressors (zarr v2 compatible)
        if name == 'none':
            return None
        elif name == 'zstd':
            return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        elif name == 'lz4':
            return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        elif name == 'gzip':
            return NcGZip(level=clevel)
        elif name == 'blosc':
            return Blosc(cname='blosclz', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        else:
            raise ValueError(f"Unknown compression type: {name}")
    else:
        # ---- Zarr v3 codecs (lazy import for safety) ----
        try:
            import zarr.codecs as codecs
        except ImportError:
            raise ImportError(
                "Zarr v3 codec system not available in this version of zarr. "
                "Please upgrade to zarr>=2.18.0."
            )

        # Map names to codecs
        # Always start with BytesCodec()
        codecs_chain = [codecs.BytesCodec()]

        if name == "zstd":
            codecs_chain.append(codecs.ZstdCodec(level=clevel))
        elif name == "lz4":
            codecs_chain.append(codecs.LZ4Codec(level=clevel))
        elif name == "gzip":
            codecs_chain.append(codecs.GzipCodec(level=clevel))
        elif name == "blosc":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.zstd,
                    clevel=clevel,
                    shuffle=codecs.BloscShuffle.shuffle,
                )
            )
        else:
            raise ValueError(f"Unknown compressor type '{name}' for Zarr v3")


def dir_size(path):
    """Return total size of a directory or file in bytes."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    elif p.is_dir():
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    else:
        return 0

def format_time(seconds):
    """Return time as H:MM:SS string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:05.2f}"

# -------------------------------
# Core functionality
# -------------------------------
def writeOMEZarr(inputTifFiles, zarrFile, *,
                  force=False, dtype=None,
                  compression='zstd', clevel=5, zarrv3=False, verbose=False):
    if os.path.exists(zarrFile):
        if force:
            shutil.rmtree(zarrFile)
        else:
            raise IOError(f"File {zarrFile} exists, use --force to overwrite")

    im0 = np.array(Image.open(inputTifFiles[0]))
    dimy, dimx = im0.shape
    n_images = len(inputTifFiles)

    # Determine output dtype if none it will be same as input TIF
    if dtype is not None:
        outtype = np.dtype(dtype)
    else:
        outtype = im0.dtype

    if verbose:
        print(f"Creating OME-Zarr v3: shape=({n_images}, {dimy}, {dimx}), dtype={dtype}, compression={compression}, clevel={clevel}")

    # Precompute chunk shape based on image dimensions
    if dimx > 512 and dimy > 512:
        chunk_shape = (1, 256, 256)
    else:
        chunk_shape = (1, dimy, dimx)

    # Create Zarr store and group (using OME-Zarr v3 format)
    if zarrv3:
        store = omezarr_parse_url(zarrFile, mode="w", fmt=FormatV05()).store
    else:
        store = omezarr_parse_url(zarrFile, mode="w", fmt=FormatV04()).store
    root = zarr.group(store=store)

    # Prepare the 3D image data array (ZYX axes)
    image_data = np.zeros((n_images, dimy, dimx), dtype=outtype)

    for i, f in enumerate(inputTifFiles):
        start = timer()
        img = np.array(Image.open(f))
        if dtype == np.float32:
            img = img.astype(np.float32)
        if img.shape != (dimy, dimx):
            raise ValueError(f"Shape mismatch in {f}: got {img.shape}, expected {(dimy, dimx)}")
        image_data[i, :, :] = img
        if verbose and (i % 100 == 0 or i == n_images - 1):
            print(f"Written frame {i+1}/{n_images} ({os.path.basename(f)}) in {timer()-start:.3f}s")

    # Write image data to OME-Zarr
    omezarr_write_image(image=image_data, group=root, axes="zyx", 
                storage_options=dict(chunks=chunk_shape, compressor=get_compressor(compression, clevel=clevel, zarrv3=True)))

    # Add optional metadata for visualization and analysis
    omezarr_add_metadata(root, {"omero": {
        "channels": [{"color": "FF0000", "window": {"start": 0, "end": 20, "min": 0, "max": 255},
                      "label": "Random Channel", "active": True}]
    }})

    if verbose:
        print(f"OME-Zarr array written to {zarrFile}")



def writeZarrFile(inputTifFiles, zarrFile, *,
                  force=False, dtype=None,
                  compression='zstd', clevel=5, zarrv3=False, verbose=False):
    """Convert a TIFF stack to a 3D Zarr array."""
    if os.path.exists(zarrFile):
        if force:
            shutil.rmtree(zarrFile)
        else:
            raise IOError(f"File {zarrFile} exists, use --force to overwrite")

    im0 = np.array(Image.open(inputTifFiles[0]))
    dimy, dimx = im0.shape
    n_images = len(inputTifFiles)
    zarr_version = 3 if zarrv3 else 2

    # Determine output dtype if none it will be same as input TIF
    if dtype is not None:
        outtype = np.dtype(dtype)
    else:
        outtype = im0.dtype

    if verbose:
        print(colored(
            f"Creating Zarr v{zarr_version}: shape=({n_images}, {dimy}, {dimx}), "
            f"dtype={dtype}, compression={compression}, clevel={clevel}",
            "green"))


    # Precompute chunk shape based on image dimensions
    if dimx > 512 and dimy > 512:
        chunk_shape = (1, 256, 256)
    else:
        chunk_shape = (1, dimy, dimx)

    # Create Zarr array
    if zarrv3:
        # Use new create_array API
        zarr_array = zarr.open(
            store=zarrFile,
            mode='w',
            shape=(n_images, dimy, dimx),
            chunks=chunk_shape,
            dtype=dtype,
            codecs=get_compressor(compression, clevel=clevel, zarrv3=True),
        )
    else:
        # Classic v2
        zarr_array = zarr.open(
            zarrFile,
            mode='w',
            shape=(n_images, dimy, dimx),
            chunks=chunk_shape,
            dtype=dtype,
            compressor=get_compressor(compression, clevel=clevel, zarrv3=False),
            zarr_format=2,
        )

    # Write images
    for i, f in enumerate(inputTifFiles):
        start = timer()
        img = np.array(Image.open(f))
        if ARG.float32:
            img = img.astype(np.float32)

        if img.shape != (dimy, dimx):
            raise ValueError(f"Shape mismatch in {f}: got {img.shape}, expected {(dimy, dimx)}")

        zarr_array[i, :, :] = img

        if ARG.verbose and (i % 100 == 0 or i == n_images - 1):
            print(f"Written frame {i+1}/{n_images} ({os.path.basename(f)}) in {timer()-start:.3f}s")

    if ARG.verbose:
        print(colored(f"Zarr array written to {zarrFile}", "cyan"))

if __name__ == '__main__':
    # ---- Benchmarking section ----
    t_start = timer()

    tiff_total_size = sum(os.path.getsize(f) for f in ARG.inputTifFiles)
    # ---- Main conversion ----

    # Basic stack info (from first image)
    im0 = np.array(Image.open(ARG.inputTifFiles[0]))
    dimy, dimx = im0.shape
    tif_dtype = im0.dtype
    n_images = len(ARG.inputTifFiles)
    output_dtype = None
    if ARG.float32:
        output_dtype = np.float32
    elif ARG.float16:
        output_dtype = np.float16

    
    if ARG.ome:
        writeOMEZarr(
            ARG.inputTifFiles,
            ARG.outputZarr,
            force=ARG.force,
            dtype=output_dtype,
            compression=ARG.compression,
            clevel=ARG.clevel,
            zarrv3=ARG.zarrv3,
            verbose=ARG.verbose
        )
    else:
        writeZarrFile(
            ARG.inputTifFiles,
            ARG.outputZarr,
            force=ARG.force,
            dtype=output_dtype,
            compression=ARG.compression,
            clevel=ARG.clevel,
            zarrv3=ARG.zarrv3,
            verbose=ARG.verbose
        )

    total_time = timer() - t_start
    zarr_total_size = dir_size(ARG.outputZarr)

    # ---- Reporting ----
    print()
    print(colored("==== Conversion Summary ====", "cyan"))
    print(f"Input TIFF stack: {tiff_total_size/1e6:.2f} MB")
    print(f"Stack details: {n_images} TIFF files, shape={dimy}×{dimx}, dtype={tif_dtype}")
    print(f"Output Zarr store: {zarr_total_size/1e6:.2f} MB")
    if tiff_total_size > 0:
        ratio = zarr_total_size / tiff_total_size
        print(f"Compression ratio: {ratio:.3f}x ({(1 - ratio)*100:.1f}% smaller)")
    print(f"Total time: {format_time(total_time)}")
    n_images = len(ARG.inputTifFiles)
    print(f"Average per frame: {total_time / n_images:.3f} s")
    throughput = (tiff_total_size / 1e6) / total_time if total_time > 0 else 0
    print(f"Effective throughput: {throughput:.2f} MB/s")
    print("=============================")

