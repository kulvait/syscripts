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
# pd.set_option('display.max_columns', 100) to display untruncated columns
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
from skimage.measure import block_reduce
from multiprocessing.dummy import Pool, Lock  # threads
import multiprocessing as mp
from multiprocessing import Value
import traceback
import os
import sys


from denpy import ZARPYRAMID  # OME-NGFF pyramids with plain zarr (replaces ome_zarr)
from denpy import ZARZS  # zip store that threads write in parallel

import re
from denpy import UTILS

parser = argparse.ArgumentParser()
parser.add_argument('inputFile', type=str)
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
                    choices=['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz',
                             'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
                    default='blosc-zstd',
                    help="Compression type (default: blosc-zstd).")
parser.add_argument('--clevel', type=int, default=5,
                    help="Compression level (default: 5).")
parser.add_argument('--zarrv3', action='store_true',
                    help="Write Zarr v3 store instead of v2.")
parser.add_argument('--ome', action='store_true',
                    help="Write an OME-NGFF image (OME-Zarr 0.5 with --zarrv3, else 0.4) with a pyramid, see --pyramid-*; "
                         "written with plain zarr by denpy.ZARPYRAMID.")
parser.add_argument('--pyramid-levels', type=int, default=5,
                    help="--ome: number of pyramid levels including full resolution (default 5, as ome-zarr).")
parser.add_argument('--pyramid-axes', choices=['yx', 'zyx'], default='yx',
                    help="--ome: reduce Y,X per level keeping Z (yx, default, as ome-zarr) or Z,Y,X (zyx, 3D).")
parser.add_argument('--pyramid-method', choices=['mean', 'nearest', 'max', 'min'], default='mean',
                    help="--ome: how a block becomes one sample of the next level (default mean).")
parser.add_argument('--pyramid-factor', type=int, default=2,
                    help="--ome: reduction factor between levels (default 2).")
parser.add_argument('--neuroglancer', action='store_true',
                    help="Write Zarr in Neuroglancer Precomputed format.")
parser.add_argument('--max-level', type=int, default=5,
                    help="Maximum pyramid level for pyramid")
parser.add_argument('--matlab-log', type=str, default=None,
                    help="Path to MATLAB log file for extracting parameters.")
parser.add_argument('--params', type=str, default=None,
                    help="Path to params file for extracting parameters.")
parser.add_argument('--binning-factor', type=int, default=1,
                    help="Binning factor used for extracting parameters from params file.")
parser.add_argument('--zip', action='store_true',
                    help="Write Zarr store as a single zip file instead of directory.")
parser.add_argument("-j", "--threads", default=-1, type=int,
                    help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
parser.add_argument("--verbose", action="store_true")

# ARG = parser.parse_args([])
ARG = parser.parse_args()

# Set up threading
if ARG.j < 0:
    ARG.j = mp.cpu_count()
    print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()" % (ARG.j))
elif ARG.j == 0:
    print("No threading will be used ARG.j=0.")
else:
    print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d" % (
        ARG.j, mp.cpu_count()))

_write_lock = None  # global in worker


def _init_writer_worker(lock):
    global _write_lock
    _write_lock = lock
# END threading setup


def get_compressor(name, clevel=5, zarrv3=False, outtype=None, endian="little"):
    """
    Return a zarr-compatible compressor/codec based on name and Zarr format version.

    Parameters
    ----------
    name : str
        Compression name (e.g., 'none', 'zstd', 'blosc-zstd', 'lz4', 'gzip', ...).
    clevel : int
        Compression level (meaning depends on the codec; Zstd/Blosc: 0..9 typical).
    zarrv3 : bool
        If True, return a Zarr v3 codec *pipeline* (list) suitable for `codecs=...`.
        If False, return a single compressor object (e.g., for Zarr v2 `compressor=`).
    outtype : Optional[Union[np.dtype, type, str]]
        Array dtype (e.g., np.uint16, 'uint16', np.dtype('uint16')). Used to set
        Blosc `typesize` (bytes per element). If None, defaults to itemsize=1.
    endian : str
        'little' or 'big' – used for v3 BytesCodec. Defaults to 'little'.
    """
    # Derive element size in bytes; default to 1 if unknown
    if outtype is not None:
        try:
            itemsize = np.dtype(outtype).itemsize
        except Exception:
            itemsize = 1
    else:
        itemsize = 1

    if not zarrv3:
        from numcodecs import Blosc, GZip
        # Old style compressors (zarr v2 compatible)
        if name == 'none':
            return None
        elif name == 'zstd' or name == 'blosc-zstd':
            return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
        elif name == 'lz4' or name == 'blosc-lz4':
            return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
        elif name == 'gzip' or name == 'blosc-zlib':
            return GZip(level=clevel)
        elif name == 'blosc' or name == 'blosc-blosclz':
            return Blosc(cname='blosclz', clevel=clevel, shuffle=Blosc.BITSHUFFLE, typesize=itemsize)
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
        codecs_chain = []
        if name == 'none':
            print("No compression selected for Zarr v3, returning empty codec chain.")
        elif name == "zstd":
            codecs_chain.append(codecs.ZstdCodec(level=clevel))
        elif name == "lz4":
            try:
                from zarr.codecs.numcodecs import LZ4  # zarr >= 3.1.3
            except ImportError:
                from numcodecs.zarr3 import LZ4
            codecs_chain.append(LZ4())  # LZ4 has no compression level
        elif name == "gzip":
            codecs_chain.append(codecs.GzipCodec(level=clevel))
        elif name == "blosc" or name == "blosc-blosclz":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.blosclz,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        elif name == "blosc-lz4":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.lz4,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        elif name == "blosc-lz4hc":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.lz4hc,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        elif name == "blosc-snappy":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.snappy,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        elif name == "blosc-zlib":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.zlib,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        elif name == "blosc-zstd":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.zstd,
                    clevel=clevel,
                    shuffle="shuffle",
                    typesize=itemsize,
                )
            )
        else:
            raise ValueError(f"Unknown compressor type '{name}' for Zarr v3")
        return codecs_chain


# Reading Matlab log

def read_matlab_log(file_path):
    """Reads the MATLAB log file and extracts relevant parameters."""
    parameters = {
        'raw_binning_factor': 1,
        'par_eff_pixel_size_um': 0.0,  # Store pixel size in um
        'par_eff_pixel_size_binned_um': 0.0  # Store binned pixel size in um
    }
    # Regex patterns to match the parameters in the log file
    binning_factor_pattern = r"(?:par\.)?raw_binning_factor\s*[:=]\s*(\d+)"
    eff_pixel_size_pattern = r"(?:par\.)?eff_pixel_size\s*[:=]\s*([\d.]+)\s*(\w+)"
    eff_pixel_size_binned_pattern = r"(?:par\.)?eff_pixel_size_binned\s*[:=]\s*([\d.]+)\s*(\w+)"
    # Open the log file
    with open(file_path, 'r') as f:
        content = f.read()
        # Find the effective pixel size and its unit (default to 0.0 and empty string if not found)
        match_pixel_size = re.search(eff_pixel_size_pattern, content)
        if match_pixel_size:
            value = float(match_pixel_size.group(1))
            unit = match_pixel_size.group(2).lower()

            if unit == "micron":
                parameters['par_eff_pixel_size_um'] = value  # Store in um
            elif unit == "um":
                parameters['par_eff_pixel_size_um'] = value  # Already in um
            else:
                raise ValueError(
                    f"Unknown unit for 'par.eff_pixel_size': {unit}. Expected 'micron' or 'um'.")
        # Find the binned effective pixel size and its unit
        match_pixel_size_binned = re.search(
            eff_pixel_size_binned_pattern, content)
        if match_pixel_size_binned:
            value = float(match_pixel_size_binned.group(1))
            unit = match_pixel_size_binned.group(2).lower()

            if unit == "micron":
                # Store in um
                parameters['par_eff_pixel_size_binned_um'] = value
            elif unit == "um":
                # Already in um
                parameters['par_eff_pixel_size_binned_um'] = value
            else:
                raise ValueError(
                    f"Unknown unit for 'par.eff_pixel_size_binned': {unit}. Expected 'micron' or 'um'.")
        # Find the raw_binning_factor (default 1)
        match_binning = re.search(binning_factor_pattern, content)
        if match_binning:
            parameters['raw_binning_factor'] = int(match_binning.group(1))
        elif parameters['par_eff_pixel_size_binned_um'] > 0.0 and parameters['par_eff_pixel_size_um'] > 0.0:
            parameters['raw_binning_factor'] = int(
                parameters['par_eff_pixel_size_um'] / parameters['par_eff_pixel_size_binned_um'])
    print("Extracted parameters from MATLAB log:", parameters)
    return parameters


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


def writeOMEZarr(inputDenFile, zarrFile, *,
                 force=False, dtype=None,
                 compression='zstd', clevel=5, zarrv3=False, spec=None, verbose=False):
    """Write a DEN file as an OME-NGFF image with a pyramid (denpy.ZARPYRAMID, plain zarr).

    Dimensions before Y, X are flattened to Z in the order of the frames in the file."""
    denInfo = DEN.readHeader(inputDenFile)
    shape = denInfo["shape"]
    dimy, dimx = shape[-2], shape[-1]
    n_frames = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    outtype = np.dtype(dtype) if dtype is not None else np.dtype(denInfo["type"])
    spec = spec if spec is not None else ZARPYRAMID.PyramidSpec()
    if zarrFile.endswith(".zip") is False and ARG.zip:
        raise ValueError("--ome --zip: give the output a .zip name")
    if verbose:
        print(colored(f"OME-NGFF pyramid: {spec.describe((n_frames, dimy, dimx))}", "green"))
    ZARPYRAMID.write_pyramid(
        zarrFile, lambda k: DEN.getFrame(inputDenFile, k), (n_frames, dimy, dimx), outtype, spec,
        zarr_format=3 if zarrv3 else 2,
        compressors=get_compressor(compression, clevel=clevel, zarrv3=zarrv3, outtype=outtype),
        overwrite=force, name=os.path.basename(inputDenFile), verbose=verbose)
    if verbose:
        print(f"OME-NGFF image written to {zarrFile}")


def writeDenFrameAtIndex(denFile, outArray, kIndex):
    """Read a single frame and write it to the specified index in the Zarr/numpy array."""
    # ---- Basic array checks ----
    if outArray is None or not hasattr(outArray, "shape") or not hasattr(outArray, "dtype"):
        raise TypeError(
            "outArray must have 'shape' and 'dtype' attributes (NumPy/Zarr-like).")

    if len(outArray.shape) < 3:
        raise ValueError(
            f"outArray must be 3D (Z, Y, X); got shape={outArray.shape}")
    dimy, dimx = outArray.shape[-2], outArray.shape[-1]
    # shape of the indexing dimensions (e.g., Z or ZT)
    index_shape = outArray.shape[:-2]
    max_flat_index = np.prod(index_shape) if len(index_shape) > 0 else 1
    if not (0 <= kIndex < max_flat_index):
        raise IndexError(
            f"kIndex {kIndex} out of bounds for array with shape {outArray.shape}")
    target_dtype = np.dtype(outArray.dtype)
    # ---- Read frame ----
    # kIndex is the flat frame number: the order of the frames in the DEN file, which is the C-order
    # flat index over the dimensions before Y, X (numpy shape[:-2])
    img = DEN.getFrame(denFile, kIndex)
    outIndex = np.unravel_index(kIndex, index_shape) if outArray.ndim > 3 else (kIndex,)
    # ---- Check shape ----
    if img.shape != (dimy, dimx):
        raise ValueError(
            f"Shape mismatch in DEN '{denFile}': got {img.shape}, expected {(dimy, dimx)}"
        )
    # ---- Cast to output dtype ALWAYS ----
    if img.dtype != target_dtype:
        try:
            img = img.astype(target_dtype, copy=False)
        except Exception as e:
            raise ValueError(
                f"Failed to convert image dtype from {img.dtype} to {target_dtype}: {e}"
            )
    outArray[tuple(int(i) for i in outIndex) + (slice(None), slice(None))] = img


def writeDenFrameAtIndex_worker(file, outArray, kIndex):
    """Worker function for multiprocessing that wraps writeDenFrameAtIndex and captures exceptions."""
    try:
        writeDenFrameAtIndex(file, outArray, kIndex)
        return {"file": file, "kIndex": kIndex, "error": None}
    except Exception as e:
        return {"file": file, "kIndex": kIndex, "error": traceback.format_exc()}


progress = Value('i', 0)   # atomic int


def progress_callback(result):
    # result is what the worker returned
    with progress.get_lock():
        progress.value += 1
        count = progress.value
    if ARG.verbose and count % 100 == 0:
        print(f"Written {count} frames, latest: index {result['kIndex']}.")
    if result["error"] is not None:
        print(colored(f"Error in worker for file '{result['file']}' at index {result['kIndex']}:\n{result['error']}", "red"))

def writeZarrFile(inputDenFile, zarrFile, *,
                  force=False, dtype=None,
                  compression='zstd', clevel=5, zarrv3=False, zipStore=False, verbose=False):
    """Convert a TIFF stack to a 3D Zarr array."""
    if os.path.exists(zarrFile):
        if force:
            if os.path.isdir(zarrFile):
                shutil.rmtree(zarrFile)
            else:
                os.remove(zarrFile)
        else:
            raise IOError(f"File {zarrFile} exists, use --force to overwrite")

    denInfo = DEN.readHeader(inputDenFile)
    den_shape = denInfo["shape"]
    den_shape_str = ", ".join(str(s) for s in den_shape)
    den_shape_str = f"({den_shape_str})"
    den_dimcount = len(den_shape)
    # For 3D or higher, we want to chunk along the last 2 dimensions (Y,X) and keep Z unchunked for better compression and access patterns.
    # 3D chunk_shape = (1, dimy, dimx)
    chunk_shape = np.ones_like(den_shape)
    for i in range(min(2, den_dimcount)):
        chunk_shape[den_dimcount - 1 - i] = den_shape[den_dimcount - 1 - i]
    chunk_shape = tuple(int(x) for x in chunk_shape)
    chunk_shape_str = ", ".join(str(s) for s in chunk_shape)
    chunk_shape_str = f"({chunk_shape_str})"
    zarr_version = 3 if zarrv3 else 2

    # Determine output dtype if none it will be same as input TIF
    den_dtype = denInfo["type"]
    if dtype is not None:
        outtype = np.dtype(dtype)
    else:
        outtype = den_dtype

    if verbose:
        msg = "Creating Zarr v{}: dtype={}, shape={}, chunk_shape={}, compression={}, clevel={}".format(
            zarr_version, outtype, den_shape_str, chunk_shape_str, compression, clevel)
        print(colored(msg, "green"))

    if zarrv3:
        codec = get_compressor(compression, clevel=clevel,
                               zarrv3=True, outtype=outtype)
    else:
        # Classic v2
        codec = get_compressor(compression, clevel=clevel,
                               zarrv3=False, outtype=outtype)

    # Create the base resolution level (s0)
    if zipStore or ARG.outputZarr.endswith('.zip'):
        store = ZARZS.ZipStore(ARG.outputZarr, mode='w')  # threads write their chunks in parallel
    else:
        store = ARG.outputZarr
    # Create Zarr array
    if zarrv3:
        # Use new create_array API
        zarr_array = zarr.create_array(
            store=store,
            shape=den_shape,
            chunks=chunk_shape,
            dtype=outtype,
            compressors=codec,
            zarr_format=3,
        )
    else:
        # Classic v2
        zarr_array = zarr.open_array(
            store=store,
            mode='w',
            shape=den_shape,
            chunks=chunk_shape,
            dtype=outtype,
            compressor=codec,
            zarr_format=2,
        )

    if den_dimcount < 3:
        den_array = DEN.getNumpyArray(inputDenFile)
        if den_array.shape != den_shape:
            raise ValueError(
                f"Shape mismatch in DEN file: got {den_array.shape}, expected {den_shape}")
        if den_array.dtype != outtype:
            try:
                den_array = den_array.astype(outtype)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert DEN array dtype from {den_array.dtype} to {outtype}: {e}")
        zarr_array[...] = den_array

    # Write images
    # shape of the indexing dimensions (e.g., Z or ZT)
    index_shape = den_shape[:-2]
    max_flat_index = np.prod(index_shape) if len(index_shape) > 0 else 1
    if ARG.j == 0:
        start = timer()
        for i in range(max_flat_index):
            writeDenFrameAtIndex(inputDenFile, zarr_array, i)
            if ARG.verbose and (i % 100 == 0 or i == max_flat_index - 1):
                elapsed = timer() - start
                print(
                    f"Written frame {i+1}/{max_flat_index} ({os.path.basename(inputDenFile)}) batch in {elapsed:.3f}s")
                start = timer()
    else:
        results = []
        with Pool(ARG.j, initializer=_init_writer_worker, initargs=(Lock(),)) as pool:
            for i in range(max_flat_index):
                res = pool.apply_async(writeDenFrameAtIndex_worker, args=(inputDenFile, zarr_array, i), callback=progress_callback)
                results.append(res)
            # Wait for all tasks to complete and check for exceptions
            pool.close()
            pool.join()
        errors = []
        for res in results:
            r = res.get()  # This will re-raise any exception from the worker
            if r["error"] is not None:
                errors.append((r["file"], r["kIndex"], r["error"]))
        if len(errors) > 0:
            print(
                colored(f"Encountered {len(errors)} errors during TIFF processing:", "red"))
            for tiffFile, kIndex, error in errors:
                print(
                    colored(f"Error processing '{tiffFile}' at index {kIndex}:\n{error}", "red"))
            raise RuntimeError(
                f"{len(errors)} errors occurred during TIFF processing. See above for details.")

    if ARG.verbose:
        print(colored(f"Zarr array written to {zarrFile}", "cyan"))


def downscale_image(image, factor=2):
    """Downscale the image by a given factor using average pooling."""
    return block_reduce(image, block_size=(factor, factor), func=np.mean)


def create_metadata(store):
    """Create .zarray and .zattrs files in the store."""
    # Assuming you have the necessary metadata for Neuroglancer format
    # Example: voxel size (in XYZ directions)
    store.attrs["resolution"] = [1.0, 1.0, 1.0]

    # Create the .zarray metadata file for the root "data" array (empty array to trigger the creation of .zarray)
    # Empty dataset to generate .zarray
    store.create_dataset("data", shape=(0, 0, 0), dtype='f4')
    store['data'].attrs["compressor"] = {"id": "zstd", "level": 5}

    # Create the .zattrs file (optional but recommended)
    store.attrs["description"] = "Neuroglancer Precomputed format"
    store.attrs["version"] = "v1.0"


def writeNeuroglancerZarr(inputDenFile, zarrFile, *,
                          force=False, dtype=None,
                          compression='zstd', clevel=5, max_level=3, zarrv3=False, verbose=False):
    """Convert a DEN file to a 3D Zarr array following Neuroglancer Precomputed format.

    Parameters:
    - inputDenFile: DEN file; dimensions before Y, X are flattened to Z in the order of its frames.
    - zarrFile: Path to save the resulting Zarr file.
    - force: Whether to overwrite the existing Zarr file if it exists.
    - dtype: Data type for the Zarr array (optional, default is from input TIF).
    - compression: Compression method for the Zarr array ('zstd' or 'gzip').
    - clevel: Compression level (1 to 9, higher means more compression).
    - max_level: Maximum number of pyramid levels (s0, s1, s2, ...).
    - zarrv3: Whether to use Zarr v3 format (default is Zarr v2).
    - verbose: Whether to print detailed output during processing.
    """
    # Check if the Zarr file already exists and handle based on force flag
    if os.path.exists(zarrFile):
        if force:
            shutil.rmtree(zarrFile)
        else:
            raise IOError(f"File {zarrFile} exists, use --force to overwrite")

    # Dimensions from the DEN header
    denInfo = DEN.readHeader(inputDenFile)
    dimy, dimx = denInfo["shape"][-2], denInfo["shape"][-1]
    n_images = int(np.prod(denInfo["shape"][:-2])) if len(denInfo["shape"]) > 2 else 1

    # Determine the output dtype if not provided
    if dtype is not None:
        outtype = np.dtype(dtype)
    else:
        outtype = np.dtype(denInfo["type"])

    # Precompute chunk shape based on image dimensions
    if dimx > 512 and dimy > 512:
        chunk_shape = (1, 256, 256)
    else:
        chunk_shape = (1, dimy, dimx)

    if verbose:
        print(f"Creating Zarr for Neuroglancer format (Zarr v{'3' if zarrv3 else '2'}): shape=({n_images}, {dimy}, {dimx}), "
              f"dtype={dtype}, compression={compression}, clevel={clevel}")
    # Default attributes for Neuroglancer Precomputed format
    # {"axes":["x","y","z"],"downsamplingFactors":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16]],"multiScale":true,"resolution":[25.08,25.08,25.08],"units":["um","um","um"]}
    voxel_size_um = [2.5, 2.5, 2.5]  # Default voxel size in micrometers
    # If MATLAB log is provided, extract voxel size
    if ARG.matlab_log is not None:
        matlab_params = read_matlab_log(ARG.matlab_log)
        pixel_size_um = matlab_params['par_eff_pixel_size_binned_um']
        print("Extracted pixel size from MATLAB log:", pixel_size_um, "um")
        if matlab_params['par_eff_pixel_size_binned_um'] > 0:
            voxel_size_um = [pixel_size_um, pixel_size_um, pixel_size_um]
        elif matlab_params['par_eff_pixel_size_um'] > 0:
            voxel_size_um = [matlab_params['par_eff_pixel_size_um']] * 3
    elif ARG.params is not None:
        params_dict = UTILS.readParamsFile(ARG.params)
        if 'pixel_size_x' in params_dict:
            voxel_size_um[0] = float(
                params_dict['pixel_size_x']) * ARG.binning_factor
            voxel_size_um[1] = float(
                params_dict['pixel_size_x']) * ARG.binning_factor
        if 'pixel_size_y' in params_dict:
            voxel_size_um[2] = float(
                params_dict['pixel_size_y']) * ARG.binning_factor

    attrs = {
        "axes": ["z", "y", "x"],
        "downsamplingFactors": [[2**i, 2**i, 2**i] for i in range(max_level + 1)],
        "multiScale": True,
        "resolution": [voxel_size_um[2], voxel_size_um[1], voxel_size_um[0]],
        "units": ["um", "um", "um"],
        "dimension_units": ["%f um" % voxel_size_um[2], "%f um" % voxel_size_um[1], "%f um" % voxel_size_um[0]],
    }
    ome_attrs = {
        "version": "0.5",
        "bioformats2raw.layout": 3,
    }
    # Create multiscales attr according to https://ngff.openmicroscopy.org/0.4/#multiscale-md spec
    multiscale_attr = [
        {
            "version": "0.5",
            "name": "AMBCAT",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "type": "skimage.measure.block_reduce",
            "metadata": {"command": "zarr_array[:] = block_reduce(zarr_array_s0, block_size=(2**level, 2**level, 2**level), func=np.mean)"},
        }
    ]
    datasets = []
    for level in range(max_level + 1):
        datasets.append({"path": f"s{level}",
                         "coordinateTransformations": [
                             {
                                 "type": "scale",
                                 "scale": [(2**level), (2**level), (2**level)]
                             }
                         ]
                         })
    multiscale_attr[0]["datasets"] = datasets
    ome_attrs["multiscales"] = multiscale_attr
    attrs["ome"] = ome_attrs
    if zarrv3:
        # Use new create_array API
        zarr_top_level = zarr.open_group(
            store=zarrFile,
            mode='w',
            attributes=attrs
        )
    else:
        # Classic v2
        zarr_top_level = zarr.open_group(
            zarrFile,
            mode='w',
            zarr_format=2,
            attributes=attrs
        )
    if zarrv3:
        codec = get_compressor(compression, clevel=clevel, zarrv3=True)
        print("For input compression:", compression, "clevel:", clevel)
        print(codec)
    else:
        # Classic v2
        codec = get_compressor(compression, clevel=clevel, zarrv3=False),
    # Create the base resolution level (s0)
    zarr_array_s0 = zarr_top_level.create_array('s0', shape=(
        n_images, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
    # Write images
    for i in range(n_images):
        start = timer()
        img = DEN.getFrame(inputDenFile, i) if len(denInfo["shape"]) > 2 else DEN.getNumpyArray(inputDenFile)
        zarr_array_s0[i, :, :] = img.astype(outtype, copy=False)
        if ARG.verbose and (i % 100 == 0 or i == n_images - 1):
            print(
                f"Written frame {i+1}/{n_images} of {os.path.basename(inputDenFile)} in {timer()-start:.3f}s")

    # Iterate through each resolution level (s0, s1, s2, ...)
    for level in range(1, max_level + 1):
        # Name the group for this level (e.g., "s0", "s1", "s2", ...)
        group_name = f"s{level}"
        level_shape = (max(1, n_images//(2**level)),
                       max(1, dimy//(2**level)), max(1, dimx//(2**level)))
        # Create the Zarr array for the current resolution level (group)
        zarr_array = zarr_top_level.create_array(
            group_name, shape=level_shape, dtype=outtype, chunks=chunk_shape, compressors=codec)
        zarr_array[:] = block_reduce(zarr_array_s0[:], block_size=(
            2**level, 2**level, 2**level), func=np.mean)
    if verbose:
        print(f"Neuroglancer Zarr pyramid written to {zarrFile}")


if __name__ == '__main__':
    # ---- Benchmarking section ----
    t_start = timer()

    den_total_size = os.path.getsize(ARG.inputFile)
    # ---- Main conversion ----

    # Basic stack info (from first image)
    denInfo = DEN.readHeader(ARG.inputFile)
    shape = denInfo["shape"]
    index_dims = shape[:-2]  # dimensions before Y,X
    n_frames = np.prod(index_dims) if len(index_dims) > 0 else 1
    dim_dimspec = denInfo["dimspec"]
    den_dtype = denInfo["type"]
    output_dtype = None
    if ARG.float32:
        output_dtype = np.float32
    elif ARG.float16:
        output_dtype = np.float16

    if ARG.ome:
        writeOMEZarr(
            ARG.inputFile,
            ARG.outputZarr,
            force=ARG.force,
            dtype=output_dtype,
            compression=ARG.compression,
            clevel=ARG.clevel,
            zarrv3=ARG.zarrv3,
            spec=ZARPYRAMID.PyramidSpec(levels=ARG.pyramid_levels, factor=ARG.pyramid_factor,
                                        axes=ARG.pyramid_axes, method=ARG.pyramid_method),
            verbose=ARG.verbose
        )
    elif ARG.neuroglancer:
        writeNeuroglancerZarr(
            ARG.inputFile,
            ARG.outputZarr,
            force=ARG.force,
            dtype=output_dtype,
            compression=ARG.compression,
            clevel=ARG.clevel,
            max_level=ARG.max_level,
            zarrv3=ARG.zarrv3,
            verbose=ARG.verbose
        )
    else:
        writeZarrFile(
            ARG.inputFile,
            ARG.outputZarr,
            force=ARG.force,
            dtype=output_dtype,
            compression=ARG.compression,
            clevel=ARG.clevel,
            zarrv3=ARG.zarrv3,
            zipStore=ARG.zip,
            verbose=ARG.verbose
        )

    total_time = timer() - t_start
    zarr_total_size = dir_size(ARG.outputZarr)

    # ---- Reporting ----
    print()
    print(colored("==== Conversion Summary ====", "cyan"))
    print(f"Input file: {ARG.inputFile} ({den_total_size/1e6:.2f} MB)")
    if ARG.zip or ARG.outputZarr.endswith('.zip'):
        print(f"Output ZIP file: {ARG.outputZarr} ({zarr_total_size/1e6:.2f} MB)")
    else:
        print(f"Output Zarr directory: {ARG.outputZarr} ({zarr_total_size/1e6:.2f} MB)")
    print(f"Stack details: shape={denInfo['shape']}, dimspec={denInfo['dimspec']}, dtype={den_dtype}")
    if den_total_size > 0 and zarr_total_size > 0:
        ratio = den_total_size / zarr_total_size
        to_den_ratio = zarr_total_size / den_total_size
        print(f"Compression ratio: {ratio:.3f}x, produced Zarr has {to_den_ratio:.3f} times size of the original DEN, reduction of {(1-to_den_ratio)*100:.1f}%")
    print(f"Total time: {format_time(total_time)}")
    avg_time_per_frame = total_time / n_frames if n_frames > 0 else 0
    frame_shape_str = ", ".join(str(s) for s in shape[-2:])
    print(f"Average per frame (shape {frame_shape_str}): {avg_time_per_frame:.3f} seconds")
    throughput = (zarr_total_size / 1e6) / total_time if total_time > 0 else 0
    print(f"Effective throughput: {throughput:.2f} MB/s")
    print("=============================")
