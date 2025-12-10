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
from skimage.measure import block_reduce
import dask.array as da


from ome_zarr.format import FormatV04
from ome_zarr.format import FormatV05
from ome_zarr.io import parse_url as omezarr_parse_url
from ome_zarr.writer import write_image as omezarr_write_image
from ome_zarr.writer import add_metadata as omezarr_add_metadata

import re
from denpy import UTILS

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
                    choices=['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
                    default='blosc-zstd',
                    help="Compression type (default: blosc-zstd).")
parser.add_argument('--clevel', type=int, default=5,
                    help="Compression level (default: 5).")
parser.add_argument('--zarrv3', action='store_true', help="Write Zarr v3 store instead of v2.")
parser.add_argument('--ome', action='store_true', help="Write OME-Zarr instead of plain.")
parser.add_argument('--neuroglancer', action='store_true', help="Write Zarr in Neuroglancer Precomputed format.")
parser.add_argument('--max-level', type=int, default=5, help="Maximum pyramid level for pyramid")
parser.add_argument('--matlab-log', type=str, default=None, help="Path to MATLAB log file for extracting parameters.")
parser.add_argument('--params', type=str, default=None, help="Path to params file for extracting parameters.")
parser.add_argument('--binning-factor', type=int, default=1, help="Binning factor used for extracting parameters from params file.")
parser.add_argument("--verbose", action="store_true")

#ARG = parser.parse_args([])
ARG = parser.parse_args()

def get_compressor(name, clevel=5):
    """Return a zarr-compatible compressor based on name."""
    if name == 'none':
        return None
    elif name == 'zstd' or name == 'blosc-zstd':
        return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    elif name == 'lz4' or name == 'blosc-lz4':
        return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    elif name == 'gzip' or name == 'blosc-zlib':
        return GZip(level=clevel)
    elif name == 'blosc' or name == 'blosc-blosclz':
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
        elif name == 'zstd' or name == 'blosc-zstd':
            return Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        elif name == 'lz4' or name == 'blosc-lz4':
            return Blosc(cname='lz4', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        elif name == 'gzip' or name == 'blosc-zlib':
            return GZip(level=clevel)
        elif name == 'blosc' or name == 'blosc-blosclz':
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
        codecs_chain = []
        if name == 'none':
            print("No compression selected for Zarr v3, returning empty codec chain.")
        elif name == "zstd":
            codecs_chain.append(codecs.ZstdCodec(level=clevel))
        elif name == "lz4":
            codecs_chain.append(codecs.LZ4Codec(level=clevel))
        elif name == "gzip":
            codecs_chain.append(codecs.GzipCodec(level=clevel))
        elif name == "blosc" or name == "blosc-blosclz":
            codecs_chain.append(
                codecs.BloscCodec(
                    cname=codecs.BloscCname.blosclz,
                    clevel=clevel,
                    shuffle=codecs.BloscShuffle.shuffle,
                )
            )
        elif name == "blosc-lz4":
                codecs_chain.append(
                        codecs.BloscCodec(
                        cname=codecs.BloscCname.lz4,
                        clevel=clevel,
                        shuffle=codecs.BloscShuffle.shuffle,
                        )
                )
        elif name == "blosc-lz4hc":
                codecs_chain.append(
                        codecs.BloscCodec(
                        cname=codecs.BloscCname.lz4hc,
                        clevel=clevel,
                        shuffle=codecs.BloscShuffle.shuffle,
                        )
                )
        elif name == "blosc-snappy":
                codecs_chain.append(
                        codecs.BloscCodec(
                        cname=codecs.BloscCname.snappy,
                        clevel=clevel,
                        shuffle=codecs.BloscShuffle.shuffle,
                        )
                )
        elif name == "blosc-zlib":
                codecs_chain.append(
                        codecs.BloscCodec(
                        cname=codecs.BloscCname.zlib,
                        clevel=clevel,
                        shuffle=codecs.BloscShuffle.shuffle,
                        )
                )
        elif name == "blosc-zstd":
                codecs_chain.append(
                        codecs.BloscCodec(
                        cname=codecs.BloscCname.zstd,
                        clevel=clevel,
                        shuffle=codecs.BloscShuffle.shuffle,
                        )
                )
        else:
            raise ValueError(f"Unknown compressor type '{name}' for Zarr v3")
        return codecs_chain


#Reading Matlab log

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
                raise ValueError(f"Unknown unit for 'par.eff_pixel_size': {unit}. Expected 'micron' or 'um'.")
        # Find the binned effective pixel size and its unit
        match_pixel_size_binned = re.search(eff_pixel_size_binned_pattern, content)
        if match_pixel_size_binned:
            value = float(match_pixel_size_binned.group(1))
            unit = match_pixel_size_binned.group(2).lower()

            if unit == "micron":
                parameters['par_eff_pixel_size_binned_um'] = value  # Store in um
            elif unit == "um":
                parameters['par_eff_pixel_size_binned_um'] = value  # Already in um
            else:
                raise ValueError(f"Unknown unit for 'par.eff_pixel_size_binned': {unit}. Expected 'micron' or 'um'.")
        # Find the raw_binning_factor (default 1)
        match_binning = re.search(binning_factor_pattern, content)
        if match_binning:
            parameters['raw_binning_factor'] = int(match_binning.group(1))
        elif parameters['par_eff_pixel_size_binned_um'] > 0.0 and parameters['par_eff_pixel_size_um'] > 0.0:
            parameters['raw_binning_factor'] = int(parameters['par_eff_pixel_size_um'] / parameters['par_eff_pixel_size_binned_um'])
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
                storage_options=dict(chunks=chunk_shape, compressor=get_compressor(compression, clevel=clevel, zarrv3=zarrv3)))

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



def downscale_image(image, factor=2):
    """Downscale the image by a given factor using average pooling."""
    return block_reduce(image, block_size=(factor, factor), func=np.mean)


def create_metadata(store):
    """Create .zarray and .zattrs files in the store."""
    # Assuming you have the necessary metadata for Neuroglancer format
    store.attrs["resolution"] = [1.0, 1.0, 1.0]  # Example: voxel size (in XYZ directions)

    # Create the .zarray metadata file for the root "data" array (empty array to trigger the creation of .zarray)
    store.create_dataset("data", shape=(0, 0, 0), dtype='f4')  # Empty dataset to generate .zarray
    store['data'].attrs["compressor"] = {"id": "zstd", "level": 5}

    # Create the .zattrs file (optional but recommended)
    store.attrs["description"] = "Neuroglancer Precomputed format"
    store.attrs["version"] = "v1.0"

def writeNeuroglancerZarr(inputTifFiles, zarrFile, *,
                          force=False, dtype=None,
                          compression='zstd', clevel=5, max_level=3, zarrv3=False, verbose=False):
    """Convert a TIFF stack to a 3D Zarr array following Neuroglancer Precomputed format.
    
    Parameters:
    - inputTifFiles: List of TIFF files containing the image stack.
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

    # Load the first image to get the dimensions
    im0 = np.array(Image.open(inputTifFiles[0]))  
    dimy, dimx = im0.shape
    n_images = len(inputTifFiles)

    # Determine the output dtype if not provided
    if dtype is not None:
        outtype = np.dtype(dtype)
    else:
        outtype = im0.dtype

    # Precompute chunk shape based on image dimensions
    if dimx > 512 and dimy > 512:
        chunk_shape = (1, 256, 256)
    else:
        chunk_shape = (1, dimy, dimx)

    if verbose:
        print(f"Creating Zarr for Neuroglancer format (Zarr v{'3' if zarrv3 else '2'}): shape=({n_images}, {dimy}, {dimx}), "
              f"dtype={dtype}, compression={compression}, clevel={clevel}")
    #Default attributes for Neuroglancer Precomputed format
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
            voxel_size_um = matlab_params['par_eff_pixel_size_um']
    elif ARG.params is not None:
        params = UTILS.readParamsFile(ARG.params)
        if 'pixel_size_x' in params_dict:
            voxel_size_um[0] = float(params_dict['pixel_size_x']) * ARG.binning_factor
            voxel_size_um[1] = float(params_dict['pixel_size_x']) * ARG.binning_factor
        if 'pixel_size_y' in params_dict:
            voxel_size_um[2] = float(params_dict['pixel_size_y']) * ARG.binning_factor

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
        "metadata": { "command": "zarr_array[:] = block_reduce(zarr_array_s0, block_size=(2**level, 2**level, 2**level), func=np.mean)" },
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
    zarr_array_s0 = zarr_top_level.create_array('s0', shape=(n_images, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
    # Write images
    for i, f in enumerate(inputTifFiles):
        start = timer()
        img = np.array(Image.open(f))
        if ARG.float32:
            img = img.astype(np.float32)
        if img.shape != (dimy, dimx):
            raise ValueError(f"Shape mismatch in {f}: got {img.shape}, expected {(dimy, dimx)}")
        zarr_array_s0[i, :, :] = img
        if ARG.verbose and (i % 100 == 0 or i == n_images - 1):
            print(f"Written frame {i+1}/{n_images} ({os.path.basename(f)}) in {timer()-start:.3f}s")

    # Iterate through each resolution level (s0, s1, s2, ...)
    for level in range(1, max_level + 1):
        # Name the group for this level (e.g., "s0", "s1", "s2", ...)
        group_name = f"s{level}"
        level_shape = (max(1, n_images//(2**level)), max(1, dimy//(2**level)), max(1, dimx//(2**level)))
        # Create the Zarr array for the current resolution level (group)
        zarr_array = zarr_top_level.create_array(group_name, shape=level_shape, dtype=outtype, chunks=chunk_shape, compressors=codec)
        zarr_array[:] = block_reduce(zarr_array_s0[:], block_size=(2**level, 2**level, 2**level), func=np.mean)
        # Use Dask for efficient downsampling
        def block_reduce_block(block):
            return block_reduce(block, block_size=(2**level, 2**level, 2**level), func=np.mean)
        #zarr_da = da.from_zarr(zarr_array_s0)
        #reduced = da.map_blocks(block_reduce_block, zarr_da, dtype=zarr_da.dtype)
        #zarr_array[:] = reduced.compute()
    if verbose:
        print(f"Neuroglancer Zarr pyramid written to {zarrFile}")



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
    elif ARG.neuroglancer:
        writeNeuroglancerZarr(
            ARG.inputTifFiles,
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

