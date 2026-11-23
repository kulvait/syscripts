#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct the beamtime HDF5 (NeXus) file of a scan from the Zarr store written by
rawTiffStructureToZarr.py or tiffScanDataToZarrForMicrotomography: the params group of the store
becomes /entry of the HDF5 file.

The store is read into lautools PETRA.ScanDescription, the canonical form of the scan metadata, and
written with ScanDescription.writeH5 in the newer P05 HDF5 layout (2025 files), whatever layout the
source file had:
  - group attributes: NX_class only, as plain strings;
  - other group attributes (scalar datasets of the source file and single strings) become scalar
    datasets: strings as variable-length ASCII bytes (UTF-8 if not ASCII), numbers as numbers;
  - arrays: same dtype and values, chunked with unlimited first dimension and no compression,
    as the beamline writes them (chunks of 1 element by default, see --chunk);
  - string arrays: variable-length ASCII bytes.
The scan table (params attributes columns/index/data) and the attributes of the top level group
(experiment info, export settings) are not part of the HDF5 file and are not written. The images are
not written; they are the dar/ref/img arrays of the store.

Usage: zarrToBeamtimeH5.py INPUT.zarr|INPUT.zip|INPUT.zar OUTPUT.h5 [--force]
"""
import argparse
import os
import sys

from lautools import PETRA

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("inputZarr", help="Zarr store with a params group: directory or zip file")
parser.add_argument("outputH5", help="HDF5 file to write")
parser.add_argument("--force", action="store_true", help="Overwrite the output file")
parser.add_argument("--chunk", type=int, default=1, help="Chunk length of the datasets (default 1, as the beamline files)")
parser.add_argument("--verbose", action="store_true")


def main():
	ARG = parser.parse_args()
	if os.path.exists(ARG.outputH5) and not ARG.force:
		parser.error(f"{ARG.outputH5} exists, use --force to overwrite")
	try:  # image codecs of the dar/ref/img arrays, in case the store lists them
		from imagecodecs.zarr import register_codecs
		register_codecs()
	except ImportError:
		pass
	import zarr
	location = os.path.realpath(ARG.inputZarr)
	if os.path.isfile(location):
		store = zarr.storage.ZipStore(location, mode="r")
	else:
		store = location
	try:
		root = zarr.open_group(store, mode="r")
		if "params" not in root:
			sys.exit(f"ERROR: {ARG.inputZarr} has no params group; is it written by rawTiffStructureToZarr.py?")
		description = PETRA.ScanDescriptionFactory.fromZarr(root, "params")
	finally:
		if not isinstance(store, str):
			store.close()
	description.writeH5(ARG.outputH5, chunk=ARG.chunk)
	if ARG.verbose:
		print(f"Read {location}: {len(description.root.groups)} groups below params")
	print(f"Written {ARG.outputH5}")


if __name__ == "__main__":
	main()
