#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct the beamtime HDF5 (NeXus) file of a scan from the Zarr store written by
rawTiffStructureToZarr.py: the params group of the store becomes /entry of the HDF5 file.

The output follows the newer P05 HDF5 layout (2025 files), whatever layout the source file had:
  - group attributes: NX_class only, as plain strings;
  - other group attributes of the Zarr store (scalar datasets of the source file, stored as
    attributes by rawTiffStructureToZarr.py) become scalar datasets: strings as variable-length
    ASCII bytes (UTF-8 if not ASCII), numbers as numbers;
  - arrays: same dtype and values, chunked with unlimited first dimension and no compression,
    as the beamline writes them (chunks of 1 element by default, see --chunk);
  - string arrays: variable-length ASCII bytes; a 1-element string array outside an NXlog group
    (older files, e.g. entry/beamline/name) becomes a scalar string dataset.
The scan table of rawTiffStructureToZarr.py (params attributes columns/index/data) and the
attributes of the top level group (experiment info, export settings) are not part of the HDF5
file and are not written. The images are not written; they are the dar/ref/img arrays of the store.

Usage: reconstructBeamtimeH5.py INPUT.zarr|INPUT.zip OUTPUT.h5 [--force]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import zarr

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("inputZarr", help="Zarr store written by rawTiffStructureToZarr.py: directory or zip file")
parser.add_argument("outputH5", help="HDF5 file to write")
parser.add_argument("--force", action="store_true", help="Overwrite the output file")
parser.add_argument("--chunk", type=int, default=1, help="Chunk length of the datasets (default 1, as the beamline files)")
parser.add_argument("--verbose", action="store_true")

ATTRIBUTES = ("NX_class",)  # group attributes that stay attributes, everything else is a scalar dataset
SCAN_TABLE_KEYS = ("columns", "index", "data")  # pandas orient="split" table on the params group


def open_store(path):
	if os.path.isfile(path):
		return zarr.storage.ZipStore(path, mode="r")
	return path


def is_string_dtype(dtype):
	return dtype.kind in "OSUT"


def string_dtype(values):
	"""Variable-length ASCII bytes as the beamline writes them, UTF-8 if a value is not ASCII."""
	return h5py.string_dtype("ascii" if all(v.isascii() for v in values) else "utf-8")


def encode(values, dtype):
	return [v.encode(h5py.check_string_dtype(dtype).encoding) for v in values]


def write_scalar(group, name, value, verbose=False):
	"""A group attribute of the Zarr store as a scalar dataset (lists as 1-D datasets)."""
	if isinstance(value, str):
		dtype = string_dtype([value])
		group.create_dataset(name, data=encode([value], dtype)[0], dtype=dtype)
	elif isinstance(value, (bool, int, float)):
		group.create_dataset(name, data=np.array(value))
	elif isinstance(value, list) and all(isinstance(v, str) for v in value):
		dtype = string_dtype(value)
		group.create_dataset(name, data=np.array(encode(value, dtype), dtype=object), dtype=dtype)
	elif isinstance(value, list) and all(isinstance(v, (bool, int, float)) for v in value):
		group.create_dataset(name, data=np.array(value))
	else:
		# None, dictionaries and mixed lists have no HDF5 dataset form; keep them as JSON text
		text = json.dumps(value)
		print(f"WARNING: {group.name}/{name} is stored as JSON text: {text[:100]}", file=sys.stderr)
		group.create_dataset(name, data=text.encode(), dtype=string_dtype([text]))
	if verbose:
		print(f"[scalar] {group.name}/{name}")


def write_array(group, name, array, nxlog, chunk, verbose=False):
	data = array[...]
	if is_string_dtype(data.dtype):
		values = [str(v) for v in np.asarray(data).ravel()]
		dtype = string_dtype(values)
		encoded = np.array(encode(values, dtype), dtype=object).reshape(data.shape)
		if data.shape == (1,) and not nxlog:
			# older files: a single string as a (1,) dataset; the newer layout has a scalar
			group.create_dataset(name, data=encoded[0], dtype=dtype)
			return
		data = encoded
	else:
		data = np.asarray(data)
		dtype = data.dtype
	if data.ndim == 0:
		group.create_dataset(name, data=data, dtype=dtype)
	else:
		group.create_dataset(name, data=data, dtype=dtype, maxshape=(None,) + data.shape[1:],
							 chunks=(chunk,) + tuple(max(1, n) for n in data.shape[1:]))
	if verbose:
		print(f"[dataset] {group.name}/{name} {data.shape} {dtype}")


def write_group(zgroup, hgroup, chunk, top=False, verbose=False):
	attributes = dict(zgroup.attrs)
	if top:
		for key in SCAN_TABLE_KEYS:
			attributes.pop(key, None)
	for key, value in attributes.items():
		if key in ATTRIBUTES:
			hgroup.attrs[key] = value if isinstance(value, str) else json.dumps(value)
		else:
			write_scalar(hgroup, key, value, verbose)
	nxlog = attributes.get("NX_class") == "NXlog"
	for name, member in sorted(zgroup.members()):
		if isinstance(member, zarr.Group):
			write_group(member, hgroup.create_group(name), chunk, verbose=verbose)
		else:
			write_array(hgroup, name, member, nxlog, chunk, verbose)


def main():
	ARG = parser.parse_args()
	if os.path.exists(ARG.outputH5) and not ARG.force:
		parser.error(f"{ARG.outputH5} exists, use --force to overwrite")
	try:  # image codecs of the dar/ref/img arrays, in case the store lists them
		from imagecodecs.zarr import register_codecs
		register_codecs()
	except ImportError:
		pass
	store = open_store(ARG.inputZarr)
	root = zarr.open_group(store, mode="r")
	if "params" not in root:
		sys.exit(f"ERROR: {ARG.inputZarr} has no params group; is it written by rawTiffStructureToZarr.py?")
	with h5py.File(ARG.outputH5, "w") as h5:
		write_group(root["params"], h5.create_group("entry"), ARG.chunk, top=True, verbose=ARG.verbose)
	if isinstance(store, zarr.storage.ZipStore):
		store.close()
	print(f"Written {ARG.outputH5}")


if __name__ == "__main__":
	main()
