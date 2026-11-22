#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of denToZarr.py, tiffImagesToZarr.py, zarrToDen.py and rawTiffStructureToZarr.py: run them as
commands on synthetic data and compare the output with the input. Needs numpy, zarr, PIL, h5py and
denpy (with ZARPYRAMID).

Run: python test_toZarr.py   (or python -m unittest test_toZarr)
"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import zarr
from PIL import Image
from denpy import DEN

HERE = os.path.dirname(os.path.abspath(__file__))
DEN_TO_ZARR = os.path.join(HERE, "denToZarr.py")
TIFF_TO_ZARR = os.path.join(HERE, "tiffImagesToZarr.py")
ZARR_TO_DEN = os.path.join(HERE, "zarrToDen.py")
RAW_TIFF_TO_ZARR = os.path.join(HERE, "rawTiffStructureToZarr.py")
RECONSTRUCT_H5 = os.path.join(HERE, "reconstructBeamtimeH5.py")
sys.path.insert(0, HERE)
import make_petra_test_data  # noqa: E402
from imagecodecs.zarr import register_codecs as register_zarr_codecs  # noqa: E402
from imagecodecs.numcodecs import register_codecs as register_numcodecs_codecs  # noqa: E402

# Readers of arrays written with image codecs (htj2k, jpeg2k, ...) must register the codecs first
register_zarr_codecs()
register_numcodecs_codecs()


def run(script, *args, cwd):
	out = subprocess.run([sys.executable, script, *args], cwd=cwd, capture_output=True, text=True)
	if out.returncode != 0:
		raise AssertionError("%s %s failed:\n%s\n%s" % (os.path.basename(script), " ".join(args), out.stdout[-2000:], out.stderr[-3000:]))
	return out.stdout


def open_group(path):
	if path.endswith(".zip"):
		return zarr.open_group(zarr.storage.ZipStore(path, mode="r"), mode="r")
	return zarr.open_group(path, mode="r")


class TestToZarr(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.tmp = tempfile.TemporaryDirectory()
		d = cls.d = cls.tmp.name
		rng = np.random.default_rng(0)
		cls.vol = rng.random((5, 40, 60)).astype(np.float32)
		cls.vol4 = rng.random((2, 3, 20, 30)).astype(np.float32)
		DEN.storeNdarrayAsDEN(os.path.join(d, "vol.den"), cls.vol, force=True)
		DEN.storeNdarrayAsDEN(os.path.join(d, "vol4.den"), cls.vol4, force=True)
		cls.tif = (rng.random((4, 48, 64)) * 60000).astype(np.uint16)
		cls.tif_files = []
		for k, frame in enumerate(cls.tif):
			path = os.path.join(d, "f%03d.tif" % k)
			Image.fromarray(frame).save(path)
			cls.tif_files.append(path)
		with open(os.path.join(d, "params"), "w") as f:
			f.write("pixel_size_x=1.5\npixel_size_y=2.0\n")
		with open(os.path.join(d, "matlab.log"), "w") as f:
			f.write("par.eff_pixel_size = 3.2 micron\npar.eff_pixel_size_binned = 0 micron\n")

	@classmethod
	def tearDownClass(cls):
		cls.tmp.cleanup()

	def out(self, name):
		return os.path.join(self.d, name)

	def test_plain_compressions(self):
		for v3 in (False, True):
			names = ["none", "zstd", "lz4", "gzip", "blosc", "blosc-lz4", "blosc-zlib", "blosc-zstd"]
			if v3:
				names += ["blosc-lz4hc"]
			for name in names:
				with self.subTest(compression=name, zarrv3=v3):
					target = self.out("plain_%s_%d.zarr" % (name, v3))
					run(DEN_TO_ZARR, "--force", "-j", "0", "--compression", name, *(["--zarrv3"] if v3 else []),
						"vol.den", target, cwd=self.d)
					np.testing.assert_array_equal(zarr.open_array(target, mode="r")[:], self.vol)

	def test_4d_den(self):
		for threads in ("0", "3"):
			with self.subTest(threads=threads):
				target = self.out("vol4_%s.zarr" % threads)
				run(DEN_TO_ZARR, "--force", "--zarrv3", "-j", threads, "vol4.den", target, cwd=self.d)
				np.testing.assert_array_equal(zarr.open_array(target, mode="r")[:], self.vol4)

	def test_ome(self):
		cases = (
			(DEN_TO_ZARR, ["vol.den"], self.vol, "den_ome_v3.zarr", ["--zarrv3"], (5, 20, 30)),
			(DEN_TO_ZARR, ["vol.den"], self.vol, "den_ome_v2_3d.zarr", ["--pyramid-axes", "zyx"], (2, 20, 30)),
			(DEN_TO_ZARR, ["vol4.den"], self.vol4.reshape(6, 20, 30), "den4_ome.zarr", ["--zarrv3"], (6, 10, 15)),
			(TIFF_TO_ZARR, self.tif_files, self.tif, "tif_ome.zarr", ["--zarrv3", "--pyramid-method", "max"], (4, 24, 32)),
			(TIFF_TO_ZARR, self.tif_files, self.tif, "tif_ome.zip", ["--zarrv3", "--pyramid-levels", "2"], (4, 24, 32)),
		)
		for script, inputs, data, target, extra, level1 in cases:
			with self.subTest(target=target):
				run(script, "--force", "-j", "0", "--ome", *extra, *inputs, self.out(target), cwd=self.d)
				group = open_group(self.out(target))
				np.testing.assert_array_equal(group["0"][:], data)
				self.assertEqual(group["1"].shape, level1)
				attrs = group.attrs["ome"] if "--zarrv3" in extra else group.attrs
				self.assertIn("multiscales", attrs)

	def test_neuroglancer(self):
		for args, resolution in ((["--params", "params"], [2.0, 1.5, 1.5]), (["--matlab-log", "matlab.log"], [3.2, 3.2, 3.2])):
			with self.subTest(args=args):
				target = self.out("ng_%s.zarr" % args[0].strip("-"))
				run(DEN_TO_ZARR, "--force", "-j", "0", "--neuroglancer", "--max-level", "2", "--zarrv3", *args,
					"vol.den", target, cwd=self.d)
				group = open_group(target)
				np.testing.assert_array_equal(group["s0"][:], self.vol)
				self.assertEqual(group.attrs["resolution"], resolution)
		target = self.out("ng_tif.zarr")
		run(TIFF_TO_ZARR, "--force", "-j", "0", "--neuroglancer", "--max-level", "2", "--zarrv3", *self.tif_files, target, cwd=self.d)
		np.testing.assert_array_equal(open_group(target)["s0"][:], self.tif)


class TestZarrToDen(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.tmp = tempfile.TemporaryDirectory()
		cls.d = cls.tmp.name
		rng = np.random.default_rng(1)
		cls.vol = (rng.random((4, 24, 36)) * 4000).astype(np.uint16)
		cls.vol4 = rng.random((2, 3, 10, 12)).astype(np.float32)
		DEN.storeNdarrayAsDEN(os.path.join(cls.d, "vol.den"), cls.vol, force=True)
		DEN.storeNdarrayAsDEN(os.path.join(cls.d, "vol4.den"), cls.vol4, force=True)

	@classmethod
	def tearDownClass(cls):
		cls.tmp.cleanup()

	def test_den_zarr_den(self):
		for name, compression, threads in (("a.zarr", "blosc-zstd", "0"), ("b.zarr", "lz4", "3"), ("c.zip", "htj2k", "2")):
			with self.subTest(store=name, compression=compression):
				store = os.path.join(self.d, name)
				run(ZARR_TO_DEN, "-j", threads, "--keep-input-dtype", "--zarr-compression", compression, "--zarr-clevel", "0",
					"vol.den", store + ":/vol", cwd=self.d)
				back = os.path.join(self.d, name + ".den")
				run(ZARR_TO_DEN, "-j", threads, "--keep-input-dtype", store + ":/vol", back, cwd=self.d)
				np.testing.assert_array_equal(DEN.getNumpyArray(back), self.vol)

	def test_4d_den(self):
		store = os.path.join(self.d, "four.zarr")
		run(ZARR_TO_DEN, "-j", "0", "vol4.den", store + ":/v", cwd=self.d)
		np.testing.assert_array_equal(zarr.open_array(store, path="v", mode="r")[:], self.vol4.reshape(6, 10, 12))
		run(ZARR_TO_DEN, "-j", "2", "vol4.den", os.path.join(self.d, "copy4.den"), cwd=self.d)
		np.testing.assert_array_equal(DEN.getNumpyArray(os.path.join(self.d, "copy4.den")), self.vol4)

	def test_failure_exit_status(self):
		bad = os.path.join(self.d, "bad.zarr")
		zarr.create_array(store=bad, shape=(2, 4, 4), chunks=(1, 4, 4), dtype=np.float32, zarr_format=3)
		os.makedirs(os.path.join(bad, "c", "0", "0"), exist_ok=True)
		with open(os.path.join(bad, "c", "0", "0", "0"), "w") as f:  # a broken chunk of frame 0
			f.write("garbage")
		out = subprocess.run([sys.executable, ZARR_TO_DEN, "-j", "0", bad + ":/", os.path.join(self.d, "bad.den")],
							 cwd=self.d, capture_output=True, text=True)
		self.assertEqual(out.returncode, 1, out.stderr[-2000:])


def h5_differences(h5file, group):
	"""Differences between the HDF5 entry and the params group of a rawTiffStructureToZarr.py output."""
	import h5py

	def norm(v):  # as sanitize_for_json: arrays to lists, bytes to str, 1-element lists to their value
		if isinstance(v, np.ndarray):
			v = v.tolist()
		if isinstance(v, np.generic):
			v = v.item()
		if isinstance(v, bytes):
			v = v.decode()
		if isinstance(v, list):
			v = [norm(x) for x in v]
			return v[0] if len(v) == 1 else v
		return v

	problems = []

	def walk(h, z):
		for k, v in h.attrs.items():
			if z.attrs.get(k, "<missing>") != norm(v):
				problems.append(f"attribute {h.name}@{k}: {norm(v)!r} != {z.attrs.get(k, '<missing>')!r}")
		for name, obj in h.items():
			if isinstance(obj, h5py.Group):
				if name in z:
					walk(obj, z[name])
				else:
					problems.append(f"missing group {obj.name}")
			elif obj.shape == ():  # scalar datasets are attributes of the group
				if z.attrs.get(name, "<missing>") != norm(obj[()]):
					problems.append(f"scalar {obj.name}: {norm(obj[()])!r} != {z.attrs.get(name, '<missing>')!r}")
			elif name not in z:
				problems.append(f"missing dataset {obj.name}")
			else:
				a, b = obj[()], z[name][()]
				if a.dtype.kind in "OS":
					same = a.shape == b.shape and [norm(x) for x in a.ravel()] == [str(x) for x in b.ravel()]
				else:
					same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
				if not same:
					problems.append(f"dataset {obj.name}: {a.dtype}{a.shape} != {b.dtype}{b.shape}")

	with h5py.File(h5file, "r") as h5:
		walk(h5["entry"], group["params"])
	return problems


def h5_reconstruction_differences(source, reconstruction, layout):
	"""Differences between /entry of two HDF5 files: the values (strings compared as text, 1-element
	arrays as their element) and, with layout, also the attribute types, dtypes, string encodings,
	shapes, chunks and maximum shapes."""
	import h5py

	def norm(v):
		if isinstance(v, np.ndarray):
			v = v.tolist()
		if isinstance(v, bytes):
			v = v.decode()
		if isinstance(v, list):
			v = [norm(x) for x in v]
			return v[0] if len(v) == 1 else v
		return v.item() if isinstance(v, np.generic) else v

	def describe(o):
		if isinstance(o, h5py.Group):
			return {k: (type(v).__name__, norm(v)) for k, v in o.attrs.items()}
		string = h5py.check_string_dtype(o.dtype)
		return (o.shape, string.encoding if string else str(o.dtype), o.chunks, o.maxshape, o.compression)

	problems = []
	with h5py.File(source, "r") as a, h5py.File(reconstruction, "r") as b:
		names_a, names_b = set(), set()
		a["entry"].visit(names_a.add)
		b["entry"].visit(names_b.add)
		problems += [f"only in {'source' if n in names_a else 'reconstruction'}: {n}" for n in sorted(names_a ^ names_b)]
		for name in sorted(names_a & names_b):
			oa, ob = a["entry"][name], b["entry"][name]
			if layout and describe(oa) != describe(ob):
				problems.append(f"{name}: {describe(oa)} != {describe(ob)}")
			elif isinstance(oa, h5py.Group):
				if {k: norm(v) for k, v in oa.attrs.items()} != {k: norm(v) for k, v in ob.attrs.items()}:
					problems.append(f"{name}: attributes differ")
			elif oa.dtype.kind == "O":
				if norm(oa[()]) != norm(ob[()]):
					problems.append(f"{name}: strings differ")
			elif oa.dtype != ob.dtype or not np.array_equal(oa[()], ob[()]):
				problems.append(f"{name}: values differ")
	return problems


class TestRawTiffStructureToZarr(unittest.TestCase):
	"""On synthetic scans in the HDF5 styles of 2021 and 2025 P05 files (make_petra_test_data.py)."""

	@classmethod
	def setUpClass(cls):
		cls.tmp = tempfile.TemporaryDirectory()
		cls.scans = {style: (os.path.join(cls.tmp.name, style), *make_petra_test_data.make(os.path.join(cls.tmp.name, style), style))
					 for style in make_petra_test_data.STYLES}

	@classmethod
	def tearDownClass(cls):
		cls.tmp.cleanup()

	def check(self, style, target, *args):
		d, keys, frames = self.scans[style]
		out = subprocess.run([sys.executable, "-W", "always", "-W", "ignore::DeprecationWarning:tifffile.tifffile", RAW_TIFF_TO_ZARR, "scan.h5", target, "--force", *args],
							 cwd=d, capture_output=True, text=True)
		self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-3000:])
		self.assertNotIn("Warning", out.stderr, out.stderr[-3000:])  # e.g. zipfile "Duplicate name"
		group = open_group(os.path.join(d, target))
		keys = np.array(keys)
		for name, key in (("img", 0), ("ref", 1), ("dar", 2)):
			np.testing.assert_array_equal(group[name][:], frames[keys == key])
		self.assertEqual(h5_differences(os.path.join(d, "scan.h5"), group), [])
		if style == "2021":
			self.assertEqual(group["params/hardware/camera"].attrs["model"], "test camera")
		return group

	def test_directory_and_codecs(self):
		for style in self.scans:
			with self.subTest(style=style):
				self.check(style, "plain.zarr", "-j", "0", "--verbose")
				self.check(style, "htj2k.zarr", "-j", "2", "--compression", "htj2k", "--clevel", "0")
				self.check(style, "v2.zarr", "-j", "2", "--v2")

	def test_zip_entries_once(self):
		import zipfile
		for style in self.scans:
			with self.subTest(style=style):
				self.check(style, "scan.zip", "--zip", "-j", "2")
				names = zipfile.ZipFile(os.path.join(self.scans[style][0], "scan.zip")).namelist()
				self.assertEqual(len(names), len(set(names)), [n for n in set(names) if names.count(n) > 1])

	def test_reconstruct_h5(self):
		# the HDF5 file back from the Zarr store, in the 2025 layout: identical to a 2025 file,
		# the same values for a 2021 file (camera@model, not NX_class, becomes a scalar dataset)
		for style in self.scans:
			for target, args in (("rec.zip", ["--zip"]), ("rec_v2.zarr", ["--v2"])):
				with self.subTest(style=style, store=target):
					d = self.scans[style][0]
					self.check(style, target, "-j", "2", *args)
					run(RECONSTRUCT_H5, target, "rec.h5", "--force", cwd=d)
					problems = h5_reconstruction_differences(os.path.join(d, "scan.h5"), os.path.join(d, "rec.h5"), layout=style == "2025")
					if style == "2021":
						self.assertEqual(problems, ["only in reconstruction: hardware/camera/model", "hardware/camera: attributes differ"])
					else:
						self.assertEqual(problems, [])

	def test_fix_corrupted_h5(self):
		import h5py
		d = os.path.join(self.tmp.name, "corrupt")
		keys, frames = make_petra_test_data.make(d)
		with h5py.File(os.path.join(d, "scan.h5"), "a") as h5:  # the images and the last refs are missing from the HDF5
			data = h5["entry/scan/data"]
			for name in ("image_key", "image_file", "s_rot"):
				for part in ("value", "time"):
					v = data[f"{name}/{part}"][:5]
					del data[f"{name}/{part}"]
					data.create_dataset(f"{name}/{part}", data=v)
		run(RAW_TIFF_TO_ZARR, "scan.h5", "out.zarr", "--fix-corrupted-h5", "-j", "0", cwd=d)
		# img: all TIFFs that are neither the darks nor the refs listed in the HDF5
		np.testing.assert_array_equal(zarr.open_group(os.path.join(d, "out.zarr"), mode="r")["img"][:], frames[5:])


if __name__ == "__main__":
	unittest.main()
