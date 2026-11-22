#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of denToZarr.py and tiffImagesToZarr.py: run them as commands on synthetic data and compare the
Zarr output with the input. Needs numpy, zarr, PIL and denpy (with ZARPYRAMID).

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


if __name__ == "__main__":
	unittest.main()
