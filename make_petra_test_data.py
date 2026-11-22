"""Create a small synthetic PETRA P05 scan (NeXus HDF5 + TIFFs) for rawTiffStructureToZarr.py tests.

Usage: python make_petra_test_data.py DIR [2021|2025]
Layout: DIR/scan.h5 and the TIFFs named in it (relative to DIR); 2 darks (key 2), 3 refs (key 1),
4 images (key 0), refs again at the end. The HDF5 follows the two styles of real P05 files:

  2021 (as exampleA.h5): root and group attributes are 1-element object (vlen str) arrays,
        strings are (1,) vlen str datasets (e.g. entry/beamline/name), datasets in chunks of
        1024; one attribute other than NX_class (hardware/camera@model)
  2025 (as exampleB.h5): NX_class is the only attribute, a plain str; strings are scalar vlen
        ASCII bytes datasets; some logs and values are empty (0,) datasets; datasets in chunks of 1

Common to both: NXlog groups with uint64 time (ms) and float value, (1,) float32/float64 values,
int16 image_key, vlen image_file, motor logs (s_rot, ...) shorter than the image list,
scan/setup values.
"""
import os
import sys

import h5py
import numpy as np
from PIL import Image

KEYS = [2, 2, 1, 1, 1, 0, 0, 0, 0, 1]
STYLES = ("2021", "2025")


def make(d, style="2021", dimy=32, dimx=48, seed=0, keys=KEYS, prefix="syn_test_000"):
	"""Write the scan to directory d; returns (image keys, frames as array [n, dimy, dimx])."""
	if style not in STYLES:
		raise ValueError(f"style must be one of {STYLES}")
	old = style == "2021"
	os.makedirs(d, exist_ok=True)
	rng = np.random.default_rng(seed)
	suffix = {0: "img", 1: "ref", 2: "dar"}
	files, frames = [], []
	for i, k in enumerate(keys):
		name = "%s_%06d_%s.tif" % (prefix, i, suffix[k])
		frame = (rng.random((dimy, dimx)) * 4000).astype(np.uint16)
		Image.fromarray(frame).save(os.path.join(d, name))
		files.append(name)
		frames.append(frame)
	t0 = 1_622_724_176_775 if old else 1_758_209_568_385
	times = t0 + np.arange(len(keys), dtype=np.uint64) * 34
	str_dtype = h5py.string_dtype("utf-8" if old else "ascii")

	def attr(obj, name, value):
		# 2021: 1-element object array of vlen str; 2025: plain str
		if old and isinstance(value, str):
			obj.attrs.create(name, np.array([value], dtype=object), dtype=h5py.string_dtype())
		else:
			obj.attrs[name] = value

	def dataset(parent, name, data, dtype=None):
		# as the beamline writes them: chunked, unlimited first dimension (2021: chunks of 1024, 2025: of 1)
		data = np.asarray(data) if dtype is None else data
		if np.ndim(data) == 0:
			return parent.create_dataset(name, data=data, dtype=dtype)
		return parent.create_dataset(name, data=data, dtype=dtype, maxshape=(None,), chunks=(1024,) if old else (1,))

	def group(parent, path, nx_class="NXcollection"):
		g = parent.require_group(path)
		attr(g, "NX_class", nx_class)
		return g

	def text(parent, name, value):
		if old:
			dataset(parent, name, np.array([value], dtype=object), dtype=str_dtype)
		else:
			dataset(parent, name, value.encode(), dtype=h5py.string_dtype("ascii"))

	def nxlog(parent, path, n, value_dtype=np.float64, empty=False):
		log = group(parent, path, "NXlog")
		n = 0 if empty else n
		dataset(log, "time", t0 + np.arange(n, dtype=np.uint64) * 200)
		dataset(log, "value", rng.random(n).astype(value_dtype))

	def values(parent, names, dtype=np.float64, empty=()):
		for name in names:
			dataset(parent, name, np.array([] if name in empty else [rng.random() * 100], dtype=dtype))

	with h5py.File(os.path.join(d, "scan.h5"), "w") as h5:
		if old:
			for name, value in (("HDF5_version", "1.10.0"), ("NX_class", "NXroot"), ("NeXus_version", "4.3.0"),
								("file_name", f"/gpfs/current/raw/{prefix}/{prefix}_nexus.h5"),
								("file_time", "2021-06-03T14:42:55.359597+0200")):
				attr(h5, name, value)
		entry = group(h5, "entry", "NXentry")
		beamline = group(entry, "beamline")
		text(beamline, "experiment", "microtomography")
		text(beamline, "name", "P05 IBL EH2")

		hardware = group(entry, "hardware")
		nxlog(hardware, "adc1", 40)
		nxlog(hardware, "beam_current/current", 30)
		group(hardware, "beam_current")
		for axis in ("rotx", "roty", "x"):
			group(hardware, f"base_stage/{axis}")
			nxlog(hardware, f"base_stage/{axis}/position", 5)
		group(hardware, "base_stage")
		for axis in ("rotx", "y"):
			group(hardware, f"camera_stage/{axis}")
			nxlog(hardware, f"camera_stage/{axis}/position", 4, empty=not old)
		group(hardware, "camera_stage")
		camera = group(hardware, "camera")
		values(camera, ("bin", "exptime", "magnification", "pixelsize", "roi_height", "roi_width"))
		if old:
			attr(camera, "model", "test camera")  # an attribute other than NX_class
		dcm = group(hardware, "dcm")
		values(dcm, ("energy",))
		group(dcm, "bragg")
		nxlog(dcm, "bragg/position", 6)
		diode = group(hardware, "diode/eh2")
		group(hardware, "diode")
		values(diode, ("amp_factor", "counter", "photon_flux"), empty=() if old else ("amp_factor", "counter", "photon_flux"))
		loadcell = group(hardware, "loadcell")
		values(loadcell, ("calibration_factor",), dtype=np.float32, empty=() if old else ("calibration_factor",))
		group(hardware, "microscope/aperture")
		nxlog(hardware, "microscope/aperture/position", 6, value_dtype=np.float32)

		scan = group(entry, "scan")
		for name, value in (("mode", 180.0), ("n_angle", 4.0), ("n_dark", 2.0), ("n_img", 1.0), ("n_ref", 3.0), ("ref_count", 0.0)):
			dataset(scan, name, np.array([value]))
		setup = group(scan, "setup")
		values(setup, ("pos_o_ccd_dist", "pos_s_stage_x", "pos_s_stage_z"))
		values(setup, ("pos_p05_energy",), dtype=np.float32)
		data = group(scan, "data")
		for name in ("image_key", "image_file", "s_rot", "s_stage_x", "s_stage_z"):
			group(data, name, "NXlog")
		dataset(data, "image_key/value", np.array(keys, dtype=np.int16))
		dataset(data, "image_key/time", times)
		dataset(data, "image_file/value", np.array([f.encode() for f in files], dtype=object),
				dtype=h5py.string_dtype("utf-8" if old else "ascii"))
		dataset(data, "image_file/time", times)
		# motor logs only for the non-dark frames
		moving = np.array([k != 2 for k in keys])
		for motor, dtype in (("s_rot", np.float32), ("s_stage_x", np.float32), ("s_stage_z", np.float32)):
			dataset(data, f"{motor}/value", np.linspace(-5, 180, int(moving.sum())).astype(dtype))
			dataset(data, f"{motor}/time", times[moving])
	return list(keys), np.stack(frames)


if __name__ == "__main__":
	make(sys.argv[1], *(sys.argv[2:3] or []))
