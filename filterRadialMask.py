#!/usr/bin/env python

import argparse
from denpy import DEN
import numpy as np

# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Apply radial mask and optional symmetric crop.")

parser.add_argument("inputDen", help="Input DEN file")
parser.add_argument("outputDen", help="Output DEN file")

parser.add_argument("--radial-mask-radius", type=float, required=True,
                    help="Radius of radial mask in pixels")

#Fine tune specification of radial center
gx = parser.add_mutually_exclusive_group()
gx.add_argument("--radial-center-offset-x", type=float, default=None, help="Optional X offset of radial center from image center (default: 0)")
gx.add_argument("--radial-center-x", type=float, default=None, help="Optional X coordinate of radial center (default: image center)")

gy = parser.add_mutually_exclusive_group()
gy.add_argument("--radial-center-offset-y", type=float, default=None, help="Optional Y offset of radial center from image center (default: 0)")
gy.add_argument("--radial-center-y", type=float, default=None, help="Optional Y coordinate of radial center (default: image center)")



parser.add_argument("--out-dimx", type=int, default=None,
                    help="Optional output dimension X (crop symmetrically)")
parser.add_argument("--out-dimy", type=int, default=None,
                    help="Optional output dimension Y (crop symmetrically)")
parser.add_argument("--out-dimz", type=int, default=None,
                    help="Optional output dimension Z (crop symmetrically)")

parser.add_argument("--force", action="store_true", help="Overwrite existing output file")

ARG = parser.parse_args()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def create_radial_mask(h, w, radius, center_x=None, center_y=None):
    """Return a boolean mask (h,w) with True inside a circle."""
    if center_x is None:
        center_x = w / 2
    if center_y is None:
        center_y = h / 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return dist <= radius


def crop_center(arr, out_x, out_y):
    """Crop a 2D array symmetrically around center."""
    h, w = arr.shape
    cx, cy = w // 2, h // 2

    x1 = cx - out_x // 2
    x2 = x1 + out_x
    y1 = cy - out_y // 2
    y2 = y1 + out_y

    return arr[y1:y2, x1:x2]


def filterFrame(frame, out_x, out_y):
    """Only applies symmetric center crop (mask already applied)."""
    h, w = frame.shape

    if out_x == w and out_y == h:
        return frame  # no crop needed

    return crop_center(frame, out_x, out_y)


# ---------------------------------------------------------
# Load header and validate dimensions
# ---------------------------------------------------------
header = DEN.readHeader(ARG.inputDen)

if header["dimcount"] == 2:
    zdim = 1
elif header["dimcount"] == 3:
    zdim = header["dimspec"][2]
else:
    raise ValueError("Input DEN file must be 2D or 3D.")

xdim, ydim = header["dimspec"][:2]

# Determine output dims
out_dimx = ARG.out_dimx if ARG.out_dimx is not None else xdim
out_dimy = ARG.out_dimy if ARG.out_dimy is not None else ydim
out_dimz = ARG.out_dimz if ARG.out_dimz is not None else zdim

if out_dimx > xdim:
    raise ValueError("Output dimension X cannot exceed input.")
if out_dimy > ydim:
    raise ValueError("Output dimension Y cannot exceed input.")
if out_dimz > zdim:
    raise ValueError("Output dimension Z cannot exceed input.")

# ---------------------------------------------------------
# Precompute radial mask here (only once!)
# ---------------------------------------------------------
if ARG.radial_center_x is not None:
    center_x = ARG.radial_center_x
elif ARG.radial_center_offset_x is not None:
    center_x = xdim / 2 + ARG.radial_center_offset_x
else:
    center_x = xdim / 2

if ARG.radial_center_y is not None:
    center_y = ARG.radial_center_y
elif ARG.radial_center_offset_y is not None:
    center_y = ydim / 2 + ARG.radial_center_offset_y
else:
    center_y = ydim / 2
radial_mask = create_radial_mask(ydim, xdim, ARG.radial_mask_radius, center_x=center_x, center_y=center_y)

# ---------------------------------------------------------
# Prepare output file
# ---------------------------------------------------------
DEN.writeEmptyDEN(
    ARG.outputDen,
    [out_dimx, out_dimy] + ([out_dimz] if header["dimcount"] == 3 else []),
    force=ARG.force
)

# ---------------------------------------------------------
# Processing loop
# ---------------------------------------------------------
z_start = (zdim - out_dimz) // 2
for k in range(out_dimz):
    print(f"Processing slice {k+1} / {out_dimz}")
    # Load frame
    frame = DEN.getFrame(ARG.inputDen, z_start + k)
    # Apply mask now (mask already precomputed)
    masked = frame * radial_mask
    # Then crop
    output_frame = filterFrame(masked, out_dimx, out_dimy)
    # Save
    DEN.writeFrame(ARG.outputDen, k, output_frame, force=True)
