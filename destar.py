#!/usr/bin/env python
"""
2023-2026

@author: Vojtech Kulvait
@license: GNU GPL v3

This script removes transient bright pixels (e.g. hot pixels or cosmic ray hits) 
from a 3D image stack (time or depth series) stored in the DEN format. It works 
by applying an iterative median filter to each frame, identifying outlier pixels 
that deviate from their local neighborhood by more than a specified number of 
standard deviations, and replacing them with the corresponding median value.

Key features:
- Configurable neighborhood size, threshold, and number of iterations.
- Per-frame processing using multiprocessing for performance.
- Optional verbose output showing the percentage of corrected pixels per frame.
- At the end, prints total number and percentage of pixels corrected across the dataset.

This method is useful when a dark frame is insufficient to remove non-repeatable 
artifacts such as cosmic rays or temporally unstable hot pixels.

For alternative method, which filter just bright spots based on average of nearby pixles, see Algotom remove_zinger method in https://algotom.readthedocs.io/en/latest/toc/api/algotom.prep.removal.html#algotom.prep.removal.check_zinger_size
Note that Algotom method works probably better on sinogram than projection frames.
"""

import sys
from lautools.scripts import removeHotPixels

def main():
    # Pass all command-line arguments to removeHotPixels main
    sys.exit(removeHotPixels.main())

if __name__ == "__main__":
    main()
