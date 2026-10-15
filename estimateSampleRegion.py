#!/usr/bin/env python
"""
Estimate illuminated bounds and select representative sample rows.

Created 05/2026
@author: Vojtěch Kulvait

Region x_max/y_max coordinates are inclusive.
Subregion start/end coordinates use half-open intervals [start, end).

The reconstruction-quality score is:
    sqrt(mean image intensity) / clipped mean standard deviation
    * relative reference intensity

Legacy JSON names "tra_avg" and "sd_div_img" are retained for compatibility.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from denpy import DEN

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)
log.propagate = False # Prevent log messages from being propagated to the root logger


PROGRAM_NAME = "estimateSampleRegion"


def configure_logging():
    """Configure console logging without adding duplicate handlers."""
    log.setLevel(logging.INFO)
    log.propagate = False

    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
                datefmt="%d.%m.%Y %H:%M:%S",
            )
        )
        log.addHandler(handler)


def format_elapsed(seconds):
    """Format elapsed time as HH:MM:SS.sss."""
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def build_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate illuminated bounds and select representative rows "
            "within vertical buckets."
        )
    )
    parser.add_argument(
        "ref_avg",
        type=Path,
        help="Reference-average DEN file used to estimate illumination.",
    )
    parser.add_argument(
        "img_avg",
        type=Path,
        help="Image-average DEN file used to compute row quality scores.",
    )
    parser.add_argument(
        "img_sd",
        type=Path,
        help="Image-standard-deviation DEN file used to compute row scores.",
    )
    parser.add_argument(
        "--snr",
        type=Path,
        help="Optional projection-SNR DEN file used to set bucket bounds.",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=23.0,
        help="Select SNR-profile values strictly above this threshold (default: 23).",
    )
    parser.add_argument(
        "--illum-factor",
        type=float,
        default=0.5,
        help=(
            "Select reference-profile values strictly above factor * mean "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=5,
        help="Number of nonempty vertical buckets (default: 5).",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        help="Save the plot as PDF instead of displaying it.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Save illuminated bounds and selected subregions as JSON.",
    )
    return parser


def validate_arguments(parser, args):
    """Validate arguments that do not require loading image data."""
    if args.buckets < 1:
        parser.error("--buckets must be at least 1.")

    if not math.isfinite(args.illum_factor) or args.illum_factor < 0:
        parser.error("--illum-factor must be finite and nonnegative.")

    if not math.isfinite(args.snr_threshold):
        parser.error("--snr-threshold must be finite.")

    for name in ("ref_avg", "img_avg", "img_sd", "snr"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"{name}: input file does not exist: {path}")


def load_frame(path, name, expected_shape=None, nonnegative=False):
    """Load and validate the first DEN frame as a floating-point array."""
    frame = np.asarray(DEN.getFrame(str(path), 0), dtype=np.float64)

    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"{name} must contain a nonempty two-dimensional frame.")
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {frame.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if nonnegative and np.any(frame < 0):
        raise ValueError(f"{name} contains negative values.")

    return frame


def profile_bounds(profile, threshold, description):
    """Return inclusive first/last indices exceeding the threshold.

    This preserves the original bounding-envelope behavior: internal gaps
    below the threshold are not excluded from the resulting interval.
    """
    indices = np.flatnonzero(profile > threshold)
    if indices.size < 2:
        raise ValueError(
            f"{description}: fewer than two positions exceed "
            f"the threshold {threshold:.6g}."
        )
    return int(indices[0]), int(indices[-1])


def region_geometry(left, right, top, bottom):
    """Build region geometry using inclusive image coordinates."""
    return {
        "x_min": left,
        "x_max": right,
        "y_min": top,
        "y_max": bottom,
        "illuminated_width": right - left + 1,
        "illuminated_height": bottom - top + 1,
    }


def report_region(region, description, criterion):
    """Print inclusive bounds and pixel counts; return the vertical message."""
    vertical = (
        f"Vertical illuminated region{description}: "
        f"[{region['y_min']}, {region['y_max']}], "
        f"height = {region['illuminated_height']}, {criterion}"
    )
    print(vertical, flush=True)
    print(
        f"Horizontal illuminated region{description}: "
        f"[{region['x_min']}, {region['x_max']}], "
        f"width = {region['illuminated_width']}, {criterion}",
        flush=True,
    )
    return vertical


def center_weight(n):
    """Return weights favoring the center of a nonempty bucket."""
    if n == 1:
        return np.ones(1, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, n)
    return 0.75 + 0.25 * np.cos(0.5 * np.pi * x)


def select_subregions(ref_profile, img_profile, sd_profile, score, top, bottom, count):
    """Partition every selected row into nonempty half-open buckets."""
    row_count = bottom - top + 1
    if count > row_count:
        raise ValueError(
            f"--buckets ({count}) exceeds the selected row count ({row_count})."
        )

    # bottom is inclusive; the final bucket must end at bottom + 1.
    edges = top + np.arange(count + 1, dtype=np.int64) * row_count // count
    regions = []

    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        start, end = int(start), int(end)
        weighted_score = score[start:end] * center_weight(end - start)
        max_row = start + int(np.argmax(weighted_score))

        regions.append(
            {
                "id": i,
                "start": start,
                "end": end,
                "max_row": max_row,
                "values": {
                    "ref_avg": float(ref_profile[max_row]),
                    "tra_avg": float(img_profile[max_row]),
                    # Preserve the original clipped-SD output semantics.
                    "img_sd": float(sd_profile[max_row]),
                    "sd_div_img": float(score[max_row]),
                },
            }
        )
        print(
            f"Bucket {i + 1}: [{start}, {end}), max at row {max_row} "
            f"with weighted score {weighted_score[max_row - start]:.3g}",
            flush=True,
        )

    return edges, regions


def plot_profiles(args, results, profiles, bucket_edges, title):
    """Display or save profiles, illumination boundaries, and selected rows."""
    # Keep backend selection and pyplot import out of module initialization.
    import matplotlib

    if args.out_pdf is not None or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    reference_region = results["illuminated_region"]
    snr_region = results.get("illuminated_region_snr")

    try:
        for i, (ax, (name, profile)) in enumerate(zip(axes, profiles)):
            ax.plot(profile, label="row profile")
            for key in ("y_min", "y_max"):
                ax.axvline(
                    reference_region[key],
                    color="blue",
                    linewidth=2,
                    alpha=0.8,
                )
                if snr_region is not None:
                    ax.axvline(
                        snr_region[key],
                        color="cyan",
                        linewidth=2,
                        alpha=0.8,
                    )

            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.axhline(
                float(np.mean(profile)),
                color="red",
                linestyle="--",
                label=f"mean = {np.mean(profile):.3g}",
            )
            ax.axhline(
                float(np.median(profile)),
                color="green",
                linestyle=":",
                label=f"median = {np.median(profile):.3g}",
            )
            if i == 0:
                threshold = reference_region["ref_intensity_threshold"]
                ax.axhline(
                    threshold,
                    color="magenta",
                    linestyle="--",
                    label=f"threshold = {threshold:.3g}",
                )
                ax.set_title(title)

        last_ax = axes[-1]
        last_ax.set_xlabel("Row index (y)")
        for edge in bucket_edges:
            last_ax.axvline(edge, color="orange", linestyle="--", alpha=0.4)

        score = profiles[-1][1]
        for region in results["subregions"]:
            row = region["max_row"]
            last_ax.plot(row, score[row], "ro")
        fig.text(
            0.5,
            0.01,
            "Dark blue lines: illuminated region from reference profile. "
            "Light blue lines: illuminated region from SNR. "
            "Orange dashed lines: bucket boundaries. "
            "Red dots: selected representative rows.",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        
        # Build legends after all artists, including bucket markers, exist.
        for ax in axes:
            ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 1])

        if args.out_pdf is not None:
            fig.savefig(str(args.out_pdf), format="pdf")
            log.info("Plot saved to %s", args.out_pdf)
        elif matplotlib.get_backend().lower() == "agg":
            log.info("Noninteractive Agg backend; use --out-pdf to save the plot.")
        else:
            plt.show()
    finally:
        plt.close(fig)


def run(args):
    """Load data, estimate bounds, select rows, and produce outputs."""
    ref_avg = load_frame(args.ref_avg, "ref_avg", nonnegative=True)
    img_avg = load_frame(
        args.img_avg, "img_avg", ref_avg.shape, nonnegative=True
    )
    img_sd = load_frame(
        args.img_sd, "img_sd", ref_avg.shape, nonnegative=True
    )

    ref_profile = ref_avg.mean(axis=1)
    ref_mean = float(np.mean(ref_profile))
    if ref_mean <= 0:
        raise ValueError("Reference mean intensity must be positive.")
    ref_rel = ref_profile / ref_mean

    top, bottom = profile_bounds(
        ref_profile,
        ref_mean * args.illum_factor,
        "Vertical reference illumination",
    )
    ref_profile_h = ref_avg.mean(axis=0)
    left, right = profile_bounds(
        ref_profile_h,
        float(np.mean(ref_profile_h)) * args.illum_factor,
        "Horizontal reference illumination",
    )

    reference_region = region_geometry(left, right, top, bottom)
    reference_region.update(
        mean_ref_intensity=ref_mean,
        threshold_factor=args.illum_factor,
        ref_intensity_threshold=ref_mean * args.illum_factor,
    )
    title = report_region(
        reference_region, "", f"factor = {args.illum_factor}"
    )
    results = {"illuminated_region": reference_region}

    if args.snr is not None:
        snr = load_frame(args.snr, "snr", ref_avg.shape)
        snr_profile = snr.mean(axis=1)
        top, bottom = profile_bounds(
            snr_profile, args.snr_threshold, "Vertical SNR illumination"
        )
        left, right = profile_bounds(
            snr.mean(axis=0),
            args.snr_threshold,
            "Horizontal SNR illumination",
        )
        snr_region = region_geometry(left, right, top, bottom)
        snr_region.update(
            mean_snr=float(np.mean(snr_profile[top:bottom + 1])),
            snr_threshold=args.snr_threshold,
        )
        results["illuminated_region_snr"] = snr_region
        report_region(
            snr_region, " based on SNR", f"threshold = {args.snr_threshold}"
        )
        # Preserve original behavior: SNR bounds replace reference bounds
        # for bucket selection; they are not intersected.

    img_profile = img_avg.mean(axis=1)
    sd_profile = img_sd.mean(axis=1)
    sd_median = float(np.median(sd_profile))
    if sd_median <= 0:
        raise ValueError(
            "Median row standard deviation must be positive; "
            "the original median-based clipping would otherwise produce "
            "an all-zero denominator."
        )
    sd_profile = np.clip(sd_profile, sd_median * 0.5, sd_median * 100.0)

    score = np.divide(
        np.sqrt(img_profile),
        sd_profile,
        out=np.zeros_like(sd_profile),
        where=sd_profile > 0,
    ) * ref_rel
    if not np.all(np.isfinite(score)):
        raise ValueError("Computed row quality scores contain nonfinite values.")

    bucket_edges, regions = select_subregions(
        ref_profile,
        img_profile,
        sd_profile,
        score,
        top,
        bottom,
        args.buckets,
    )
    results["subregion_count"] = len(regions)
    results["subregions"] = regions

    # Save numerical results before any interactive plot can block.
    if args.out_json is not None:
        with args.out_json.open("w", encoding="utf-8") as output:
            json.dump(results, output, indent=2, allow_nan=False)
            output.write("\n")
        log.info("Results saved to %s", args.out_json)

    plot_profiles(
        args,
        results,
        [
            ("ref_avg", ref_profile),
            ("tra_avg", img_profile),
            ("RQP", score),
        ],
        bucket_edges,
        title,
    )
    return 0


def main(argv=None):
    """Console-script entry point; return an exit status."""
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()

    print(
        "START %s %s" % (PROGRAM_NAME, " ".join(command_arguments)),
        flush=True,
    )
    print(
        "Date: %s" % datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        flush=True,
    )

    try:
        configure_logging()
        parser = build_parser()
        args = parser.parse_args(command_arguments)
        validate_arguments(parser, args)
        return run(args)
    except KeyboardInterrupt:
        log.error("%s interrupted by the user.", PROGRAM_NAME)
        return 130
    except Exception:
        log.exception("Error in %s.", PROGRAM_NAME)
        return 1
    finally:
        print(
            "END %s in %s"
            % (PROGRAM_NAME, format_elapsed(time.perf_counter() - started)),
            flush=True,
        )

if __name__ == "__main__":
    sys.exit(main())
