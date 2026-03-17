"""
Peak finding for inspectrum.

Detects peaks in background-subtracted diffraction data and returns
their positions, intensities, and widths in d-spacing units.

Uses :func:`scipy.signal.find_peaks` internally, with defaults tuned
for neutron powder diffraction on SNAP-like instruments (~863 points,
d ~ 0.8–2.5 Å).

Typical usage::

    from inspectrum.loaders import load_mantid_csv
    from inspectrum.background import estimate_background
    from inspectrum.peakfinding import find_peaks_in_spectrum

    spectra = load_mantid_csv("SNAP059056_all_test-0.csv")
    s = spectra[0]
    bg, peaks_y = estimate_background(s.y)
    peak_table = find_peaks_in_spectrum(s.x, peaks_y)

    for i in range(peak_table.n_peaks):
        print(f"d={peak_table.positions[i]:.4f}  "
              f"height={peak_table.heights[i]:.1f}  "
              f"fwhm={peak_table.fwhm[i]:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_prominences, peak_widths


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PeakTable:
    """Container for detected peaks.

    All arrays have length ``n_peaks`` and are sorted by decreasing
    d-spacing (lowest-angle first, matching reflection list order).

    Attributes:
        positions: d-spacing (Å) of each peak centre.
        heights: Intensity at each peak apex.
        prominences: Prominence of each peak (height above surrounding
            baseline — a robust measure of peak significance).
        fwhm: Full width at half-maximum in d-spacing units (Å).
        indices: Index into the original x/y arrays for each peak.
    """

    positions: NDArray[np.float64]
    heights: NDArray[np.float64]
    prominences: NDArray[np.float64]
    fwhm: NDArray[np.float64]
    indices: NDArray[np.intp]

    @property
    def n_peaks(self) -> int:
        """Number of detected peaks."""
        return len(self.positions)

    def __repr__(self) -> str:
        return f"PeakTable(n_peaks={self.n_peaks})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_peaks_in_spectrum(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    min_prominence: float | None = None,
    min_width_pts: int = 5,
    min_distance_pts: int = 3,
) -> PeakTable:
    """Find peaks in a background-subtracted spectrum.

    Args:
        x: Independent variable (d-spacing in Å).  Must be
            monotonically increasing or decreasing.
        y: Background-subtracted intensity (peak-only signal).
            Same length as *x*.
        min_prominence: Minimum peak prominence (absolute).  Peaks
            below this are rejected.  If ``None`` (default), a
            threshold is estimated automatically as the standard
            deviation of the lower quartile of *y* — a robust
            measure of counting noise that ignores peaks.
        min_width_pts: Minimum full-width at half-maximum in data
            points.  Rejects noise spikes narrower than this.
            Default 5 points.
        min_distance_pts: Minimum separation between neighbouring
            peaks in data points.  Default 3.

    Returns:
        :class:`PeakTable` sorted by decreasing d-spacing.

    Raises:
        ValueError: If *x* and *y* have different lengths or fewer
            than 3 points.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have equal length, got {len(x)} and {len(y)}"
        )
    if len(x) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x)}")

    # Ensure we work on a positive-upward signal
    y_work = np.asarray(y, dtype=np.float64)

    # Auto-threshold: noise σ from the lower quartile of the signal.
    # After background subtraction, peak-free regions cluster in the
    # lower quartile.  Their standard deviation is a robust estimate
    # of the counting-noise floor.
    if min_prominence is None:
        q25 = float(np.percentile(y_work, 25))
        lower_quarter = y_work[y_work <= q25]
        noise_sigma = float(np.std(lower_quarter)) if len(lower_quarter) > 1 else 1.0
        min_prominence = max(noise_sigma, 1e-10)

    # ---- Find candidate peaks ----
    idx, _ = find_peaks(
        y_work,
        prominence=min_prominence,
        distance=min_distance_pts,
    )

    if len(idx) == 0:
        return _empty_table()

    # ---- Compute properties ----
    proms, _, _ = peak_prominences(y_work, idx)
    widths_pts, _, _, _ = peak_widths(y_work, idx, rel_height=0.5)

    # ---- Filter by width ----
    keep = widths_pts >= min_width_pts
    idx = idx[keep]
    proms = proms[keep]
    widths_pts = widths_pts[keep]

    if len(idx) == 0:
        return _empty_table()

    # ---- Convert widths from points to d-spacing ----
    # Use local point spacing at each peak for accurate conversion
    dx = np.abs(np.diff(x))
    # For each peak index, use the local spacing (clamped to valid range)
    local_dx = dx[np.clip(idx, 0, len(dx) - 1)]
    fwhm = widths_pts * local_dx

    # ---- Build output arrays ----
    positions = x[idx]
    heights = y_work[idx]

    # Sort by decreasing d-spacing (to match reflection list convention)
    order = np.argsort(-positions)
    return PeakTable(
        positions=positions[order],
        heights=heights[order],
        prominences=proms[order],
        fwhm=fwhm[order],
        indices=idx[order],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_table() -> PeakTable:
    """Return an empty PeakTable."""
    empty = np.array([], dtype=np.float64)
    return PeakTable(
        positions=empty,
        heights=empty,
        prominences=empty,
        fwhm=empty,
        indices=np.array([], dtype=np.intp),
    )
