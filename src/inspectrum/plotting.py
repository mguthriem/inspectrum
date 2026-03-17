"""
Diagnostic plotting for inspectrum.

Quick-look visualisation of diffraction spectra and pipeline
intermediate results.  These are development/inspection helpers,
not publication-quality figures.

All functions return the matplotlib ``(fig, ax)`` tuple so you can
further customise or save::

    fig, ax = plot_spectrum(spectrum)
    fig.savefig("quick_look.png", dpi=150)

Functions that overlay multiple datasets (e.g. background + peaks)
use a consistent colour scheme:

- blue:   observed data
- orange: estimated background
- green:  peak-only signal
- red:    calculated/expected tick marks
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from inspectrum.models import DiffractionSpectrum
from inspectrum.peakfinding import PeakTable


def plot_spectrum(
    spectrum: DiffractionSpectrum,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single diffraction spectrum.

    Args:
        spectrum: Spectrum to plot.
        title: Override for the plot title (default: spectrum label).
        ax: Existing axes to draw on.  A new figure is created if None.

    Returns:
        ``(fig, ax)`` tuple.
    """
    fig, ax = _get_fig_ax(ax)
    ax.plot(spectrum.x, spectrum.y, linewidth=0.6, color="C0")
    ax.set_xlabel(spectrum.x_unit)
    ax.set_ylabel(spectrum.y_unit)
    ax.set_title(title or spectrum.label or "Spectrum")
    return fig, ax


def plot_background(
    spectrum: DiffractionSpectrum,
    background: NDArray[np.float64],
    peaks: NDArray[np.float64],
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Overlay observed data, estimated background, and peak signal.

    Args:
        spectrum: Original observed spectrum.
        background: Background estimate (same length as spectrum.y).
        peaks: Peak-only signal (spectrum.y − background).
        title: Override for the plot title.
        ax: Existing axes to draw on.

    Returns:
        ``(fig, ax)`` tuple.
    """
    fig, ax = _get_fig_ax(ax)
    ax.plot(spectrum.x, spectrum.y, linewidth=0.6, color="C0",
            label="observed", alpha=0.7)
    ax.plot(spectrum.x, background, linewidth=1.0, color="C1",
            label="background")
    ax.plot(spectrum.x, peaks, linewidth=0.5, color="C2",
            label="peaks", alpha=0.6)
    ax.set_xlabel(spectrum.x_unit)
    ax.set_ylabel(spectrum.y_unit)
    ax.set_title(title or f"{spectrum.label or 'Spectrum'} — background")
    ax.legend(fontsize="small")
    return fig, ax


def plot_peak_markers(
    spectrum: DiffractionSpectrum,
    peaks: NDArray[np.float64],
    *,
    observed_positions: NDArray[np.float64] | PeakTable | None = None,
    reflections: list[dict[str, Any]] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot peak-only signal with observed and/or calculated tick marks.

    Vertical lines mark peak positions:

    - **Red ticks (top)**: calculated d-spacings from ``reflections``
    - **Blue ticks (bottom)**: observed peak positions

    Args:
        spectrum: Original spectrum (provides the x-axis).
        peaks: Background-subtracted peak signal.
        observed_positions: Observed peak d-spacings — either an array
            or a :class:`~inspectrum.peakfinding.PeakTable` (optional).
        reflections: List of reflection dicts from
            :func:`~inspectrum.crystallography.generate_reflections`
            (optional).  Each dict must have a ``"d"`` key.
        title: Override for the plot title.
        ax: Existing axes to draw on.

    Returns:
        ``(fig, ax)`` tuple.
    """
    fig, ax = _get_fig_ax(ax)
    ax.plot(spectrum.x, peaks, linewidth=0.5, color="C2", label="peaks")

    y_max = float(np.max(peaks)) if len(peaks) > 0 else 1.0

    if reflections is not None:
        calc_d = np.array([r["d"] for r in reflections])
        ax.vlines(calc_d, ymin=y_max * 0.85, ymax=y_max,
                  colors="C3", linewidth=0.8, label="calc d")

    if observed_positions is not None:
        obs_d = (
            observed_positions.positions
            if isinstance(observed_positions, PeakTable)
            else observed_positions
        )
        ax.vlines(obs_d, ymin=0, ymax=y_max * 0.15,
                  colors="C0", linewidth=0.8, label="obs peaks")

    ax.set_xlabel(spectrum.x_unit)
    ax.set_ylabel(spectrum.y_unit)
    ax.set_title(title or f"{spectrum.label or 'Spectrum'} — peak markers")
    ax.legend(fontsize="small")
    return fig, ax


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_fig_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return (fig, ax), creating a new figure if *ax* is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    else:
        fig = ax.get_figure()
    return fig, ax


# ---------------------------------------------------------------------------
# Interactive inspection
# ---------------------------------------------------------------------------


def inspect_peaks(
    spectrum: DiffractionSpectrum,
    *,
    win_size: int = 40,
    decrease: bool = True,
    lls: bool = True,
    smoothing: float = 5.0,
    min_prominence: float | None = None,
    min_width_pts: int = 5,
) -> PeakTable:
    """Run the full pipeline and open an interactive 3-panel figure.

    Panels (top to bottom):

    1. **Raw data** — observed spectrum as-is.
    2. **Background subtraction** — observed (blue) + estimated
       background (orange) + peak-only residual (green).
    3. **Peak detection** — peak-only signal (green) with red
       vertical lines at each detected peak position.

    The matplotlib window provides **zoom** (rectangle-zoom icon or
    scroll wheel), **pan** (hand icon), and a **crosshair cursor**
    that reports ``(d, intensity)`` coordinates in the bottom-left
    corner of the toolbar.

    All background and peak-finding parameters are forwarded (see
    :func:`~inspectrum.background.estimate_background` and
    :func:`~inspectrum.peakfinding.find_peaks_in_spectrum`).

    Args:
        spectrum: Spectrum to inspect.
        win_size: Background estimator half-window size.
        decrease: Background estimator scan direction.
        lls: Apply LLS transform during background estimation.
        smoothing: Gaussian smoothing σ for background estimation.
        min_prominence: Peak prominence threshold (None = auto).
        min_width_pts: Minimum peak FWHM in data points.

    Returns:
        :class:`~inspectrum.peakfinding.PeakTable` of detected peaks
        (so you can inspect it programmatically too).
    """
    from matplotlib.widgets import Cursor

    from inspectrum.background import estimate_background
    from inspectrum.peakfinding import find_peaks_in_spectrum

    # --- Run pipeline ---
    bg, peaks_y = estimate_background(
        spectrum.y,
        win_size=win_size,
        decrease=decrease,
        lls=lls,
        smoothing=smoothing,
    )
    peak_table = find_peaks_in_spectrum(
        spectrum.x,
        peaks_y,
        min_prominence=min_prominence,
        min_width_pts=min_width_pts,
    )

    # --- Build figure ---
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # Panel 1: raw data
    ax1 = axes[0]
    ax1.plot(spectrum.x, spectrum.y, linewidth=0.6, color="C0")
    ax1.set_ylabel(spectrum.y_unit)
    ax1.set_title(f"{spectrum.label or 'Spectrum'} — raw data")

    # Panel 2: background subtraction
    ax2 = axes[1]
    ax2.plot(spectrum.x, spectrum.y, linewidth=0.5, color="C0",
             label="observed", alpha=0.5)
    ax2.plot(spectrum.x, bg, linewidth=1.0, color="C1",
             label="background")
    ax2.plot(spectrum.x, peaks_y, linewidth=0.5, color="C2",
             label="peaks", alpha=0.7)
    ax2.set_ylabel(spectrum.y_unit)
    ax2.set_title("Background subtraction")
    ax2.legend(fontsize="small", loc="upper right")

    # Panel 3: peaks with markers
    ax3 = axes[2]
    ax3.plot(spectrum.x, peaks_y, linewidth=0.5, color="C2")
    if peak_table.n_peaks > 0:
        ax3.vlines(
            peak_table.positions, ymin=0,
            ymax=peak_table.heights,
            colors="C3", linewidth=0.8, alpha=0.8,
            label=f"{peak_table.n_peaks} peaks",
        )
        # Small dots at the apex
        ax3.plot(
            peak_table.positions, peak_table.heights,
            "v", color="C3", markersize=5,
        )
    ax3.set_xlabel(spectrum.x_unit)
    ax3.set_ylabel(spectrum.y_unit)
    ax3.set_title("Detected peaks")
    ax3.legend(fontsize="small", loc="upper right")

    # Crosshair cursor on all panels
    cursors = []
    for ax in axes:
        cursors.append(
            Cursor(ax, useblit=True, color="gray", linewidth=0.5)
        )
    # Keep references alive so the cursors don't get garbage-collected
    fig._inspectrum_cursors = cursors  # type: ignore[attr-defined]

    fig.align_ylabels(axes)
    plt.show(block=False)

    # Print a summary to the console
    print(f"\n{'─' * 50}")
    print(f"  {peak_table.n_peaks} peaks detected")
    print(f"{'─' * 50}")
    for i in range(peak_table.n_peaks):
        print(
            f"  d = {peak_table.positions[i]:.4f} Å   "
            f"height = {peak_table.heights[i]:.1f}   "
            f"FWHM = {peak_table.fwhm[i]:.4f} Å"
        )
    print(f"{'─' * 50}\n")

    return peak_table
