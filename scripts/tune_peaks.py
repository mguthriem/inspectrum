#!/usr/bin/env python
"""
Interactive parameter tuning for inspectrum background + peak finding.

Opens a matplotlib window with sliders for all background and peak-finding
parameters.  Drag a slider → the background, peaks, and peak markers
update live.  Use the toolbar to zoom/pan and the cursor to read coords.

Usage:
    pixi run python scripts/tune_peaks.py tests/test_data/SNAP059056_all_test-0.csv

Or from a Python session:
    from scripts.tune_peaks import tune_peaks
    tune_peaks("tests/test_data/SNAP059056_all_test-0.csv")
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

# Ensure the src directory is importable when running as a script
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from inspectrum.loaders import load_mantid_csv
from inspectrum.background import estimate_background
from inspectrum.peakfinding import find_peaks_in_spectrum


def tune_peaks(csv_path: str, bank: int = 0) -> None:
    """Open an interactive tuning window.

    Args:
        csv_path: Path to a Mantid-exported CSV file.
        bank: Which spectrum (bank index) to use if the file has
            multiple banks.
    """
    spectra = load_mantid_csv(csv_path)
    s = spectra[bank]

    # ── Initial parameter values ──
    init = dict(
        win_size=4,
        smoothing=1.0,
        lls=True,
        decrease=True,
        min_prominence=4.0,
        min_width_pts=6,
    )

    # ── Build figure ──
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title(f"inspectrum tuner — {s.label or csv_path}")

    # Reserve space: top 70% for plots, bottom 30% for sliders
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], top=0.92, bottom=0.38,
                          hspace=0.15, left=0.07, right=0.97)
    ax_raw = fig.add_subplot(gs[0])
    ax_bg  = fig.add_subplot(gs[1], sharex=ax_raw)
    ax_pk  = fig.add_subplot(gs[2], sharex=ax_raw)

    # Slider axes
    slider_left = 0.20
    slider_width = 0.55
    slider_h = 0.025
    slider_gap = 0.035
    y0 = 0.28  # top of slider area

    ax_win   = fig.add_axes([slider_left, y0 - 0*slider_gap, slider_width, slider_h])
    ax_sm    = fig.add_axes([slider_left, y0 - 1*slider_gap, slider_width, slider_h])
    ax_prom  = fig.add_axes([slider_left, y0 - 2*slider_gap, slider_width, slider_h])
    ax_wid   = fig.add_axes([slider_left, y0 - 3*slider_gap, slider_width, slider_h])
    ax_lls   = fig.add_axes([slider_left, y0 - 4*slider_gap, 0.08, slider_h])
    ax_dec   = fig.add_axes([slider_left + 0.15, y0 - 4*slider_gap, 0.08, slider_h])
    ax_reset = fig.add_axes([slider_left + 0.35, y0 - 4*slider_gap, 0.08, slider_h])

    s_win  = Slider(ax_win,  "win_size",     1, 80, valinit=init["win_size"], valstep=1)
    s_sm   = Slider(ax_sm,   "smoothing",  0.0, 20.0, valinit=init["smoothing"])
    s_prom = Slider(ax_prom, "prominence", 0.0, 20.0, valinit=init["min_prominence"])
    s_wid  = Slider(ax_wid,  "min_width",    1, 30, valinit=init["min_width_pts"], valstep=1)

    b_lls   = Button(ax_lls,   "LLS: ON")
    b_dec   = Button(ax_dec,   "dec: ON")
    b_reset = Button(ax_reset, "Reset")

    # Mutable state for toggle buttons
    state = dict(lls=init["lls"], decrease=init["decrease"])

    # ── Plot raw data (static) ──
    ax_raw.plot(s.x, s.y, linewidth=0.5, color="C0")
    ax_raw.set_ylabel(s.y_unit)
    ax_raw.set_title(f"{s.label or Path(csv_path).name} — raw data")

    # ── Dynamic plot elements ──
    line_obs,  = ax_bg.plot([], [], linewidth=0.5, color="C0", alpha=0.5, label="observed")
    line_bg,   = ax_bg.plot([], [], linewidth=1.0, color="C1", label="background")
    line_res,  = ax_bg.plot([], [], linewidth=0.5, color="C2", alpha=0.7, label="peaks")
    ax_bg.set_ylabel(s.y_unit)
    ax_bg.set_title("Background subtraction")
    ax_bg.legend(fontsize="small", loc="upper right")

    line_pk, = ax_pk.plot([], [], linewidth=0.5, color="C2")
    vlines = ax_pk.vlines([], 0, 1, colors="C3", linewidth=0.8)
    markers, = ax_pk.plot([], [], "v", color="C3", markersize=5)
    peak_label = ax_pk.text(0.99, 0.95, "", transform=ax_pk.transAxes,
                            ha="right", va="top", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="wheat", alpha=0.8))
    ax_pk.set_xlabel(s.x_unit)
    ax_pk.set_ylabel(s.y_unit)
    ax_pk.set_title("Detected peaks")

    def update(_=None):
        win = int(s_win.val)
        sm  = float(s_sm.val)
        prom_val = float(s_prom.val)
        wid = int(s_wid.val)
        use_lls = state["lls"]
        use_dec = state["decrease"]
        prom = prom_val if prom_val > 0 else None

        # Run pipeline
        bg, peaks_y = estimate_background(
            s.y, win_size=win, decrease=use_dec, lls=use_lls, smoothing=sm,
        )
        pt = find_peaks_in_spectrum(
            s.x, peaks_y,
            min_prominence=prom, min_width_pts=wid,
        )

        # Update background panel
        line_obs.set_data(s.x, s.y)
        line_bg.set_data(s.x, bg)
        line_res.set_data(s.x, peaks_y)
        ax_bg.relim()
        ax_bg.autoscale_view()

        # Update peaks panel
        line_pk.set_data(s.x, peaks_y)

        # Remove old vlines and redraw
        for c in ax_pk.collections[:]:
            c.remove()
        if pt.n_peaks > 0:
            ax_pk.vlines(pt.positions, 0, pt.heights,
                         colors="C3", linewidth=0.8, alpha=0.8)
            markers.set_data(pt.positions, pt.heights)
        else:
            markers.set_data([], [])

        peak_label.set_text(f"{pt.n_peaks} peaks")
        ax_pk.relim()
        ax_pk.autoscale_view()

        fig.canvas.draw_idle()

        # Print to console
        print(f"\rwin={win:3d}  sm={sm:5.1f}  lls={use_lls}  "
              f"dec={use_dec}  prom={'auto' if prom is None else f'{prom:.1f}':>5s}  "
              f"wid={wid:2d}  → {pt.n_peaks} peaks", end="", flush=True)

    def toggle_lls(_):
        state["lls"] = not state["lls"]
        b_lls.label.set_text(f"LLS: {'ON' if state['lls'] else 'OFF'}")
        update()

    def toggle_dec(_):
        state["decrease"] = not state["decrease"]
        b_dec.label.set_text(f"dec: {'ON' if state['decrease'] else 'OFF'}")
        update()

    def reset(_):
        s_win.set_val(init["win_size"])
        s_sm.set_val(init["smoothing"])
        s_prom.set_val(init["min_prominence"])
        s_wid.set_val(init["min_width_pts"])
        state["lls"] = init["lls"]
        state["decrease"] = init["decrease"]
        b_lls.label.set_text(f"LLS: {'ON' if state['lls'] else 'OFF'}")
        b_dec.label.set_text(f"dec: {'ON' if state['decrease'] else 'OFF'}")
        update()

    # Connect callbacks
    s_win.on_changed(update)
    s_sm.on_changed(update)
    s_prom.on_changed(update)
    s_wid.on_changed(update)
    b_lls.on_clicked(toggle_lls)
    b_dec.on_clicked(toggle_dec)
    b_reset.on_clicked(reset)

    # Initial draw
    update()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tune_peaks.py <csv_file> [bank]")
        print("Example: python tune_peaks.py tests/test_data/SNAP059056_all_test-0.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    bank_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tune_peaks(csv_file, bank=bank_idx)
