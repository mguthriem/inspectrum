"""Quick test to verify FWHM bars render and save to PNG."""
import matplotlib
matplotlib.use("Agg")

from inspectrum.loaders import load_mantid_csv
from inspectrum.background import estimate_background
from inspectrum.peakfinding import find_peaks_in_spectrum
import matplotlib.pyplot as plt
import numpy as np

s = load_mantid_csv("tests/test_data/SNAP059056_all_test-0.csv")[0]
bg, peaks_y = estimate_background(s.y, win_size=4, smoothing=1.0)
pt = find_peaks_in_spectrum(s.x, peaks_y, min_prominence=4.0, min_width_pts=6)

print("n_peaks:", pt.n_peaks)
for i in range(pt.n_peaks):
    print(f"  d={pt.positions[i]:.4f}  h={pt.heights[i]:.1f}  fwhm={pt.fwhm[i]:.4f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s.x, peaks_y, linewidth=0.5, color="C2")
ax.vlines(pt.positions, ymin=0, ymax=pt.heights, colors="C3", linewidth=0.8, alpha=0.8)
ax.plot(pt.positions, pt.heights, "v", color="C3", markersize=5)

# This is what inspect_peaks should be doing:
half_heights = pt.heights / 2.0
ax.errorbar(
    pt.positions, half_heights,
    xerr=pt.fwhm / 2,
    fmt="none", ecolor="k", elinewidth=1.5,
    capsize=4, capthick=1.5, alpha=0.8,
    label="FWHM",
)

ax.legend()
out = "/tmp/fwhm_test.png"
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
