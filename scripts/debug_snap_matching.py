"""Diagnose why SNAP tungsten peaks aren't matching."""

from pathlib import Path

import numpy as np

from inspectrum.background import estimate_background
from inspectrum.crystallography import generate_reflections
from inspectrum.engine import tof_to_d
from inspectrum.loaders import load_gsa, load_instprm, load_phase_descriptions
from inspectrum.matching import match_peaks_at_strain
from inspectrum.peakfinding import find_peaks_in_spectrum
from inspectrum.resolution import fwhm_at_d, parse_resolution_curve

TEST_DATA = Path("tests/test_data")

# Load data
spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
spec = spectra[0]
inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")
exp = load_phase_descriptions(TEST_DATA / "snap_phases.json")

# Background subtraction
_, peaks = estimate_background(spec.y)

# Resolution curve for tolerance
d_curve, fwhm_curve = parse_resolution_curve(inst)

# Peak finding
d_axis = tof_to_d(spec.x, inst)
peak_table = find_peaks_in_spectrum(
    d_axis, peaks, resolution=(d_curve, fwhm_curve)
)

# Tolerance
tol = fwhm_at_d(peak_table.positions, d_curve, fwhm_curve)

print(f"d-axis range: [{d_axis.min():.4f}, {d_axis.max():.4f}] A")
print(f"Found {len(peak_table.positions)} peaks:")
for i, (d, h, fw, t) in enumerate(
    zip(peak_table.positions, peak_table.heights, peak_table.fwhm, tol)
):
    print(f"  [{i}] d={d:.5f} A  h={h:.1f}  fwhm={fw:.5f}  tol={t:.5f}")

# Generate reflections
for desc in exp.phases:
    if desc.phase is None:
        continue
    refs = generate_reflections(desc.phase, d_axis.min(), d_axis.max())
    print(f"\n{desc.name}: {len(refs)} reflections")
    for r in refs:
        print(f"  {r['hkl']}  d={r['d']:.5f}  mult={r['multiplicity']}  F_sq={r['F_sq']:.4f}")

    # Try matching at s=1.0 with generous tolerance
    matches = match_peaks_at_strain(
        peak_table.positions, peak_table.heights, peak_table.fwhm,
        refs, 1.0, tol,
    )
    print(f"  Matches at s=1.0 (tol=fwhm): {len(matches)}")
    for m in matches:
        print(f"    obs_d={m.obs_d:.5f} calc_d={m.calc_d:.5f} strained_d={m.strained_d:.5f} res={m.residual:.5f}")

    # Try with 2x tolerance
    matches2 = match_peaks_at_strain(
        peak_table.positions, peak_table.heights, peak_table.fwhm,
        refs, 1.0, tol * 2,
    )
    print(f"  Matches at s=1.0 (tol=2*fwhm): {len(matches2)}")
    for m in matches2:
        print(f"    obs_d={m.obs_d:.5f} calc_d={m.calc_d:.5f} res={m.residual:.5f}")

    # Show closest observed peak for each reflection
    print(f"  Nearest obs for each reflection:")
    for r in refs:
        dists = np.abs(peak_table.positions - r["d"])
        closest = np.argmin(dists)
        print(
            f"    {r['hkl']} calc_d={r['d']:.5f}  "
            f"nearest obs={peak_table.positions[closest]:.5f}  "
            f"dist={dists[closest]:.5f}  "
            f"tol={tol[closest]:.5f}"
        )
