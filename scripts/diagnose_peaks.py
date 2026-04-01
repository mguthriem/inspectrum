"""Diagnostic: characterize all detected peaks vs resolution & noise floor."""

from pathlib import Path
import numpy as np
from inspectrum.background import estimate_background
from inspectrum.crystallography import generate_reflections
from inspectrum.engine import tof_to_d
from inspectrum.loaders import load_gsa, load_instprm, load_phase_descriptions
from inspectrum.matching import sweep_pressure
from inspectrum.peakfinding import find_peaks_in_spectrum
from inspectrum.resolution import parse_resolution_curve, fwhm_at_d

TEST_DATA = Path("tests/test_data")
spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
spec = spectra[0]
inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")
exp = load_phase_descriptions(TEST_DATA / "snap_phases.json")
_, peaks = estimate_background(spec.y)
d_curve, fwhm_curve = parse_resolution_curve(inst)
d_axis = tof_to_d(spec.x, inst)

# Get peaks with current resolution filter
pt = find_peaks_in_spectrum(d_axis, peaks, resolution=(d_curve, fwhm_curve))

# Noise floor estimate
q25 = float(np.percentile(peaks, 25))
lower_q = peaks[peaks <= q25]
noise_sigma = float(np.std(lower_q))
print(f"Noise sigma (lower quartile): {noise_sigma:.4f}")
print(f"Auto prominence threshold (1*sigma): {noise_sigma:.4f}")
print()

# Expected instrument FWHM at each peak
expected_fwhm = fwhm_at_d(pt.positions, d_curve, fwhm_curve)

header = "{:>2s} {:>10s} {:>10s} {:>12s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
    "#", "d-spacing", "height", "prominence", "obs_FWHM", "exp_FWHM", "FWHM_ratio", "prom/sigma"
)
print(f"Peaks found: {pt.n_peaks}")
print(header)
print("-" * len(header))
for i in range(pt.n_peaks):
    ratio = pt.fwhm[i] / expected_fwhm[i]
    snr = pt.prominences[i] / noise_sigma
    print(
        f"{i+1:2d} {pt.positions[i]:10.5f} {pt.heights[i]:10.2f} "
        f"{pt.prominences[i]:12.4f} {pt.fwhm[i]:10.5f} "
        f"{expected_fwhm[i]:10.5f} {ratio:10.3f} {snr:10.2f}"
    )

# --- Match analysis ---
print()
tol = fwhm_at_d(pt.positions, d_curve, fwhm_curve)
phase_reflections = {}
for desc in exp.phases:
    if desc.phase is None:
        continue
    refs = generate_reflections(desc.phase, d_axis.min(), d_axis.max())
    phase_reflections[desc.name] = refs

best_P, result = sweep_pressure(
    pt.positions, pt.heights, pt.fwhm,
    exp.phases, phase_reflections, tol,
    P_min=0.0, P_max=60.0, n_coarse=301, n_fine=201,
)

matched_indices = set()
for pm in result.phase_matches:
    for mp in pm.matched_peaks:
        matched_indices.add(mp.obs_idx)

print(f"Best P = {best_P:.2f} GPa")
print()
print("MATCHED peaks:")
for i in range(pt.n_peaks):
    if i in matched_indices:
        ratio = pt.fwhm[i] / expected_fwhm[i]
        snr = pt.prominences[i] / noise_sigma
        print(f"  #{i+1:2d} d={pt.positions[i]:.5f} prom/s={snr:.1f} fwhm_r={ratio:.3f}")

print()
print("UNMATCHED peaks:")
for i in range(pt.n_peaks):
    if i not in matched_indices:
        ratio = pt.fwhm[i] / expected_fwhm[i]
        snr = pt.prominences[i] / noise_sigma
        print(f"  #{i+1:2d} d={pt.positions[i]:.5f} prom/s={snr:.1f} fwhm_r={ratio:.3f}")
