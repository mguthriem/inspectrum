"""Quick diagnostic: test sweep_pressure on SNAP data."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from inspectrum.background import estimate_background
from inspectrum.crystallography import generate_reflections
from inspectrum.engine import tof_to_d
from inspectrum.lattice import format_refinement_report, refine_all_phases
from inspectrum.loaders import load_gsa, load_instprm, load_phase_descriptions
from inspectrum.matching import sweep_pressure
from inspectrum.models import DiffractionSpectrum
from inspectrum.peakfinding import find_peaks_in_spectrum
from inspectrum.plotting import inspect_phase_matches
from inspectrum.resolution import fwhm_at_d, parse_resolution_curve

TEST_DATA = Path("tests/test_data")

# Load data
spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
spec = spectra[0]
inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")
exp = load_phase_descriptions(TEST_DATA / "snap_phases.json")

# Background subtraction & peak finding
_, peaks = estimate_background(spec.y)
d_curve, fwhm_curve = parse_resolution_curve(inst)
d_axis = tof_to_d(spec.x, inst)
peak_table = find_peaks_in_spectrum(
    d_axis, peaks, resolution=(d_curve, fwhm_curve)
)
tol = fwhm_at_d(peak_table.positions, d_curve, fwhm_curve)

print(f"Found {len(peak_table.positions)} observed peaks")

# Generate reflections per phase
phase_reflections = {}
for desc in exp.phases:
    if desc.phase is None:
        continue
    refs = generate_reflections(desc.phase, d_axis.min(), d_axis.max())
    phase_reflections[desc.name] = refs
    print(f"  {desc.name}: {len(refs)} reflections, EOS={desc.eos}")

# Run pressure sweep
print("\n--- Pressure sweep (0-60 GPa) ---")
best_P, result = sweep_pressure(
    peak_table.positions, peak_table.heights, peak_table.fwhm,
    exp.phases, phase_reflections, tol,
    P_min=0.0, P_max=60.0,
    n_coarse=301, n_fine=201,
)

print(f"\nBest pressure: {best_P:.2f} GPa")
for pm in result.phase_matches:
    print(f"  {pm}")
    for mp in pm.matched_peaks:
        print(
            f"    {mp.hkl} obs_d={mp.obs_d:.5f} strained_d={mp.strained_d:.5f} "
            f"res={mp.residual:+.5f} F²={mp.F_sq:.4f}"
        )
print(f"  Unmatched: {result.unmatched_indices}")

# Show predicted strains at best pressure
from inspectrum.eos import predicted_strain
print(f"\nPredicted strains at P={best_P:.2f} GPa:")
for desc in exp.phases:
    if desc.eos is None:
        continue
    if desc.is_stable_at(best_P):
        s = predicted_strain(desc.eos, best_P)
        print(f"  {desc.name}: s={s:.6f}")
    else:
        print(f"  {desc.name}: not stable at this pressure")

# --- Lattice parameter refinement ---
# Estimate noise floor from the lower quartile of peak-subtracted data
q25 = float(np.percentile(peaks, 25))
lower_quarter = peaks[peaks <= q25]
noise_sigma = float(np.std(lower_quarter))

refinements = refine_all_phases(
    result, exp.phases, noise_sigma=noise_sigma,
)
report = format_refinement_report(refinements, sweep_pressure_gpa=best_P)
print(f"\n{report}")

d_spectrum = DiffractionSpectrum(
    x=d_axis,
    y=spec.y,
    e=spec.e,
    x_unit="d-Spacing (A)",
    y_unit=spec.y_unit,
    label=spec.label or "SNAP059056",
    bank=spec.bank,
    metadata=spec.metadata,
)
fig, _, table_text = inspect_phase_matches(
    d_spectrum,
    peaks,
    result,
    observed_positions=peak_table,
    phase_reflections=phase_reflections,
    refinements=refinements,
    title=f"Pressure-sweep matches at {best_P:.2f} GPa",
)
out = Path("/tmp/pressure_match_overlay.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nSaved phase-match overlay to {out}")
