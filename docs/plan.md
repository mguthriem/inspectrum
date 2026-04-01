# Inspectrum Implementation Plan

**Created**: 2026-03-30  
**Status**: In progress  
**Reference**: [project.md](project.md), [ground_truths.md](ground_truths.md)

---

## Project Summary

Inspectrum pre-inspects powder diffraction spectra to estimate good starting parameters (lattice parameters, scale factors, peak widths) for Rietveld refinement. It is designed for challenging data: high-pressure neutron diffraction with structured backgrounds and low signal-to-noise.

**Critical design constraint**: inspectrum does NOT refine. It estimates parameters by peak matching and geometric analysis, not by fitting a calculated pattern to observed data.

---

## Pipeline Overview

```
Input                        Pre-processing              Inspection                 Output
─────                        ──────────────              ──────────                 ──────
Spectra (.gsa/.csv)    ──►   Background subtraction ──►  Peak finding         ──►  Optimised lattice params
CIF files              ──►   Generate reflections   ──►  Peak matching        ──►  Optimised scale factors
Instrument (.instprm)  ──►   Resolution analysis    ──►  Parameter estimation ──►  Optimised peak widths
Phase descriptions     ──►   EOS-predicted strain   ──►  (narrows search)     ──►  Processed spectra
```

### Pipeline Steps (in order)

| # | Step | Input | Output | Module | Status |
|---|------|-------|--------|--------|--------|
| 1 | Load data | Files on disk | DiffractionSpectrum, CrystalPhase, Instrument | `loaders.py` | ✅ Done |
| 2 | Load phase descriptions | JSON file | PhaseDescription with EOS | `loaders.py` | ⬜ Todo |
| 3 | Background subtraction | Raw spectrum | Background + peak signal | `background.py` | ✅ Done |
| 4 | Peak finding | Peak signal | PeakTable (positions, FWHM, heights) | `peakfinding.py` | ✅ Done |
| 5 | Generate expected reflections | CrystalPhase + d-range | Reflection list (d, hkl, F², mult) | `crystallography.py` | ✅ Done |
| 6 | **EOS strain prediction** | PhaseDescription + pressure | Predicted strain s = (V/V₀)^(1/3) | `eos.py` (new) | ⬜ Todo |
| 7 | **Peak matching** | Observed peaks + calculated reflections | Matched pairs (obs_d ↔ calc_d, phase assignment) | `matching.py` (new) | ⬜ Todo |
| 8 | **Lattice parameter estimation** | Matched pairs | Updated lattice params per phase | `engine.py` (refactored) | ⬜ Todo |
| 9 | **Scale + width estimation** | Matched pairs + observed heights/FWHM | Scale factors, peak width params | `engine.py` (refactored) | ⬜ Todo |

---

## Phase 1: Schema & EOS (Current)

### 1.1 New data models in `models.py`

Add three new dataclasses:

- **`EquationOfState`**: EOS type, order, V₀ (ų/cell), K₀ (GPa), K′, source citation, extra params
- **`SampleConditions`**: pressure (GPa), temperature (K) — both optional  
- **`PhaseDescription`**: wraps a CIF path + role (calibrant/sample) + reference conditions + EOS + stability range

Key design decisions:
- V₀ is always stored internally in **ų per unit cell** — conversion from published units (cm³/mol, ų/atom) happens at load time
- EOS is optional — if absent, the matcher does a blind strain search
- `role: "calibrant"` marks phases with well-known EOS usable for pressure determination

### 1.2 JSON serialization + loader

- Human-editable JSON format with `V_0_unit` and `Z` fields for unit conversion
- Error values stored as `_err` suffixed fields → go into `eos.extra`
- `load_phase_descriptions(json_path)` returns list of PhaseDescription, CIF loaded automatically

### 1.3 Test data

JSON for the current SNAP test dataset:
- **Tungsten** (calibrant): Vinet EOS from Dewaele et al., PRB 70 094112 (2004). V₀=15.862 ų/atom, K₀=295.2 GPa, K′=4.32
- **Ice VII** (sample): 3rd-order Birch-Murnaghan from Hemley et al., Nature 330 737 (1987). V₀=12.3 cm³/mol, K₀=23.7 GPa, K′=4.15

---

## Phase 2: Peak Matching Engine

### 2.1 Strain search (step 7)

For each phase, find the isotropic strain factor s that maximizes the number of matched peaks between observed and calculated d-spacings:

- **With EOS + pressure**: narrow search around s_predicted ± 0.02
- **With calibrant, no pressure**: determine pressure from calibrant first, then predict sample strain
- **No EOS**: blind search s ∈ [0.90, 1.10]

Algorithm: for each trial s, compute s·d_calc for all reflections, count observed peaks within tolerance (FWHM from resolution curve). Select s with most matches. This is brute-force and cannot diverge.

### 2.2 Multi-phase assignment

Observed peaks may belong to any phase. Strategy:
- Score each (observed_peak, phase, reflection) triple by proximity and expected intensity (F² × multiplicity)  
- Strong reflections get priority in matching
- Assign each observed peak to best-matching reflection across all phases
- Unmatched observed peaks flagged as potential unknowns

### 2.3 Lattice parameter estimation (step 8)

Given matched pairs with strain s:
- **Cubic**: single parameter a_new = s × a_cif → done
- **Non-cubic**: different Miller indices have different sensitivity to a, b, c → solve a small linear system from matched peak offsets
- Uses crystal-system-aware parameterization from existing `_build_param_vector()` logic

### 2.4 Scale + width estimation (step 9)

- **Relative scale**: ratio of observed peak height to (F² × multiplicity) for matched peaks, median across all matched peaks per phase
- **Peak widths**: median observed FWHM, mapped to instrument sigma parameters using the resolution model

---

## Phase 3: Integration & Refinement of Engine

### 3.1 Refactor `engine.py`

- Keep: `d_to_tof()`, `tof_to_d()`, `simulate_pattern()`, `_tof_profile_batch()`, `_get_crystal_system()`, `_build_param_vector()` (crystal-system-aware parameterization)
- Replace: `inspect()` — swap least-squares fitting for the peak-matching pipeline
- Keep `InspectionResult` as the output container

### 3.2 CLI

Replace the stub `cli.py` with real commands:
- `inspectrum inspect` — run the full pipeline
- `inspectrum preprocess` — background subtraction only
- `inspectrum peaks` — peak finding only

### 3.3 Export

Output formats for downstream tools:
- Mantid workspace export (for Mantid Workbench visualization)
- GSAS-II parameter file update (write optimised params back to .instprm / .EXP)

---

## What's Already Done

| Module | Status | Summary |
|--------|--------|---------|
| `models.py` | ✅ | DiffractionSpectrum, DiffractionDataset, CrystalPhase, Instrument, InspectionResult |
| `loaders.py` | ✅ | load_gsa, load_mantid_csv, load_instprm, load_cif |
| `crystallography.py` | ✅ | generate_reflections with all Laue classes, centering rules, structure factors |
| `background.py` | ✅ | Rolling-ball peak clipping with LLS, tuned for SNAP |
| `peakfinding.py` | ✅ | find_peaks_in_spectrum with auto-threshold, FWHM, centroid positions |
| `resolution.py` | ✅ | parse_resolution_curve, fwhm_at_d, recommend_parameters |
| `plotting.py` | ✅ | plot_spectrum, plot_background, plot_peak_markers, inspect_peaks (interactive) |
| `engine.py` | ⚠️ | Has simulate_pattern (keep) + least-squares inspect (replace) |
| `cli.py` | ⚠️ | Stub only |
| Tests | ✅ | 150+ tests across 7 files |

---

## Open Questions

1. **Anisotropic strain**: For non-cubic phases at high pressure, strain may differ along a, b, c. Phase 2.3 handles this via Miller-index sensitivity, but needs experimental validation.
2. **Pressure from calibrant**: If spectrum pressure is unknown but a calibrant is present, can we first match the calibrant, derive pressure, then use EOS to predict sample strain? This is a powerful workflow but adds complexity.
3. **Multiple banks**: Current design is single-bank. Multi-bank matching (same phase, different resolution/d-range) is Phase 3+.
4. **Phase transitions at unknown conditions**: Current assumption is all present phases have CIFs provided. Detecting unexpected phases is out of scope.
