# Inspectrum Implementation Plan

**Created**: 2026-03-30  
**Updated**: 2026-04-01  
**Status**: In progress — Phase 4  
**Reference**: [project.md](project.md), [ground_truths.md](ground_truths.md)

---

## Project Summary

Inspectrum pre-inspects powder diffraction spectra to estimate good starting parameters (lattice parameters, scale factors, peak widths) for Rietveld refinement. It is designed for challenging data: high-pressure neutron diffraction with structured backgrounds and low signal-to-noise.

**Critical design constraint**: inspectrum does NOT perform Rietveld refinement. It estimates parameters by peak matching and geometric analysis. The one exception is per-phase lattice parameter refinement in 1/d² space (fitting discrete peak positions, not a full profile), which is a small well-constrained problem that cannot diverge.

---

## Pipeline Overview

```
Input                        Pre-processing              Inspection                 Output
─────                        ──────────────              ──────────                 ──────
Spectra (.gsa/.csv)    ──►   Background subtraction ──►  Peak finding         ──►  Optimised lattice params
CIF files              ──►   Generate reflections   ──►  Peak matching        ──►  Per-phase pressures
Instrument (.instprm)  ──►   Resolution analysis    ──►  Lattice refinement   ──►  Diagnostic plots
Phase descriptions     ──►   EOS-predicted strain   ──►  (narrows search)     ──►  Refinement report
```

### Pipeline Steps (in order)

| # | Step | Input | Output | Module | Status |
|---|------|-------|--------|--------|--------|
| 1 | Load data | Files on disk | DiffractionSpectrum, CrystalPhase, Instrument | `loaders.py` | ✅ Done |
| 2 | Load phase descriptions | JSON file | PhaseDescription with EOS | `loaders.py` | ✅ Done |
| 3 | Background subtraction | Raw spectrum | Background + peak signal | `background.py` | ✅ Done |
| 4 | Peak finding | Peak signal + resolution | PeakTable (positions, FWHM, heights) | `peakfinding.py` | ✅ Done |
| 5 | Generate expected reflections | CrystalPhase + d-range | Reflection list (d, hkl, F², mult) | `crystallography.py` | ✅ Done |
| 6 | EOS strain prediction | PhaseDescription + pressure | Predicted strain s = (V/V₀)^(1/3) | `eos.py` | ✅ Done |
| 7 | Pressure sweep + peak matching | Observed peaks + reflections + EOS | MatchResult (per-phase matched peaks) | `matching.py` | ✅ Done |
| 8 | Lattice parameter refinement | Matched peaks per phase | Refined lattice params + EOS pressure | `lattice.py` | ✅ Done |

---

## Phase 1: Schema & EOS — ✅ Complete

### 1.1 New data models in `models.py`

- **`EquationOfState`**: EOS type, order, V₀ (ų/cell), K₀ (GPa), K′, source citation, extra params
- **`SampleConditions`**: pressure (GPa), temperature (K) — both optional  
- **`PhaseDescription`**: wraps a CIF path + role (calibrant/sample) + reference conditions + EOS + stability range

### 1.2 JSON serialization + loader

- `load_phase_descriptions(json_path)` with V₀ unit conversion (ų/atom, cm³/mol → ų/cell)

### 1.3 Test data

- Tungsten (calibrant): Vinet EOS, V₀=31.724 ų/cell
- Ice VII (sample): 3rd-order Birch-Murnaghan, V₀=40.849 ų/cell

---

## Phase 2: Peak Matching Engine — ✅ Complete

### 2.1 Pressure sweep (step 7)

`sweep_pressure()` in `matching.py`: two-pass coarse+fine grid search over shared pressure. At each trial P, per-phase strains from EOS, Gaussian-weighted scoring by residual and F²×multiplicity. Contested peaks resolved by smallest |residual|.

### 2.2 Multi-phase assignment

All phases matched simultaneously at shared pressure. Unmatched peaks tracked. Spurious peak discrimination via resolution floor (0.75× FWHM) and 5σ prominence threshold.

### 2.3 Lattice parameter refinement (step 8)

`refine_lattice_parameters()` in `lattice.py`: per-phase least-squares in 1/d² space for all 7 crystal systems. Weak-peak exclusion. EOS-derived per-phase pressures from refined cell volumes. Fixes snapwrap `cubic_d2Inv` bug.

### 2.4 Scale + width estimation — Superseded

Originally planned as a separate step. In practice:
- **Scale**: not needed — inspectrum's purpose is lattice parameter estimation; Rietveld handles scale refinement
- **Peak widths**: observed FWHM is already captured in `PeakTable` and used as fitting weights in lattice refinement; instrument width params are handled by GSAS-II

---

## Phase 3: Engine Refactor — ✅ Complete

### 3.1 Refactor `engine.py`

Replaced the old least-squares `inspect()` with the new pipeline:

- **Kept**: `d_to_tof()`, `tof_to_d()` (coordinate transforms used by loaders and plotting)
- **Replaced**: `inspect()` → orchestrates background → peaks → reflections → sweep → refine
- **Removed**: `simulate_pattern()`, `_tof_profile_batch()`, `_build_param_vector()`, `_unpack_params()`, `_get_crystal_system()`, `_chi_squared()`, `_build_result()` — profile simulation is GSAS-II's job
- **Output**: `InspectionResult` updated to carry `MatchResult` + `list[LatticeRefinementResult]` + `PeakTable` + `sweep_pressure_gpa`
- **Tests**: 14 new engine tests (5 coordinate transform + 9 pipeline integration), 314 total passing

---

## Phase 4: Interfacing — Planned

### 4.1 CLI

Replace the stub `cli.py` with real commands:
- `inspectrum inspect` — run the full pipeline on a spectrum + phase descriptions
- `inspectrum preprocess` — background subtraction only
- `inspectrum peaks` — peak finding only
- `inspectrum report` — generate refinement report from saved results

### 4.2 SNAPWrap integration

Wire inspectrum into the SNAPWrap workflow so it can be called from within SNAP reduction scripts. inspectrum provides the lattice parameter estimation; SNAPWrap handles Mantid-specific I/O and orchestration.

### 4.3 Further interfacing TBD

Additional integration points to be defined based on user needs (e.g. Mantid Workbench plugin, web dashboard, batch processing).

---

## Backlog

Items parked for future consideration:

- **Export to GSAS-II / Mantid**: Write optimised params back to `.instprm` / `.EXP` files, or export as Mantid workspaces. Deferred until the interface layer (Phase 4) clarifies what formats are needed.
- **Anisotropic strain**: Non-cubic phases at high pressure may have different strain along a, b, c. The lattice refinement already handles this (separate params per axis), but validation on real non-cubic data is needed.
- **Pressure from calibrant**: If sample pressure is unknown but a calibrant is present, derive pressure from calibrant strain first, then predict sample strain. Currently the sweep does this implicitly (shared pressure); an explicit calibrant-first workflow could improve robustness.
- **Multiple banks**: Current design is single-bank. Multi-bank matching (same phase, different resolution/d-range) would improve peak statistics.
- **Phase transitions at unknown conditions**: Detecting unexpected phases is out of scope; current assumption is all present phases have CIFs provided.
- **Mantid workspace loader**: Read directly from a Mantid `Workspace2D` object (in-memory, no file I/O) for live-processing workflows.

---

## What's Already Done

| Module | Status | Summary |
|--------|--------|---------|
| `models.py` | ✅ | DiffractionSpectrum, CrystalPhase, Instrument, EquationOfState, PhaseDescription |
| `loaders.py` | ✅ | load_gsa, load_mantid_csv, load_instprm, load_cif, load_phase_descriptions |
| `crystallography.py` | ✅ | generate_reflections with full symop expansion, structure factors |
| `background.py` | ✅ | Rolling-ball peak clipping with LLS, reflect-padded edges |
| `peakfinding.py` | ✅ | find_peaks_in_spectrum with resolution floor + 5σ discrimination |
| `resolution.py` | ✅ | parse_resolution_curve, fwhm_at_d, recommend_parameters |
| `eos.py` | ✅ | Murnaghan, Birch-Murnaghan, Vinet; pressure_at, predicted_strain |
| `matching.py` | ✅ | sweep_pressure, sweep_strain, identify_phases, match_peaks_at_strain |
| `lattice.py` | ✅ | refine_lattice_parameters for all 7 crystal systems, format_refinement_report |
| `plotting.py` | ✅ | Phase match overlay with refined ticks, annotation box, inspect_peaks |
| `engine.py` | ✅ | d_to_tof/tof_to_d + inspect() pipeline orchestrator (49 stmts, 98% coverage) |
| `cli.py` | ⚠️ | Stub only |
| Tests | ✅ | 314 tests across 13 files |
