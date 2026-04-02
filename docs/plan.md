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

## Phase 4: PyQt5 UI Widget — In Progress

**Goal**: Interactive GUI launched from Mantid Workbench (or standalone) for
spectrum inspection with pressure/temperature controls, CIF loading, and
results display.

**Design pattern**: Follows SNAPWrap's CalibrationManager architecture
(`/Users/66j/Documents/ORNL/code/SNAPWrap/src/snapwrap/calibrationManager/`).

**Deployment**: Dev in snapwrap pixi env + inspectrum deps → test on RHEL9
analysis cluster with .nxs datasets → users launch from same Workbench as
snapwrap.

### 4.1 UI Package — ✅ Scaffold Complete

Seven files in `src/inspectrum/ui/`:

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `__init__.py` | ~70 | `show()` (Workbench via QAppThreadCall) + `show_standalone()` (dev mode) | ✅ |
| `model.py` | ~290 | Pure-Python model: phase mgmt, data loading, serialization, pipeline execution | ✅ |
| `worker.py` | ~45 | `InspectionWorker(QObject)` with finished/error signals for background thread | ✅ |
| `dataPanel.py` | ~210 | File/workspace source toggle, bank selector, P min/max, temperature | ✅ |
| `phasePanel.py` | ~300 | CIF list with drag-drop, EOS editor (type/V₀/K₀/K′), stability range, save/load JSON | ✅ |
| `resultsPanel.py` | ~175 | Embedded matplotlib (`FigureCanvasQTAgg`) reusing `plot_phase_matches()`, summary table | ✅ |
| `mainWindow.py` | ~230 | `InspectrumWindow(QDialog)` — splitter layout, Run/Clear, QThread worker, progress bar | ✅ |

**Key patterns used**:
- `qtpy` for Qt bindings (same as CalibrationManager)
- `QAppThreadCall` for safe launch from Workbench script console
- `QThread` + `QObject` worker for pipeline execution (~3 sec on SNAP data)
- `FigureCanvasQTAgg` + `NavigationToolbar2QT` for plot embedding
- Pure-Python model layer (Qt-agnostic, testable with pytest)

### 4.2 Next Steps — Testing & Polish

| # | Task | Status |
|---|------|--------|
| 1 | **Test launch** on analysis cluster (RHEL9 with Mantid Workbench) | ✅ Done (2026-04-02) |
| 2 | **End-to-end pipeline test** in Workbench: load spectrum + instprm + CIFs + EOS → Run → results | ✅ Done (2026-04-02) |
| 3 | **Visual polish**: field validation, sensible defaults, error messages, UX improvements | Not started |
| 4 | **Engine metadata for plotting**: bg_subtracted, spectrum, phase_reflections stored in result.metadata | ✅ Done |
| 5 | **Manual phase definition** (define lattice params without CIF) | Deferred to v2 |

### 4.3 Launch Instructions

**From Mantid Workbench (inside snapwrap pixi env):**

The dev environment uses the snapwrap fork at
`/SNS/SNAP/shared/Malcolm/code/forks/SNAPWrap` with `cryspy` added to its
`[tool.pixi.pypi-dependencies]`. Setup:

1. Activate the snapwrap pixi shell (from the fork directory)
2. Launch workbench: `python -m workbench`
3. In the Workbench script console:
```python
import sys
sys.path.insert(0, "/SNS/SNAP/shared/Malcolm/code/inspectrum/src")
from inspectrum.ui import show
show()
```

**Standalone (no Mantid required):**
```python
from inspectrum.ui import show_standalone
show_standalone()
```

### 4.4 CLI (Future)

Replace the stub `cli.py` with real commands:
- `inspectrum inspect` — run the full pipeline on a spectrum + phase descriptions
- `inspectrum preprocess` — background subtraction only
- `inspectrum peaks` — peak finding only
- `inspectrum report` — generate refinement report from saved results

### 4.5 SNAPWrap Integration (Future)

Wire inspectrum into the SNAPWrap workflow so it can be called from within
SNAP reduction scripts. inspectrum provides the lattice parameter estimation;
SNAPWrap handles Mantid-specific I/O and orchestration.

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
| `ui/` | ✅ scaffold | 7 files: model, worker, dataPanel, phasePanel, resultsPanel, mainWindow, __init__ |
| `cli.py` | ⚠️ | Stub only |
| Tests | ✅ | 314 tests across 13 files, CI passing (ubuntu + macOS, Python 3.10–3.13) |
