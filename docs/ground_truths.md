# Ground Truths

This file captures key findings, decisions, and verified facts discovered during development. It serves as a persistent knowledge base that AI assistants (like GitHub Copilot) and developers can reference across sessions.

**Why this matters:** AI assistants don't remember previous conversations. By recording important discoveries here, you ensure that context isn't lost between sessions. When Copilot reads this file, it can make better suggestions based on what's already been learned about your project.

## How to Use This File

- **Add entries as you discover important facts** — things like API quirks, configuration requirements, performance constraints, or design decisions.
- **Include the date and context** so future-you (or Copilot) understands why something was noted.
- **Link to relevant code or docs** when helpful.
- Copilot is instructed to update this file automatically when it discovers key findings during development.

## Findings

### 2026-03-11: Environment management uses pixi, not venv

The organisation uses pixi for deployment. The project uses `[tool.pixi.*]` sections in `pyproject.toml` (not a separate `pixi.toml`) for environment management. venv instructions are kept as a fallback in docs but pixi is the primary workflow.

### 2026-03-17: Pixi environment setup — TLS and Python version

- **pixi binary**: `~/.pixi/bin/pixi` (v0.62.1). Not on PATH by default.
- **TLS issue**: ORNL network uses a TLS-intercepting proxy causing `invalid peer certificate: UnknownIssuer`. Fix: `~/.config/pixi/config.toml` with `tls-no-verify = true`. This is machine-local, not committed.
- **Python pinned to `>=3.10,<3.14`**: Without an upper bound pixi solved to Python 3.14, which broke pandas. GSAS-II supports 3.10–3.13.
- **`[tool.pixi.project]` → `[tool.pixi.workspace]`**: Pixi 0.62 deprecated `project` in favour of `workspace`.
- **Verified env**: Python 3.13.12, numpy 2.4.2, scipy 1.17.1, matplotlib 3.10.8, h5py 3.15.1, pandas 2.3.3, cryspy, inspectrum 0.1.0. All 115 tests pass.
- **Pixi tasks**: `pixi run test` (pytest), editable install via `inspectrum = { path = ".", editable = true }` in pypi-dependencies.

### 2026-03-11: GSAS-II is a prerequisite, not a managed dependency

GSAS-II has a complex build process (Fortran compilers, platform-specific tooling) and cannot be installed via pip or conda. It must be installed separately. inspectrum's dependency version floors are aligned with GSAS-II's `pixi/pixi.toml` to ensure compatibility:

- Python: >=3.10 (GSAS-II supports 3.10–3.13)
- numpy: >=2.2.0 (GSAS-II uses >=2.2.1)
- scipy: >=1.15.0
- matplotlib: >=3.10.0
- h5py: >=3.12.0

In the pixi config, we use `matplotlib-base` (conda-forge package without Qt/Tk backends) since the backend choice is left to the user/GSAS-II environment.

### 2026-03-11: Target platforms are macOS and Linux only

Platforms: `osx-arm64`, `osx-64`, `linux-64`. No Windows support currently planned.

### 2026-03-16: CIF parsing / crystallographic library choice — cryspy

Chose `cryspy` (>=0.10.0, MIT license) over `crystals` (GPL-3.0, license-incompatible with BSD-3) and `gemmi` (good CIF reader but no powder diffraction tools). Reasons:

- **Neutron-specific**: built for polarized neutron diffraction, supports powder and single-crystal, has TOF support.
- **CIF support**: uses the core CIF dictionary of IUCr for data exchange.
- **MIT license**: compatible with inspectrum's BSD-3-Clause license.
- **Active**: v0.10.0 released Jan 2026, uses pixi for its own dev workflow.
- **PyPI only**: not on conda-forge, so added as a `[tool.pixi.pypi-dependencies]` entry.

cryspy is not on conda-forge so it's declared in pixi's pypi-dependencies section rather than conda dependencies.

### 2026-03-16: Test data — SNAP high-pressure neutron diffraction

Test fixtures are in `tests/test_data/`:

- **CIF files**: tungsten (Im-3m, a=3.16475 Å) and ice VII (Pn-3m, a=3.3891 Å) from ICSD.
- **Diffraction data**: 6 SNAP spectra (runs 059056–059063) as Mantid-exported CSV, each a single bank (bank 0). Format: d-spacing (Å) vs counts/µA·hr with uncertainties. ~863 points, d-range ~0.79–2.50 Å. Pressure increases across the series → lattice parameters shift.
- **Instrument params**: GSAS-II `.instprm` files (Type:PNT). Key params: difC=5218.45, 2θ=85.3035°, fltPath=15.5806 m. All 6 files have identical instrument params (same bank).

### 2026-03-16: Data model design decisions

- `DiffractionSpectrum` holds one bank's data (x, y, e arrays). `DiffractionDataset` is a list of spectra (e.g. pressure series).
- `CrystalPhase` holds lattice params, space group, atom sites, and a scale factor. Populated from CIF.
- `Instrument` mirrors GSAS-II .instprm key-value pairs (PNT type). Designed for one bank per instance; multi-bank instruments use multiple Instrument objects.
- `InspectionResult` holds deep copies of optimised phases + instrument. Originals are never modified.
- SNAP can output 1–6 banks per measurement with varying resolution and d-range. Testing with single bank; design accommodates multi-bank.

### 2026-03-16: Data loaders — formats and implementation details

Implemented in `src/inspectrum/loaders.py`. Four loaders:

- **`load_gsa(filepath)`** → `list[DiffractionSpectrum]`: Parses GSAS FXYE format (`.gsa`). Handles JSON metadata headers, comment lines (`#`), and `BANK` directives. FXYE = fixed-width X, Y, E columns. x_unit="TOF". Supports multi-bank files (one spectrum per BANK line).
- **`load_mantid_csv(filepath)`** → `list[DiffractionSpectrum]`: Parses Mantid-exported CSV. Extracts x-axis unit from header comment (`# X , Y , E`). Typically d-Spacing (Å). Comma-separated numerical data.
- **`load_instprm(filepath)`** → `Instrument`: Parses GSAS-II `.instprm` key:value format. Maps file keys (e.g. `difC`, `sig-1`, `2-theta`) to `Instrument` dataclass fields. Captures the `pdabc` multi-line absorption table block in `raw_params`.
- **`load_cif(filepath)`** → `CrystalPhase`: Uses pycifstar (`to_data(filepath)`) to parse CIF. Extracts cell parameters with `_parse_cif_number()` to strip CIF uncertainty notation (e.g. `3.16475(20)` → `3.16475`). Reads space group (H-M name + number), atom sites from CIF loops.

**pycifstar API notes**: `to_data(filepath)` returns a `Data` object. Access values via `data['_key'].value` (returns string). Loop data via `data.loops` — each loop has `.names` (list of column names) and you index with `loop['_column_name']` to get list of values.

### 2026-03-16: GSA ↔ CSV consistency

GSA (TOF) and CSV (d-spacing) files for the same SNAP run contain the same number of points (863). The relationship is: d = TOF / difC (approximately, for the linear term). This can be used for cross-validation.

### 2026-03-16: TODO — Mantid workspace loader

Future enhancement: add a loader that reads directly from a Mantid `Workspace2D` object (in-memory, no file I/O). This is for live-processing workflows where inspectrum is called from within Mantid scripts. Logged for Phase 3+.

### 2026-03-16: CRITICAL DESIGN DECISION — inspectrum does NOT refine

inspectrum's purpose is to **estimate good starting parameters** for Rietveld refinement, *without* performing a refinement itself. Rietveld refinements diverge when initial parameters are too far from the true values — inspectrum solves this by inspecting the spectrum directly.

**What inspectrum is NOT**: a profile-fitting optimizer or least-squares refinement engine. Do not use `scipy.optimize.least_squares` or similar to fit a calculated pattern to observed data — that *is* refinement and will suffer the same divergence problems.

**What inspectrum IS**: a peak-matching and parameter estimation tool. The pipeline:

1. **Calculate expected peak positions + intensities** from CIF lattice parameters and structure factors (crystallography module)
2. **Pre-process**: remove structured background (rolling ball technique) to cleanly expose peaks
3. **Find peaks** in the cleaned observed spectrum (peak detection)
4. **Match observed peaks to calculated peaks** — this is a 1D array matching problem (experimental mini-phase, needs experimentation)
5. **Estimate lattice parameters** from the offsets between matched peak positions (geometric/analytical)
6. **Estimate scale and peak widths** from observed peak heights and shapes

**Parameters to estimate at this stage**: lattice parameters (a, b, c, α, β, γ), relative scale factors (per phase), peak widths. These are sufficient to bootstrap a Rietveld refinement.

**Inspection via Mantid Workbench**: users will visually inspect intermediate results (background-subtracted spectra, found peaks, matched peaks) by loading them into Mantid Workbench, which has all the plotting/analysis tools needed. inspectrum does not need its own visualization.

### 2026-03-16: Crystallography module — cryspy building blocks

`src/inspectrum/crystallography.py` uses cryspy's low-level functions:

- **`cryspy.calc_inverse_d_by_hkl_abc_angles(h, k, l, a, b, c, α, β, γ)`** — returns 1/d for given Miller indices and cell params. Angles in **radians**.
- **`cryspy.get_scat_length_neutron(symbol)`** — returns complex bound coherent neutron scattering length (fm). E.g. W → 0.486, O → 0.5803, H → −0.3739.
- **`cryspy.get_crystal_system_by_it_number(sg_number)`** — returns crystal system string.
- **`cryspy.tof_Jorgensen(alpha, beta, sigma, time, time_hkl)`** — TOF peak profile (Ikeda-Carpenter convolved with Gaussian). Available but NOT used in the inspection pipeline (no profile fitting).

Key implementation details:
- Centering translation vectors (`_CENTERING_VECTORS` dict) applied to structure factor calculation for I, F, C, A, B, R lattices
- CIF atom site coordinates may contain uncertainty strings (`"0.202(3)"`) — `_parse_cif_number()` strips these
- Multiplicities computed from Laue class symmetry by enumerating equivalent reflections
- Reflections merged by d-spacing tolerance to remove duplicates from positive-hkl enumeration

### 2026-03-16: Peak finding module — tuning insights from SNAP data

`src/inspectrum/peakfinding.py` wraps `scipy.signal.find_peaks` with defaults tuned for neutron TOF powder diffraction.

**Auto-threshold strategy**: prominence threshold = std of the lower quartile of the peak signal. After background subtraction, peak-free regions cluster in the lower 25% — their std is a robust noise estimate. On SNAP data: noise σ ≈ 3.7, giving a prominence threshold of ~3.7 counts. This was chosen after testing several strategies:
- 3 × MAD was too aggressive (MAD=6.8, threshold=20.4 → 0 peaks found) because the residual background pedestal inflates the MAD
- Lower-quartile σ adapts well to the actual noise floor

**Width filter is the key discriminant**: real diffraction peaks have FWHM ≈ 0.01–0.06 Å (5–30 data points at SNAP resolution). Noise spikes have FWHM < 0.005 Å (1–3 points). The `min_width_pts=5` default effectively separates signal from noise.

**SNAP059056 results**: 9 peaks found (with auto-threshold + width filter). These correspond to tungsten and ice VII reflections. Offsets from ambient CIF d-spacings are 0.02–0.10 Å, consistent with ~3–5% lattice compression under high pressure.

**Diagnostic plotting**: `src/inspectrum/plotting.py` provides `plot_spectrum()`, `plot_background()`, and `plot_peak_markers()` — thin matplotlib wrappers that return `(fig, ax)` for one-liner visual inspection. `plot_peak_markers()` accepts a `PeakTable` directly.

### 2026-03-17: Background edge divergence — fixed with reflect-padding

The original peak-clipping implementation skipped points within `w` of the array edges (`if i < w: continue`). These untouched edge points kept their pre-clipping values, and after inverse-LLS + global normalization they diverged strongly from the real data.

**Fix**: Reflect-pad the working array by `win_size + 1` points on each side before clipping (like `np.pad(mode='reflect')`). This gives full windows at the edges. After clipping, strip the padding. The background now faithfully follows the data to the end points.

### 2026-03-17: Tuned parameters for SNAP high-pressure data

Tuned via `scripts/tune_peaks.py` interactive slider UI:
- `win_size=4` (was 40): SNAP spectra have structured, gnarly backgrounds that need tight tracking
- `smoothing=1.0` (was 5.0): less pre-smoothing preserves background features
- `min_prominence=4.0`: explicit threshold works better than auto for this data
- `min_width_pts=6` (was 5): slightly wider filter for SNAP resolution

These are stored as defaults in `scripts/tune_peaks.py`. The `estimate_background()` API defaults remain general-purpose (`win_size=40`, `smoothing=5.0`).

### 2026-03-17: Instrument resolution (pdabc) parsing

The instprm `pdabc` block contains 5 columns: d-spacing, TOF, 0, 0, σ_TOF. σ_TOF is the Gaussian sigma in TOF µs measured on NIST strain-free Si. NaN rows at low-d and high-d edges have no calibration data.

**Conversion**: `FWHM_d = 2·√(2·ln2) · σ_TOF / DIFC`

**SNAP Bank 0 resolution** (DIFC=5218.45):
- d=0.8 Å: FWHM ≈ 0.0045 Å (4.2 pts)
- d=1.0 Å: FWHM ≈ 0.0079 Å (6.0 pts)
- d=1.5 Å: FWHM ≈ 0.0174 Å (8.7 pts)
- d=2.0 Å: FWHM ≈ 0.0228 Å (8.6 pts)
- d=2.5 Å: FWHM ≈ 0.0307 Å (9.2 pts)

**Key insight**: `recommend_parameters()` derives background/peakfinding params from resolution. The `win_size` recommendation (3× max FWHM) gives 28 — too large for SNAP's structured backgrounds but reasonable for smoother instruments. Manual tuning (win_size=4) is needed for gnarly backgrounds.
