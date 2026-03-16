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
