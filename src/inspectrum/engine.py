"""
Core inspection engine for inspectrum.

Provides pattern simulation and parameter optimisation to obtain
good initial values for Rietveld refinement.  The main entry point
is :func:`inspect`.

Typical usage::

    from inspectrum.loaders import load_gsa, load_instprm, load_cif
    from inspectrum.engine import inspect

    spectra = load_gsa("data.gsa")
    instrument = load_instprm("data.instprm")
    phases = [load_cif("tungsten.cif"), load_cif("sample.cif")]

    result = inspect(
        spectra,
        phases,
        instrument,
        optimize_crystal=True,
        optimize_instrument=False,
    )

    # Examine in Mantid Workbench: result contains calculated pattern
    # as a DiffractionSpectrum alongside optimised parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import erfc

from inspectrum.crystallography import generate_reflections
from inspectrum.models import (
    CrystalPhase,
    DiffractionDataset,
    DiffractionSpectrum,
    InspectionResult,
    Instrument,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def d_to_tof(d: NDArray[np.float64], instrument: Instrument) -> NDArray[np.float64]:
    """Convert d-spacing (Å) to time-of-flight (µs).

    Uses the GSAS-II relationship:
        TOF = difA·d² + difC·d + Zero

    Args:
        d: Array of d-spacing values in Angstroms.
        instrument: Instrument with difA, difC, zero parameters.

    Returns:
        Array of TOF values in microseconds.
    """
    return instrument.difA * d**2 + instrument.difC * d + instrument.zero


def tof_to_d(tof: NDArray[np.float64], instrument: Instrument) -> NDArray[np.float64]:
    """Convert time-of-flight (µs) to d-spacing (Å).

    Inverts the GSAS-II relationship.  When difA ≈ 0 (common case)
    this simplifies to d = (TOF - Zero) / difC.

    Args:
        tof: Array of TOF values in microseconds.
        instrument: Instrument with difA, difC, zero parameters.

    Returns:
        Array of d-spacing values in Angstroms.
    """
    if abs(instrument.difA) < 1e-12:
        return (tof - instrument.zero) / instrument.difC

    # Solve quadratic: difA·d² + difC·d + (Zero - TOF) = 0
    a = instrument.difA
    b = instrument.difC
    c = instrument.zero - tof
    discriminant = b**2 - 4 * a * c
    discriminant = np.maximum(discriminant, 0.0)
    return (-b + np.sqrt(discriminant)) / (2 * a)


def simulate_pattern(
    tof_axis: NDArray[np.float64],
    phases: list[CrystalPhase],
    instrument: Instrument,
    d_min: float | None = None,
    d_max: float | None = None,
) -> NDArray[np.float64]:
    """Simulate a calculated diffraction pattern on the given TOF axis.

    Generates reflections for each phase, computes TOF peak profiles
    using the instrument parameters, and sums all contributions.

    Args:
        tof_axis: Observed TOF values (µs) at which to evaluate
            the calculated pattern.
        phases: Crystal phases contributing to the pattern.
        instrument: Instrument description with profile parameters.
        d_min: Minimum d-spacing for reflection generation (Å).
            Defaults to value derived from tof_axis minimum.
        d_max: Maximum d-spacing for reflection generation (Å).
            Defaults to value derived from tof_axis maximum.

    Returns:
        Calculated intensity array on the same tof_axis grid.
    """
    if d_min is None or d_max is None:
        d_from_tof = tof_to_d(tof_axis, instrument)
        if d_min is None:
            d_min = float(np.min(d_from_tof)) * 0.95
        if d_max is None:
            d_max = float(np.max(d_from_tof)) * 1.05

    y_calc = np.zeros_like(tof_axis, dtype=np.float64)

    for phase in phases:
        reflections = generate_reflections(phase, d_min, d_max)
        if not reflections:
            continue

        # Collect all peak parameters
        d_values = np.array([r["d"] for r in reflections])
        multiplicities = np.array([r["multiplicity"] for r in reflections])
        f_sq_values = np.array([r["F_sq"] for r in reflections])

        # Peak centres in TOF
        tof_hkl = d_to_tof(d_values, instrument)

        # d-dependent profile parameters
        alpha_arr = np.full_like(d_values, instrument.alpha)
        beta_arr = (
            instrument.beta_0
            + instrument.beta_1 / d_values**4
            + instrument.beta_q / d_values**2
        )
        sigma_sq = (
            instrument.sig_0
            + instrument.sig_1 * d_values**2
            + instrument.sig_2 * d_values**4
            + instrument.sig_q / d_values**2
        )
        sigma_arr = np.sqrt(np.maximum(sigma_sq, 1e-10))

        # Peak intensities: scale × multiplicity × |F|²
        intensities = phase.scale * multiplicities * f_sq_values

        # Evaluate TOF profiles for all reflections simultaneously
        profiles = _tof_profile_batch(
            alpha_arr, beta_arr, sigma_arr, tof_axis, tof_hkl
        )

        # Sum weighted peak contributions: profiles shape is (n_tof, n_refl)
        y_calc += profiles @ intensities

    return y_calc


def inspect(
    spectra: list[DiffractionSpectrum],
    crystal_phases: list[CrystalPhase],
    instrument: Instrument,
    optimize_crystal: bool = True,
    optimize_instrument: bool = False,
) -> InspectionResult:
    """Pre-inspect diffraction spectra to obtain good initial values.

    Fits a calculated multi-phase pattern to the observed spectrum
    by adjusting lattice parameters (and optionally instrument profile
    parameters).  Returns an :class:`InspectionResult` with deep
    copies of the optimised phases, instrument, and a calculated
    pattern.

    For multiple spectra, the first spectrum is used for fitting.
    This will be extended to simultaneous multi-spectrum fitting in
    a future version.

    Args:
        spectra: Observed diffraction spectra (TOF x-axis).
        crystal_phases: Crystal phases with initial lattice parameters.
        instrument: Instrument description with profile parameters.
        optimize_crystal: If True, optimise lattice parameters and
            scale factors.
        optimize_instrument: If True, also optimise instrument
            profile parameters (sig_1, beta_0, beta_1).

    Returns:
        InspectionResult with optimised parameters and calculated pattern.

    Raises:
        ValueError: If no spectra or phases are provided.
    """
    if not spectra:
        raise ValueError("At least one spectrum is required")
    if not crystal_phases:
        raise ValueError("At least one crystal phase is required")

    # Work with deep copies to avoid mutating originals
    opt_phases = [p.copy() for p in crystal_phases]
    opt_instrument = instrument.copy()
    spectrum = spectra[0]

    # Ensure we're working in TOF
    if spectrum.x_unit != "TOF":
        tof_axis = d_to_tof(spectrum.x, instrument)
    else:
        tof_axis = spectrum.x.copy()

    y_obs = spectrum.y
    e_obs = spectrum.e

    # Build parameter vector and bounds
    params, bounds_lo, bounds_hi, param_names = _build_param_vector(
        opt_phases, opt_instrument, optimize_crystal, optimize_instrument
    )

    if len(params) == 0:
        # Nothing to optimise — just simulate
        y_calc = simulate_pattern(tof_axis, opt_phases, opt_instrument)
        chi_sq = _chi_squared(y_obs, y_calc, e_obs)
        return _build_result(
            opt_phases, opt_instrument, spectrum, tof_axis, y_calc, chi_sq
        )

    # Estimate a flat background as median of lowest 20% of intensities
    sorted_y = np.sort(y_obs)
    background = float(np.median(sorted_y[: max(1, len(sorted_y) // 5)]))

    def residual_fn(p: NDArray[np.float64]) -> NDArray[np.float64]:
        _unpack_params(p, opt_phases, opt_instrument, param_names)
        y_calc = simulate_pattern(tof_axis, opt_phases, opt_instrument)
        y_model = y_calc + background
        return (y_obs - y_model) / e_obs

    result = least_squares(
        residual_fn,
        params,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=200 * len(params),
        ftol=1e-8,
        xtol=1e-8,
    )

    # Unpack final parameters
    _unpack_params(result.x, opt_phases, opt_instrument, param_names)

    # Generate final calculated pattern
    y_calc = simulate_pattern(tof_axis, opt_phases, opt_instrument)
    chi_sq = _chi_squared(y_obs, y_calc + background, e_obs)

    return _build_result(
        opt_phases, opt_instrument, spectrum, tof_axis,
        y_calc + background, chi_sq,
        metadata={
            "optimizer_nfev": result.nfev,
            "optimizer_cost": float(result.cost),
            "optimizer_message": result.message,
            "background_estimate": background,
            "param_names": param_names,
            "param_values": result.x.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# TOF peak profile
# ---------------------------------------------------------------------------


def _tof_profile_batch(
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
    sigma: NDArray[np.float64],
    tof_axis: NDArray[np.float64],
    tof_hkl: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the back-to-back exponential × Gaussian TOF profile.

    Implements the Jorgensen (Ikeda-Carpenter) profile for all
    reflections simultaneously.

    Args:
        alpha: Rising-edge parameter for each reflection (n_refl,).
        beta: Falling-edge parameter for each reflection (n_refl,).
        sigma: Gaussian sigma for each reflection (n_refl,).
        tof_axis: TOF values at which to evaluate (n_tof,).
        tof_hkl: Peak centre TOF for each reflection (n_refl,).

    Returns:
        Profile matrix of shape (n_tof, n_refl).
    """
    # Reshape for broadcasting: tof_axis (n_tof, 1), params (1, n_refl)
    t = tof_axis[:, np.newaxis]       # (n_tof, 1)
    t0 = tof_hkl[np.newaxis, :]      # (1, n_refl)
    a = alpha[np.newaxis, :]          # (1, n_refl)
    b = beta[np.newaxis, :]           # (1, n_refl)
    s = sigma[np.newaxis, :]          # (1, n_refl)

    delta = t - t0  # (n_tof, n_refl)

    # Normalisation
    norm = 0.5 * a * b / (a + b)

    # Arguments for erfc components
    s2 = s**2
    u = 0.5 * a * (a * s2 + 2 * delta)
    v = 0.5 * b * (b * s2 - 2 * delta)
    y = (a * s2 + delta) / (np.sqrt(2.0) * s)
    z = (b * s2 - delta) / (np.sqrt(2.0) * s)

    # Clip exponents to avoid overflow
    u = np.clip(u, None, 500.0)
    v = np.clip(v, None, 500.0)

    profile = norm * (np.exp(u) * erfc(y) + np.exp(v) * erfc(z))

    return profile  # (n_tof, n_refl)


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------


def _build_param_vector(
    phases: list[CrystalPhase],
    instrument: Instrument,
    optimize_crystal: bool,
    optimize_instrument: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[str],
]:
    """Build the flat parameter vector, bounds, and name list.

    Returns:
        (params, bounds_lo, bounds_hi, param_names)
    """
    params: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    names: list[str] = []

    if optimize_crystal:
        for i, phase in enumerate(phases):
            crystal_system = _get_crystal_system(phase)

            # Lattice parameters — constrain based on crystal system
            if crystal_system == "cubic":
                params.append(phase.a)
                lo.append(phase.a * 0.95)
                hi.append(phase.a * 1.05)
                names.append(f"phase_{i}_a")
            elif crystal_system in ("tetragonal", "hexagonal", "trigonal"):
                params.extend([phase.a, phase.c])
                lo.extend([phase.a * 0.95, phase.c * 0.95])
                hi.extend([phase.a * 1.05, phase.c * 1.05])
                names.extend([f"phase_{i}_a", f"phase_{i}_c"])
            elif crystal_system == "orthorhombic":
                params.extend([phase.a, phase.b, phase.c])
                lo.extend([phase.a * 0.95, phase.b * 0.95, phase.c * 0.95])
                hi.extend([phase.a * 1.05, phase.b * 1.05, phase.c * 1.05])
                names.extend([
                    f"phase_{i}_a", f"phase_{i}_b", f"phase_{i}_c"
                ])
            else:
                # Monoclinic / triclinic — all 6 params
                params.extend([
                    phase.a, phase.b, phase.c,
                    phase.alpha, phase.beta, phase.gamma,
                ])
                lo.extend([
                    phase.a * 0.95, phase.b * 0.95, phase.c * 0.95,
                    phase.alpha - 2.0, phase.beta - 2.0, phase.gamma - 2.0,
                ])
                hi.extend([
                    phase.a * 1.05, phase.b * 1.05, phase.c * 1.05,
                    phase.alpha + 2.0, phase.beta + 2.0, phase.gamma + 2.0,
                ])
                names.extend([
                    f"phase_{i}_a", f"phase_{i}_b", f"phase_{i}_c",
                    f"phase_{i}_alpha", f"phase_{i}_beta", f"phase_{i}_gamma",
                ])

            # Scale factor — always optimised
            params.append(phase.scale)
            lo.append(0.0)
            hi.append(phase.scale * 100.0)
            names.append(f"phase_{i}_scale")

    if optimize_instrument:
        # Only optimise a few key profile parameters
        params.extend([instrument.sig_1, instrument.beta_0, instrument.beta_1])
        lo.extend([0.0, 0.0, 0.0])
        hi.extend([instrument.sig_1 * 10.0, 1.0, 1.0])
        names.extend(["inst_sig_1", "inst_beta_0", "inst_beta_1"])

    return (
        np.array(params, dtype=np.float64),
        np.array(lo, dtype=np.float64),
        np.array(hi, dtype=np.float64),
        names,
    )


def _unpack_params(
    p: NDArray[np.float64],
    phases: list[CrystalPhase],
    instrument: Instrument,
    param_names: list[str],
) -> None:
    """Unpack the flat parameter vector back into model objects.

    Modifies *phases* and *instrument* in place.
    """
    for i, name in enumerate(param_names):
        val = float(p[i])
        if name.startswith("phase_"):
            parts = name.split("_")
            phase_idx = int(parts[1])
            attr = "_".join(parts[2:])

            phase = phases[phase_idx]
            if attr == "a":
                phase.a = val
                crystal_system = _get_crystal_system(phase)
                if crystal_system == "cubic":
                    phase.b = val
                    phase.c = val
                elif crystal_system in ("tetragonal", "hexagonal", "trigonal"):
                    phase.b = val
            elif attr == "b":
                phase.b = val
            elif attr == "c":
                phase.c = val
            elif attr == "alpha":
                phase.alpha = val
            elif attr == "beta":
                phase.beta = val
            elif attr == "gamma":
                phase.gamma = val
            elif attr == "scale":
                phase.scale = val
        elif name == "inst_sig_1":
            instrument.sig_1 = val
        elif name == "inst_beta_0":
            instrument.beta_0 = val
        elif name == "inst_beta_1":
            instrument.beta_1 = val


def _get_crystal_system(phase: CrystalPhase) -> str:
    """Get the crystal system string from the space group number."""
    if phase.space_group_number > 0:
        return str(cryspy.get_crystal_system_by_it_number(
            phase.space_group_number
        ))
    return "triclinic"


# ---------------------------------------------------------------------------
# Chi-squared and result building
# ---------------------------------------------------------------------------


def _chi_squared(
    y_obs: NDArray[np.float64],
    y_calc: NDArray[np.float64],
    e_obs: NDArray[np.float64],
) -> float:
    """Reduced chi-squared: Σ((y_obs - y_calc)² / e²) / (N - 1)."""
    n = len(y_obs)
    if n <= 1:
        return 0.0
    return float(np.sum(((y_obs - y_calc) / e_obs) ** 2) / (n - 1))


def _build_result(
    phases: list[CrystalPhase],
    instrument: Instrument,
    spectrum: DiffractionSpectrum,
    tof_axis: NDArray[np.float64],
    y_calc: NDArray[np.float64],
    chi_sq: float,
    metadata: dict[str, Any] | None = None,
) -> InspectionResult:
    """Construct the InspectionResult from optimised models."""
    # Create a calculated-pattern spectrum for inspection in Mantid
    calc_spectrum = DiffractionSpectrum(
        x=tof_axis.copy(),
        y=y_calc.copy(),
        e=np.zeros_like(y_calc),
        x_unit="TOF",
        y_unit=spectrum.y_unit,
        label=f"calculated_{spectrum.label}",
        bank=spectrum.bank,
    )

    # Also provide the difference spectrum
    diff_y = spectrum.y - y_calc
    diff_spectrum = DiffractionSpectrum(
        x=tof_axis.copy(),
        y=diff_y,
        e=spectrum.e.copy(),
        x_unit="TOF",
        y_unit=spectrum.y_unit,
        label=f"difference_{spectrum.label}",
        bank=spectrum.bank,
    )

    return InspectionResult(
        crystal_phases=[p.copy() for p in phases],
        instrument=instrument.copy(),
        processed_spectra=DiffractionDataset(
            spectra=[calc_spectrum, diff_spectrum],
            label="inspect_output",
        ),
        chi_squared=chi_sq,
        metadata=metadata or {},
    )


# Lazy import to avoid circular reference at module level
import cryspy  # noqa: E402
