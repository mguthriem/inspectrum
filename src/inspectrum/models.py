"""
Data models for inspectrum.

Defines the core domain objects used throughout the package:

- DiffractionSpectrum: a single 1D powder-diffraction spectrum
- DiffractionDataset: a collection of spectra (e.g. a pressure series)
- CrystalPhase: crystallographic phase with lattice params and atom sites
- Instrument: instrument description (GSAS-II TOF profile parameters)
- InspectionResult: container for optimisation output
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Diffraction data
# ---------------------------------------------------------------------------

@dataclass
class DiffractionSpectrum:
    """A single 1D powder-diffraction spectrum.

    Holds d-spacing (or TOF) vs intensity data for one detector bank
    of one measurement.

    Attributes:
        x: Independent variable — d-spacing in Angstroms (or TOF in µs).
        y: Dependent variable — intensity (counts / µA·hr or similar).
        e: Uncertainties on y (same units as y).
        x_unit: Unit label for x-axis (e.g. "d-Spacing", "TOF").
        y_unit: Unit label for y-axis.
        label: Human-readable label (e.g. filename or run number).
        bank: Detector bank index (0-based).
        metadata: Arbitrary key-value metadata from the file header.

    Example:
        >>> spectrum = DiffractionSpectrum(
        ...     x=np.array([0.79, 0.80, 0.81]),
        ...     y=np.array([85.0, 88.0, 91.0]),
        ...     e=np.array([2.1, 1.9, 1.8]),
        ...     label="SNAP059056",
        ... )
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    e: NDArray[np.float64]
    x_unit: str = "d-Spacing"
    y_unit: str = "Counts per microAmp.hour"
    label: str = ""
    bank: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y) or len(self.x) != len(self.e):
            raise ValueError(
                f"x, y, e must have equal length, got "
                f"{len(self.x)}, {len(self.y)}, {len(self.e)}"
            )

    @property
    def d_min(self) -> float:
        """Minimum d-spacing (Å)."""
        return float(np.min(self.x))

    @property
    def d_max(self) -> float:
        """Maximum d-spacing (Å)."""
        return float(np.max(self.x))

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.x)

    def __repr__(self) -> str:
        return (
            f"DiffractionSpectrum(label={self.label!r}, bank={self.bank}, "
            f"n_points={self.n_points}, "
            f"d_range=[{self.d_min:.4f}, {self.d_max:.4f}] Å)"
        )


@dataclass
class DiffractionDataset:
    """A collection of 1D diffraction spectra.

    Represents a set of measurements — e.g. a pressure series where
    each entry is one spectrum from the same bank at a different
    condition.

    Attributes:
        spectra: Ordered list of DiffractionSpectrum objects.
        label: Dataset-level label (e.g. experiment name).
    """

    spectra: list[DiffractionSpectrum] = field(default_factory=list)
    label: str = ""

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)

    def __repr__(self) -> str:
        return (
            f"DiffractionDataset(label={self.label!r}, "
            f"n_spectra={self.n_spectra})"
        )

    def __getitem__(self, index: int) -> DiffractionSpectrum:
        return self.spectra[index]

    def __len__(self) -> int:
        return self.n_spectra


# ---------------------------------------------------------------------------
# Crystal phase
# ---------------------------------------------------------------------------

@dataclass
class CrystalPhase:
    """A crystallographic phase.

    Holds lattice parameters, space group, atom site information, and
    a scale factor.  Designed to be populated from a CIF file.

    For inspectrum's purposes the key optimisable parameters are the
    lattice constants (a, b, c, alpha, beta, gamma) and a relative
    scale factor.

    Attributes:
        name: Phase label (e.g. "tungsten", "ice-VII").
        a: Lattice parameter a (Å).
        b: Lattice parameter b (Å).
        c: Lattice parameter c (Å).
        alpha: Lattice angle α (degrees).
        beta: Lattice angle β (degrees).
        gamma: Lattice angle γ (degrees).
        space_group: Hermann-Mauguin space group symbol.
        space_group_number: International Tables number.
        atom_sites: List of atom site dicts, each with at minimum
            keys 'label', 'type_symbol', 'fract_x/y/z', 'occupancy'.
        scale: Relative scale factor (dimensionless).
        metadata: Extra CIF fields or user annotations.

    Example:
        >>> tungsten = CrystalPhase(
        ...     name="tungsten",
        ...     a=3.16475, b=3.16475, c=3.16475,
        ...     alpha=90.0, beta=90.0, gamma=90.0,
        ...     space_group="I m -3 m",
        ...     space_group_number=229,
        ... )
    """

    name: str = ""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    space_group: str = ""
    space_group_number: int = 0
    atom_sites: list[dict[str, Any]] = field(default_factory=list)
    scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def volume(self) -> float:
        """Unit cell volume (ų) from lattice parameters.

        Uses the general triclinic formula so it works for all
        crystal systems.
        """
        a, b, c = self.a, self.b, self.c
        al = np.radians(self.alpha)
        be = np.radians(self.beta)
        ga = np.radians(self.gamma)
        cos_al, cos_be, cos_ga = np.cos(al), np.cos(be), np.cos(ga)
        return float(
            a * b * c
            * np.sqrt(
                1
                - cos_al**2
                - cos_be**2
                - cos_ga**2
                + 2 * cos_al * cos_be * cos_ga
            )
        )

    def copy(self) -> CrystalPhase:
        """Return a deep copy of this phase."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"CrystalPhase(name={self.name!r}, "
            f"SG={self.space_group!r} #{self.space_group_number}, "
            f"a={self.a:.5f}, b={self.b:.5f}, c={self.c:.5f}, "
            f"V={self.volume:.2f} ų, scale={self.scale:.4f})"
        )


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

@dataclass
class Instrument:
    """GSAS-II style TOF instrument description.

    Stores the key parameters from a GSAS-II .instprm file for a
    pulsed-neutron TOF instrument (Type PNT).  Parameters follow
    the GSAS-II naming convention.

    The design anticipates multiple banks — each bank would have its
    own Instrument instance with different difC, 2-theta, profile
    coefficients, etc.

    Attributes:
        inst_type: Instrument type string (e.g. "PNT").
        bank: Bank number from the file header.
        two_theta: Scattering angle 2θ (degrees).
        flt_path: Total flight path (metres).
        difA: Quadratic d-to-TOF coefficient (µs/ų).
        difB: Linear offset d-to-TOF coefficient.
        difC: Primary d-to-TOF coefficient (µs/Å)
            — TOF ≈ difC·d + difA·d² + Zero.
        zero: Zero-point offset (µs).
        alpha: Ikeda-Carpenter alpha parameter.
        beta_0: Profile beta-0 coefficient.
        beta_1: Profile beta-1 coefficient.
        beta_q: Profile beta-q coefficient.
        sig_0: Gaussian sigma-0 coefficient.
        sig_1: Gaussian sigma-1 coefficient.
        sig_2: Gaussian sigma-2 coefficient.
        sig_q: Gaussian sigma-q coefficient.
        X: Lorentzian X profile parameter.
        Y: Lorentzian Y profile parameter.
        Z: Lorentzian Z profile parameter.
        azimuth: Detector azimuthal angle (degrees).
        source_file: Path to the .instprm file (if loaded from file).
        raw_params: Full dict of all key-value pairs from the file.
    """

    inst_type: str = "PNT"
    bank: int = 1
    two_theta: float = 0.0
    flt_path: float = 0.0
    difA: float = 0.0
    difB: float = 0.0
    difC: float = 0.0
    zero: float = 0.0
    alpha: float = 0.0
    beta_0: float = 0.0
    beta_1: float = 0.0
    beta_q: float = 0.0
    sig_0: float = 0.0
    sig_1: float = 0.0
    sig_2: float = 0.0
    sig_q: float = 0.0
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    azimuth: float = 0.0
    source_file: str = ""
    raw_params: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, float]:
        """Return the numeric instrument parameters as a flat dict.

        Useful for iteration and display (matches the project.md
        example: ``for param in result.instrument.params``).
        """
        return {
            "difA": self.difA,
            "difB": self.difB,
            "difC": self.difC,
            "Zero": self.zero,
            "2-theta": self.two_theta,
            "fltPath": self.flt_path,
            "alpha": self.alpha,
            "beta-0": self.beta_0,
            "beta-1": self.beta_1,
            "beta-q": self.beta_q,
            "sig-0": self.sig_0,
            "sig-1": self.sig_1,
            "sig-2": self.sig_2,
            "sig-q": self.sig_q,
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "Azimuth": self.azimuth,
        }

    def copy(self) -> Instrument:
        """Return a deep copy of this instrument."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"Instrument(type={self.inst_type!r}, bank={self.bank}, "
            f"2θ={self.two_theta:.2f}°, difC={self.difC:.2f}, "
            f"fltPath={self.flt_path:.4f})"
        )


# ---------------------------------------------------------------------------
# Inspection result
# ---------------------------------------------------------------------------

@dataclass
class InspectionResult:
    """Container for the output of inspect().

    Holds deep copies of the optimised crystal phases and instrument,
    plus any processed spectra.  The originals passed to inspect() are
    left untouched.

    Attributes:
        crystal_phases: Optimised CrystalPhase objects (copies).
        instrument: Optimised Instrument (copy), or None if instrument
            optimisation was not requested.
        processed_spectra: Processed DiffractionDataset (e.g. after
            background subtraction), or None.
        chi_squared: Goodness-of-fit metric, if available.
        metadata: Additional diagnostic information from the optimiser.
    """

    crystal_phases: list[CrystalPhase] = field(default_factory=list)
    instrument: Instrument | None = None
    processed_spectra: DiffractionDataset | None = None
    chi_squared: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        phase_names = [p.name for p in self.crystal_phases]
        return (
            f"InspectionResult(phases={phase_names}, "
            f"has_instrument={self.instrument is not None}, "
            f"chi²={self.chi_squared})"
        )
