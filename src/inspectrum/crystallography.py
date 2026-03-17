"""
Crystallographic calculations for inspectrum.

Provides reflection generation, d-spacing calculation, multiplicity
determination, and neutron structure factor computation.  Uses cryspy
for low-level crystallographic math and neutron scattering lengths.

Typical usage::

    from inspectrum.crystallography import generate_reflections
    from inspectrum.loaders import load_cif

    phase = load_cif("tungsten.cif")
    reflections = generate_reflections(phase, d_min=0.5, d_max=3.0)
    for r in reflections:
        print(r["hkl"], r["d"], r["multiplicity"], r["F_sq"])
"""

from __future__ import annotations

from typing import Any

import cryspy
import numpy as np

from inspectrum.models import CrystalPhase


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_reflections(
    phase: CrystalPhase,
    d_min: float,
    d_max: float,
) -> list[dict[str, Any]]:
    """Generate allowed reflections for *phase* in the d-spacing range.

    For each unique reflection the returned dict contains:

    - ``hkl``: tuple (h, k, l)
    - ``d``: d-spacing in Angstroms
    - ``multiplicity``: number of symmetry-equivalent reflections
    - ``F_sq``: neutron structure factor |F(hkl)|²

    Reflections are sorted by decreasing d-spacing (lowest-angle first).

    Args:
        phase: Crystal phase with lattice parameters, space group, and
            atom sites populated (e.g. from :func:`load_cif`).
        d_min: Minimum d-spacing (Å).
        d_max: Maximum d-spacing (Å).

    Returns:
        List of reflection dicts, sorted by decreasing d.

    Raises:
        ValueError: If d_min >= d_max or lattice parameters are invalid.
    """
    if d_min >= d_max:
        raise ValueError(f"d_min ({d_min}) must be less than d_max ({d_max})")

    centering = _extract_centering(phase.space_group)

    # Upper bound on Miller index from the smallest d-spacing
    a_max = max(phase.a, phase.b, phase.c)
    h_max = int(np.ceil(a_max / d_min)) + 1

    # Convert angles to radians once
    al = np.radians(phase.alpha)
    be = np.radians(phase.beta)
    ga = np.radians(phase.gamma)

    reflections: list[dict[str, Any]] = []

    for h in range(0, h_max + 1):
        for k in range(0, h_max + 1):
            for l_idx in range(0, h_max + 1):
                if h == 0 and k == 0 and l_idx == 0:
                    continue

                # Lattice centering systematic absences
                if not _passes_centering(h, k, l_idx, centering):
                    continue

                inv_d = cryspy.calc_inverse_d_by_hkl_abc_angles(
                    h, k, l_idx,
                    phase.a, phase.b, phase.c,
                    al, be, ga,
                )
                if inv_d <= 0:
                    continue

                d = 1.0 / inv_d
                if d < d_min or d > d_max:
                    continue

                mult = _multiplicity(h, k, l_idx, phase.space_group_number)
                f_sq = _calc_structure_factor_sq(
                    h, k, l_idx, d, phase.atom_sites, centering
                )

                reflections.append({
                    "hkl": (h, k, l_idx),
                    "d": d,
                    "multiplicity": mult,
                    "F_sq": f_sq,
                })

    # Sort by decreasing d-spacing
    reflections.sort(key=lambda r: -r["d"])

    # Remove near-duplicate d-spacings (symmetry-equivalent reflections
    # that we enumerated separately because we loop over all positive hkl).
    reflections = _merge_equivalent_reflections(reflections)

    return reflections


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_centering(space_group: str) -> str:
    """Return the lattice centering letter from a Hermann-Mauguin symbol.

    Examples:
        "I m -3 m" → "I"
        "P n -3 m" → "P"
        "F m -3 m" → "F"
        "C 2/m"    → "C"
    """
    sg = space_group.strip()
    if sg:
        return sg[0].upper()
    return "P"


def _passes_centering(h: int, k: int, l_val: int, centering: str) -> bool:
    """Check if (h, k, l) passes the lattice centering condition.

    Centering conditions (integral reflection conditions):

    - P: no condition
    - I: h + k + l = 2n
    - F: h, k, l all even or all odd
    - A: k + l = 2n
    - B: h + l = 2n
    - C: h + k = 2n
    - R (hexagonal axes): -h + k + l = 3n
    """
    if centering == "P":
        return True
    elif centering == "I":
        return (h + k + l_val) % 2 == 0
    elif centering == "F":
        # All even or all odd
        parities = (h % 2, k % 2, l_val % 2)
        return parities[0] == parities[1] == parities[2]
    elif centering == "A":
        return (k + l_val) % 2 == 0
    elif centering == "B":
        return (h + l_val) % 2 == 0
    elif centering == "C":
        return (h + k) % 2 == 0
    elif centering == "R":
        return (-h + k + l_val) % 3 == 0
    else:
        # Unknown centering — allow everything
        return True


def _multiplicity(h: int, k: int, l_val: int, sg_number: int) -> int:
    """Estimate reflection multiplicity from the Laue class.

    Uses the International Tables space group number to determine the
    crystal system and point group, then counts the number of
    symmetry-equivalent (h, k, l) reflections.

    This is an approximation — it uses the *Laue class* multiplicity
    (which is correct for powder diffraction where Friedel's law
    applies).
    """
    crystal_system = cryspy.get_crystal_system_by_it_number(sg_number)
    return _laue_multiplicity(h, k, l_val, crystal_system, sg_number)


def _laue_multiplicity(
    h: int, k: int, l_val: int, crystal_system: str, sg_number: int
) -> int:
    """Count equivalent reflections for the Laue class.

    For powder diffraction the multiplicity is the number of
    reflections that overlap at the same d-spacing.
    """
    # Generate all permutations and sign changes, count unique
    equivalents: set[tuple[int, int, int]] = set()

    if crystal_system == "cubic":
        _cubic_equivalents(h, k, l_val, equivalents)
    elif crystal_system == "hexagonal" or crystal_system == "trigonal":
        _hexagonal_equivalents(h, k, l_val, equivalents)
    elif crystal_system == "tetragonal":
        _tetragonal_equivalents(h, k, l_val, equivalents)
    elif crystal_system == "orthorhombic":
        _orthorhombic_equivalents(h, k, l_val, equivalents)
    elif crystal_system == "monoclinic":
        _monoclinic_equivalents(h, k, l_val, equivalents)
    elif crystal_system == "triclinic":
        _triclinic_equivalents(h, k, l_val, equivalents)
    else:
        _triclinic_equivalents(h, k, l_val, equivalents)

    return len(equivalents)


def _add_sign_permutations(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Add all sign combinations of (±h, ±k, ±l)."""
    for sh in (1, -1):
        for sk in (1, -1):
            for sl in (1, -1):
                equivalents.add((sh * h, sk * k, sl * l_val))


def _cubic_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """All permutations of indices with all sign combinations (m-3m)."""
    from itertools import permutations

    for perm in set(permutations((h, k, l_val))):
        _add_sign_permutations(perm[0], perm[1], perm[2], equivalents)


def _hexagonal_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Equivalents for hexagonal Laue class 6/mmm."""
    i = -(h + k)
    # The 12 in-plane operations on (h, k, i) keeping l sign
    hex_perms = [
        (h, k), (k, i), (i, h),
        (-h, -k), (-k, -i), (-i, -h),
        (k, h), (h, i), (i, k),
        (-k, -h), (-h, -i), (-i, -k),
    ]
    for hh, kk in hex_perms:
        equivalents.add((hh, kk, l_val))
        equivalents.add((hh, kk, -l_val))


def _tetragonal_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Equivalents for tetragonal Laue class 4/mmm."""
    perms_hk = [(h, k), (-h, -k), (-k, h), (k, -h),
                (k, h), (-k, -h), (h, -k), (-h, k)]
    for hh, kk in perms_hk:
        equivalents.add((hh, kk, l_val))
        equivalents.add((hh, kk, -l_val))


def _orthorhombic_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Equivalents for orthorhombic Laue class mmm."""
    _add_sign_permutations(h, k, l_val, equivalents)


def _monoclinic_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Equivalents for monoclinic Laue class 2/m (unique axis b)."""
    equivalents.add((h, k, l_val))
    equivalents.add((-h, k, -l_val))
    equivalents.add((-h, -k, -l_val))
    equivalents.add((h, -k, l_val))


def _triclinic_equivalents(
    h: int, k: int, l_val: int, equivalents: set[tuple[int, int, int]]
) -> None:
    """Equivalents for triclinic Laue class -1."""
    equivalents.add((h, k, l_val))
    equivalents.add((-h, -k, -l_val))


def _calc_structure_factor_sq(
    h: int,
    k: int,
    l_val: int,
    d: float,
    atom_sites: list[dict[str, Any]],
    centering: str = "P",
) -> float:
    """Calculate |F(hkl)|² for neutron scattering.

    Uses the kinematic approximation:
        F(hkl) = Σ_j  b_j · occ_j · T_j · exp(2πi(h·x_j + k·y_j + l·z_j))

    where b_j is the bound coherent neutron scattering length,
    occ_j is the site occupancy, and T_j is a simple isotropic
    Debye-Waller factor (set to 1 if B_iso not provided).

    The atom sites from the CIF are expanded by the lattice centering
    translations (I, F, C, etc.) to get all centering-equivalent
    atoms in the unit cell.  Rotational symmetry operations beyond
    centering are not applied — this is an approximation that works
    well for simple structures and is adequate for pre-inspection.
    """
    if not atom_sites:
        return 1.0

    centering_vectors = _CENTERING_VECTORS.get(centering, [(0.0, 0.0, 0.0)])

    F = 0.0 + 0.0j
    sthovl = 0.5 / d  # sin(theta)/lambda = 1/(2d)

    for site in atom_sites:
        symbol = _extract_element(site.get("type_symbol", site.get("label", "")))
        try:
            b = cryspy.get_scat_length_neutron(symbol)
        except (KeyError, ValueError):
            b = 0.5 + 0j  # fallback

        occ = float(_parse_cif_number(str(site.get("occupancy", 1.0))))

        x = float(_parse_cif_number(str(site.get("fract_x", 0.0))))
        y = float(_parse_cif_number(str(site.get("fract_y", 0.0))))
        z = float(_parse_cif_number(str(site.get("fract_z", 0.0))))

        # Isotropic Debye-Waller factor
        # CIF may store B_iso or U_iso (B = 8π²U)
        b_iso_str = site.get("B_iso_or_equiv", site.get("b_iso_or_equiv", "."))
        u_iso_str = site.get("U_iso_or_equiv", site.get("u_iso_or_equiv", "."))
        b_iso = 0.0
        if b_iso_str not in (".", "", None):
            b_iso = float(_parse_cif_number(str(b_iso_str)))
        elif u_iso_str not in (".", "", None):
            u_iso = float(_parse_cif_number(str(u_iso_str)))
            b_iso = 8 * np.pi**2 * u_iso
        dw = np.exp(-b_iso * sthovl**2)

        # Sum over centering translations
        for tx, ty, tz in centering_vectors:
            xc = x + tx
            yc = y + ty
            zc = z + tz
            phase = 2 * np.pi * (h * xc + k * yc + l_val * zc)
            F += complex(b) * occ * dw * np.exp(1j * phase)

    return float(abs(F) ** 2)


def _extract_element(type_symbol: str) -> str:
    """Extract the element symbol from a CIF type_symbol.

    Handles cases like "W", "O2-", "H1+", "Fe3+", "W0+".
    """
    # Take the alphabetic prefix
    element = ""
    for char in type_symbol:
        if char.isalpha():
            element += char
        else:
            break
    return element if element else type_symbol


def _parse_cif_number(value_str: str) -> float:
    """Parse a CIF numeric value, stripping uncertainty in parentheses.

    Examples:
        "3.16475(20)" → 3.16475
        "0.202(3)"    → 0.202
        "90."         → 90.0
        "0"           → 0.0
    """
    s = str(value_str).strip()
    paren = s.find("(")
    if paren != -1:
        s = s[:paren]
    return float(s)


# Centering translation vectors for each Bravais lattice type
_CENTERING_VECTORS: dict[str, list[tuple[float, float, float]]] = {
    "P": [(0.0, 0.0, 0.0)],
    "I": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    "F": [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ],
    "A": [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],
    "B": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],
    "C": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],
    "R": [(0.0, 0.0, 0.0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3)],
}


def _merge_equivalent_reflections(
    reflections: list[dict[str, Any]],
    d_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Merge reflections with the same d-spacing (within tolerance).

    When we loop over all positive (h, k, l) we may generate
    symmetry-equivalent reflections separately (e.g. (2,1,0) and
    (2,0,1) for cubic).  This function merges them, keeping the
    first hkl tuple as representative and summing nothing (they
    have the same d, F_sq, and multiplicity by construction).
    """
    if not reflections:
        return reflections

    merged: list[dict[str, Any]] = [reflections[0]]
    for r in reflections[1:]:
        if abs(r["d"] - merged[-1]["d"]) < d_tolerance:
            # Same d-spacing — skip duplicate
            continue
        merged.append(r)
    return merged
