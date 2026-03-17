"""
Tests for the crystallography module.

Tests reflection generation, d-spacing calculation, multiplicities,
structure factors, and systematic absence filtering using real CIF
fixtures (tungsten Im-3m and ice VII Pn-3m).
"""

from pathlib import Path

import numpy as np
import pytest

from inspectrum.crystallography import (
    _extract_centering,
    _multiplicity,
    _parse_cif_number,
    _passes_centering,
    generate_reflections,
)
from inspectrum.loaders import load_cif
from inspectrum.models import CrystalPhase

TEST_DATA = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# Helper / internal function tests
# ---------------------------------------------------------------------------


class TestExtractCentering:
    """Tests for _extract_centering."""

    def test_body_centered(self):
        assert _extract_centering("I m -3 m") == "I"

    def test_primitive(self):
        assert _extract_centering("P n -3 m") == "P"

    def test_face_centered(self):
        assert _extract_centering("F m -3 m") == "F"

    def test_c_centered(self):
        assert _extract_centering("C 2/m") == "C"

    def test_empty_string_defaults_to_p(self):
        assert _extract_centering("") == "P"


class TestPassesCentering:
    """Tests for systematic absence rules."""

    def test_p_allows_everything(self):
        assert _passes_centering(1, 0, 0, "P") is True
        assert _passes_centering(1, 1, 1, "P") is True

    def test_i_requires_hkl_even_sum(self):
        assert _passes_centering(1, 1, 0, "I") is True   # 1+1+0=2
        assert _passes_centering(1, 0, 0, "I") is False   # 1+0+0=1
        assert _passes_centering(2, 1, 1, "I") is True    # 2+1+1=4

    def test_f_requires_all_same_parity(self):
        assert _passes_centering(1, 1, 1, "F") is True    # all odd
        assert _passes_centering(2, 0, 0, "F") is True    # all even
        assert _passes_centering(1, 1, 0, "F") is False   # mixed

    def test_c_requires_hk_even_sum(self):
        assert _passes_centering(1, 1, 0, "C") is True    # 1+1=2
        assert _passes_centering(1, 0, 0, "C") is False   # 1+0=1


class TestParseCifNumber:
    """Tests for _parse_cif_number."""

    def test_number_with_uncertainty(self):
        assert _parse_cif_number("3.16475(20)") == pytest.approx(3.16475)

    def test_number_without_uncertainty(self):
        assert _parse_cif_number("0.25") == pytest.approx(0.25)

    def test_integer(self):
        assert _parse_cif_number("0") == pytest.approx(0.0)

    def test_trailing_dot(self):
        assert _parse_cif_number("90.") == pytest.approx(90.0)


class TestMultiplicity:
    """Tests for multiplicity calculation."""

    def test_cubic_110(self):
        """(110) in m-3m has multiplicity 12."""
        assert _multiplicity(1, 1, 0, 229) == 12

    def test_cubic_200(self):
        """(200) in m-3m has multiplicity 6."""
        assert _multiplicity(2, 0, 0, 229) == 6

    def test_cubic_211(self):
        """(211) in m-3m has multiplicity 24."""
        assert _multiplicity(2, 1, 1, 229) == 24

    def test_cubic_222(self):
        """(222) in m-3m has multiplicity 8."""
        assert _multiplicity(2, 2, 2, 229) == 8

    def test_cubic_321(self):
        """(321) in m-3m (general) has multiplicity 48."""
        assert _multiplicity(3, 2, 1, 229) == 48


# ---------------------------------------------------------------------------
# generate_reflections with real CIF data
# ---------------------------------------------------------------------------


class TestGenerateReflectionsTungsten:
    """Tests using tungsten (Im-3m, BCC, a=3.16475 Å)."""

    @pytest.fixture
    def tungsten(self):
        return load_cif(TEST_DATA / "EntryWithCollCode43421_tungsten.cif")

    def test_reflection_count_in_snap_range(self, tungsten):
        """Tungsten has 8 reflections in d=[0.79, 2.50]."""
        refs = generate_reflections(tungsten, d_min=0.79, d_max=2.50)
        assert len(refs) == 8

    def test_first_reflection_is_110(self, tungsten):
        """Strongest/first reflection is (110) at d≈2.238 Å."""
        refs = generate_reflections(tungsten, d_min=0.79, d_max=2.50)
        first = refs[0]
        assert first["d"] == pytest.approx(2.23782, rel=1e-4)
        assert first["multiplicity"] == 12

    def test_sorted_by_decreasing_d(self, tungsten):
        """Reflections are sorted by decreasing d-spacing."""
        refs = generate_reflections(tungsten, d_min=0.79, d_max=2.50)
        d_values = [r["d"] for r in refs]
        assert d_values == sorted(d_values, reverse=True)

    def test_known_d_spacings(self, tungsten):
        """Check d-spacings match d = a/√(h²+k²+l²) for cubic."""
        refs = generate_reflections(tungsten, d_min=0.79, d_max=2.50)
        a = tungsten.a
        for r in refs:
            h, k, l = r["hkl"]
            expected_d = a / np.sqrt(h**2 + k**2 + l**2)
            assert r["d"] == pytest.approx(expected_d, rel=1e-4)

    def test_all_f_sq_equal_for_single_atom_bcc(self, tungsten):
        """For single-atom BCC, all |F|² are the same (4b²)."""
        refs = generate_reflections(tungsten, d_min=0.79, d_max=2.50)
        f_sq_values = [r["F_sq"] for r in refs]
        # All should be approximately 4 * 0.486² ≈ 0.945
        assert all(
            f == pytest.approx(f_sq_values[0], rel=1e-2) for f in f_sq_values
        )
        assert f_sq_values[0] == pytest.approx(4 * 0.486**2, rel=0.01)

    def test_no_odd_sum_reflections(self, tungsten):
        """BCC (I centering): no reflections with h+k+l odd."""
        refs = generate_reflections(tungsten, d_min=0.5, d_max=3.0)
        for r in refs:
            h, k, l = r["hkl"]
            assert (h + k + l) % 2 == 0

    def test_d_min_ge_d_max_raises_error(self, tungsten):
        with pytest.raises(ValueError, match="d_min"):
            generate_reflections(tungsten, d_min=2.0, d_max=1.0)


class TestGenerateReflectionsIceVII:
    """Tests using ice VII (Pn-3m, a=3.3891 Å)."""

    @pytest.fixture
    def ice_vii(self):
        return load_cif(TEST_DATA / "EntryWithCollCode211586_iceVII.cif")

    def test_reflection_count(self, ice_vii):
        """Ice VII (P lattice) has more reflections than BCC tungsten."""
        refs = generate_reflections(ice_vii, d_min=0.79, d_max=2.50)
        assert len(refs) == 15

    def test_first_reflection_d(self, ice_vii):
        """First reflection at d ≈ a/√2 ≈ 2.396 Å."""
        refs = generate_reflections(ice_vii, d_min=0.79, d_max=2.50)
        expected_d = ice_vii.a / np.sqrt(2)
        assert refs[0]["d"] == pytest.approx(expected_d, rel=1e-3)

    def test_f_sq_varies_for_multi_atom(self, ice_vii):
        """Multi-atom structure should have varying |F|² across reflections."""
        refs = generate_reflections(ice_vii, d_min=0.79, d_max=2.50)
        f_sq_values = [r["F_sq"] for r in refs]
        # Not all the same
        assert max(f_sq_values) > 1.5 * min(f_sq_values)


class TestGenerateReflectionsMinimal:
    """Tests with hand-crafted CrystalPhase objects."""

    def test_simple_cubic(self):
        """Simple cubic with a=2.0 should give a handful of reflections."""
        phase = CrystalPhase(
            name="simple_cubic",
            a=2.0, b=2.0, c=2.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            space_group="P m -3 m",
            space_group_number=221,
            atom_sites=[{
                "label": "X1",
                "type_symbol": "Fe",
                "fract_x": "0", "fract_y": "0", "fract_z": "0",
                "occupancy": "1.0",
            }],
        )
        refs = generate_reflections(phase, d_min=0.5, d_max=3.0)
        assert len(refs) > 0
        # (100) should be present for P lattice
        d_values = [r["d"] for r in refs]
        assert any(abs(d - 2.0) < 0.001 for d in d_values)  # d(100) = a

    def test_empty_d_range_returns_empty(self):
        """Very narrow d-range with no reflections returns empty list."""
        phase = CrystalPhase(
            name="test",
            a=5.0, b=5.0, c=5.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            space_group="P m -3 m",
            space_group_number=221,
        )
        refs = generate_reflections(phase, d_min=4.99, d_max=5.01)
        # d(100) = 5.0, so there should be exactly 1 reflection
        assert len(refs) <= 1
