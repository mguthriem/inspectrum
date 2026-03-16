"""
Tests for inspectrum data models.

Covers DiffractionSpectrum, DiffractionDataset, CrystalPhase,
Instrument, and InspectionResult.
"""

import numpy as np
import pytest

from inspectrum.models import (
    CrystalPhase,
    DiffractionDataset,
    DiffractionSpectrum,
    InspectionResult,
    Instrument,
)


# ---------------------------------------------------------------------------
# DiffractionSpectrum
# ---------------------------------------------------------------------------

class TestDiffractionSpectrum:
    """Tests for DiffractionSpectrum."""

    def test_create_spectrum_with_valid_data(self):
        """Test creating a spectrum with consistent x, y, e arrays."""
        x = np.array([0.79, 0.80, 0.81])
        y = np.array([85.0, 88.0, 91.0])
        e = np.array([2.1, 1.9, 1.8])
        spec = DiffractionSpectrum(x=x, y=y, e=e, label="test")

        assert spec.n_points == 3
        assert spec.label == "test"
        assert spec.bank == 0

    def test_d_min_and_d_max(self):
        """Test d_min and d_max properties."""
        x = np.array([0.79, 1.50, 2.50])
        y = np.zeros(3)
        e = np.zeros(3)
        spec = DiffractionSpectrum(x=x, y=y, e=e)

        assert spec.d_min == pytest.approx(0.79)
        assert spec.d_max == pytest.approx(2.50)

    def test_mismatched_array_lengths_raises_error(self):
        """Test that mismatched x, y, e lengths raise ValueError."""
        x = np.array([0.79, 0.80])
        y = np.array([85.0, 88.0, 91.0])
        e = np.array([2.1, 1.9])

        with pytest.raises(ValueError, match="equal length"):
            DiffractionSpectrum(x=x, y=y, e=e)

    def test_repr_contains_key_info(self):
        """Test that repr includes label, bank, n_points, d_range."""
        spec = DiffractionSpectrum(
            x=np.array([1.0, 2.0]),
            y=np.zeros(2),
            e=np.zeros(2),
            label="SNAP059056",
            bank=1,
        )
        r = repr(spec)
        assert "SNAP059056" in r
        assert "bank=1" in r
        assert "n_points=2" in r

    def test_default_units(self):
        """Test default x and y unit labels."""
        spec = DiffractionSpectrum(
            x=np.zeros(1), y=np.zeros(1), e=np.zeros(1)
        )
        assert spec.x_unit == "d-Spacing"
        assert "Counts" in spec.y_unit


# ---------------------------------------------------------------------------
# DiffractionDataset
# ---------------------------------------------------------------------------

class TestDiffractionDataset:
    """Tests for DiffractionDataset."""

    def test_empty_dataset(self):
        """Test creating an empty dataset."""
        ds = DiffractionDataset(label="empty")
        assert ds.n_spectra == 0
        assert len(ds) == 0

    def test_dataset_with_spectra(self):
        """Test dataset indexing and length."""
        specs = [
            DiffractionSpectrum(
                x=np.zeros(5), y=np.zeros(5), e=np.zeros(5),
                label=f"run{i}"
            )
            for i in range(3)
        ]
        ds = DiffractionDataset(spectra=specs, label="pressure_series")

        assert ds.n_spectra == 3
        assert len(ds) == 3
        assert ds[0].label == "run0"
        assert ds[2].label == "run2"

    def test_repr_shows_count(self):
        """Test repr includes spectrum count."""
        ds = DiffractionDataset(label="test")
        assert "n_spectra=0" in repr(ds)


# ---------------------------------------------------------------------------
# CrystalPhase
# ---------------------------------------------------------------------------

class TestCrystalPhase:
    """Tests for CrystalPhase."""

    def test_tungsten_cubic_volume(self):
        """Test volume calculation for BCC tungsten (cubic)."""
        # Tungsten: a = 3.16475 Å, cubic
        w = CrystalPhase(
            name="tungsten",
            a=3.16475, b=3.16475, c=3.16475,
            alpha=90.0, beta=90.0, gamma=90.0,
            space_group="I m -3 m",
            space_group_number=229,
        )
        # V = a³ for cubic
        expected_volume = 3.16475**3
        assert w.volume == pytest.approx(expected_volume, rel=1e-6)

    def test_ice_vii_volume(self):
        """Test volume calculation for ice VII (cubic)."""
        ice = CrystalPhase(
            name="ice-VII",
            a=3.3891, b=3.3891, c=3.3891,
            alpha=90.0, beta=90.0, gamma=90.0,
            space_group="P n -3 m Z",
            space_group_number=224,
        )
        expected_volume = 3.3891**3
        assert ice.volume == pytest.approx(expected_volume, rel=1e-6)

    def test_copy_is_independent(self):
        """Test that copy() returns a deep copy."""
        original = CrystalPhase(
            name="tungsten",
            a=3.16475, b=3.16475, c=3.16475,
            atom_sites=[{"label": "W1", "fract_x": 0.0}],
        )
        copied = original.copy()

        # Modify the copy
        copied.a = 3.20
        copied.atom_sites[0]["fract_x"] = 0.5

        # Original must be untouched
        assert original.a == pytest.approx(3.16475)
        assert original.atom_sites[0]["fract_x"] == pytest.approx(0.0)

    def test_default_scale_is_one(self):
        """Test that default scale factor is 1.0."""
        phase = CrystalPhase(name="test")
        assert phase.scale == pytest.approx(1.0)

    def test_repr_contains_name_and_space_group(self):
        """Test repr shows name and space group."""
        phase = CrystalPhase(
            name="tungsten",
            a=3.16, b=3.16, c=3.16,
            space_group="I m -3 m",
            space_group_number=229,
        )
        r = repr(phase)
        assert "tungsten" in r
        assert "I m -3 m" in r
        assert "#229" in r


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

class TestInstrument:
    """Tests for Instrument."""

    def test_create_snap_instrument(self):
        """Test creating an instrument with SNAP-like parameters."""
        inst = Instrument(
            inst_type="PNT",
            bank=1,
            two_theta=85.3035,
            flt_path=15.5806,
            difC=5218.45,
            alpha=0.986512012223,
            beta_0=0.0235,
            beta_1=0.03,
            sig_1=66.0,
        )
        assert inst.difC == pytest.approx(5218.45)
        assert inst.two_theta == pytest.approx(85.3035)

    def test_params_dict_contains_all_keys(self):
        """Test that params property returns all expected keys."""
        inst = Instrument(difC=5218.45, sig_1=66.0)
        p = inst.params

        expected_keys = {
            "difA", "difB", "difC", "Zero", "2-theta", "fltPath",
            "alpha", "beta-0", "beta-1", "beta-q",
            "sig-0", "sig-1", "sig-2", "sig-q",
            "X", "Y", "Z", "Azimuth",
        }
        assert set(p.keys()) == expected_keys
        assert p["difC"] == pytest.approx(5218.45)
        assert p["sig-1"] == pytest.approx(66.0)

    def test_copy_is_independent(self):
        """Test that copy() returns a deep copy."""
        original = Instrument(difC=5218.45)
        copied = original.copy()
        copied.difC = 9999.0

        assert original.difC == pytest.approx(5218.45)

    def test_repr_contains_key_values(self):
        """Test repr shows type, bank, 2θ, difC."""
        inst = Instrument(
            inst_type="PNT", bank=1,
            two_theta=85.3, difC=5218.45, flt_path=15.58,
        )
        r = repr(inst)
        assert "PNT" in r
        assert "bank=1" in r
        assert "5218.45" in r


# ---------------------------------------------------------------------------
# InspectionResult
# ---------------------------------------------------------------------------

class TestInspectionResult:
    """Tests for InspectionResult."""

    def test_create_empty_result(self):
        """Test creating a result with defaults."""
        result = InspectionResult()
        assert result.crystal_phases == []
        assert result.instrument is None
        assert result.processed_spectra is None
        assert result.chi_squared is None

    def test_result_with_phases_and_instrument(self):
        """Test creating a result with phases and instrument."""
        phases = [
            CrystalPhase(name="tungsten", a=3.16),
            CrystalPhase(name="ice-VII", a=3.39),
        ]
        inst = Instrument(difC=5218.45)
        result = InspectionResult(
            crystal_phases=phases,
            instrument=inst,
            chi_squared=1.23,
        )
        assert len(result.crystal_phases) == 2
        assert result.instrument is not None
        assert result.chi_squared == pytest.approx(1.23)

    def test_repr_shows_phase_names(self):
        """Test repr shows phase names."""
        result = InspectionResult(
            crystal_phases=[CrystalPhase(name="tungsten")],
        )
        r = repr(result)
        assert "tungsten" in r
