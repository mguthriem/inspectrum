"""
Tests for inspectrum data loaders.

Uses real test fixtures in tests/test_data/ to verify loading
of GSA, Mantid CSV, instprm, and CIF files.
"""

from pathlib import Path

import numpy as np
import pytest

from inspectrum.loaders import (
    load_cif,
    load_gsa,
    load_instprm,
    load_mantid_csv,
)

# Path to test data directory
TEST_DATA = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# GSA loader
# ---------------------------------------------------------------------------

class TestLoadGsa:
    """Tests for the GSAS FXYE (.gsa) loader."""

    def test_load_single_bank_gsa(self):
        """Test loading a single-bank GSA file."""
        spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")

        assert len(spectra) == 1
        spec = spectra[0]
        assert spec.x_unit == "TOF"
        assert spec.bank == 0
        assert spec.n_points == 863
        assert "SNAP059056" in spec.label

    def test_gsa_tof_range(self):
        """Test that TOF values are in a reasonable range for SNAP."""
        spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
        spec = spectra[0]

        # BANK line says: SLOG 4123 13024
        assert spec.x[0] == pytest.approx(4125.524, rel=1e-3)
        assert spec.x[-1] == pytest.approx(13024.07, rel=1e-2)

    def test_gsa_intensities_positive(self):
        """Test that intensities and uncertainties are positive."""
        spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
        spec = spectra[0]

        assert np.all(spec.y > 0)
        assert np.all(spec.e > 0)

    def test_gsa_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gsa(TEST_DATA / "nonexistent.gsa")

    def test_all_gsa_files_loadable(self):
        """Test that all GSA files in the test directory load."""
        gsa_files = sorted(TEST_DATA.glob("*.gsa"))
        assert len(gsa_files) >= 6

        for gsa_file in gsa_files:
            spectra = load_gsa(gsa_file)
            assert len(spectra) >= 1
            assert spectra[0].n_points > 0


# ---------------------------------------------------------------------------
# Mantid CSV loader
# ---------------------------------------------------------------------------

class TestLoadMantidCsv:
    """Tests for the Mantid CSV loader."""

    def test_load_single_spectrum_csv(self):
        """Test loading a single-spectrum Mantid CSV."""
        spectra = load_mantid_csv(TEST_DATA / "SNAP059056_all_test-0.csv")

        assert len(spectra) == 1
        spec = spectra[0]
        assert spec.x_unit == "d-Spacing"
        assert spec.n_points == 863
        assert spec.bank == 0

    def test_csv_d_spacing_range(self):
        """Test that d-spacing values match expected range."""
        spectra = load_mantid_csv(TEST_DATA / "SNAP059056_all_test-0.csv")
        spec = spectra[0]

        # d-range roughly 0.79 to 2.50 Å
        assert spec.d_min == pytest.approx(0.79, abs=0.01)
        assert spec.d_max == pytest.approx(2.50, abs=0.01)

    def test_csv_intensities_positive(self):
        """Test that intensities are reasonable."""
        spectra = load_mantid_csv(TEST_DATA / "SNAP059056_all_test-0.csv")
        spec = spectra[0]

        # Counts should be positive
        assert np.all(spec.y > 0)
        assert np.all(spec.e > 0)

    def test_csv_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_mantid_csv(TEST_DATA / "nonexistent.csv")

    def test_gsa_and_csv_same_n_points(self):
        """Test that GSA and CSV for the same run have equal point count."""
        gsa_spectra = load_gsa(TEST_DATA / "SNAP059056_all.gsa")
        csv_spectra = load_mantid_csv(TEST_DATA / "SNAP059056_all_test-0.csv")

        assert gsa_spectra[0].n_points == csv_spectra[0].n_points


# ---------------------------------------------------------------------------
# Instrument parameter loader
# ---------------------------------------------------------------------------

class TestLoadInstprm:
    """Tests for the GSAS-II .instprm loader."""

    def test_load_snap_instprm(self):
        """Test loading SNAP instrument parameters."""
        inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")

        assert inst.inst_type == "PNT"
        assert inst.bank == 1
        assert inst.difC == pytest.approx(5218.45)
        assert inst.two_theta == pytest.approx(85.3035)
        assert inst.flt_path == pytest.approx(15.5806)

    def test_profile_parameters_loaded(self):
        """Test that peak profile parameters are loaded."""
        inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")

        assert inst.alpha == pytest.approx(0.986512012223)
        assert inst.beta_0 == pytest.approx(0.0235)
        assert inst.beta_1 == pytest.approx(0.03)
        assert inst.sig_1 == pytest.approx(66.0)

    def test_params_dict_matches_fields(self):
        """Test that the .params dict matches individual fields."""
        inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")
        params = inst.params

        assert params["difC"] == inst.difC
        assert params["2-theta"] == inst.two_theta
        assert params["sig-1"] == inst.sig_1

    def test_raw_params_includes_pdabc(self):
        """Test that the raw pdabc absorption table is captured."""
        inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")

        assert "pdabc" in inst.raw_params

    def test_source_file_recorded(self):
        """Test that source_file records the path."""
        inst = load_instprm(TEST_DATA / "SNAP059056_all.instprm")
        assert "SNAP059056_all.instprm" in inst.source_file

    def test_instprm_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_instprm(TEST_DATA / "nonexistent.instprm")

    def test_all_instprm_files_consistent(self):
        """Test that all instprm files have the same difC (same bank)."""
        instprm_files = sorted(TEST_DATA.glob("*.instprm"))
        difc_values = []
        for f in instprm_files:
            inst = load_instprm(f)
            difc_values.append(inst.difC)

        # All should be the same (5218.45)
        assert all(
            v == pytest.approx(difc_values[0]) for v in difc_values
        )


# ---------------------------------------------------------------------------
# CIF loader
# ---------------------------------------------------------------------------

class TestLoadCif:
    """Tests for the CIF loader."""

    def test_load_tungsten_cif(self):
        """Test loading tungsten CIF — BCC, Im-3m."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode43421_tungsten.cif")

        assert phase.a == pytest.approx(3.16475)
        assert phase.b == pytest.approx(3.16475)
        assert phase.c == pytest.approx(3.16475)
        assert phase.alpha == pytest.approx(90.0)
        assert phase.space_group_number == 229
        assert "I m -3 m" in phase.space_group

    def test_load_ice_vii_cif(self):
        """Test loading ice VII CIF — Pn-3m."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode211586_iceVII.cif")

        assert phase.a == pytest.approx(3.3891, rel=1e-3)
        assert phase.space_group_number == 224
        assert "P n -3 m" in phase.space_group

    def test_tungsten_atom_sites(self):
        """Test that tungsten has one atom site (W at origin)."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode43421_tungsten.cif")

        assert len(phase.atom_sites) == 1
        site = phase.atom_sites[0]
        assert site["label"] == "W1"
        assert site["fract_x"] == "0"

    def test_ice_vii_atom_sites(self):
        """Test that ice VII has O and H sites."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode211586_iceVII.cif")

        assert len(phase.atom_sites) == 2
        labels = {s["label"] for s in phase.atom_sites}
        assert "O1" in labels
        assert "H1" in labels

    def test_tungsten_name_extracted(self):
        """Test that the chemical name is used as the phase label."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode43421_tungsten.cif")
        assert "Tungsten" in phase.name or "tungsten" in phase.name.lower()

    def test_tungsten_volume_matches_cif(self):
        """Test calculated volume matches CIF-reported volume."""
        phase = load_cif(TEST_DATA / "EntryWithCollCode43421_tungsten.cif")

        # CIF says 31.7 ų; our calculation from a³
        assert phase.volume == pytest.approx(31.7, abs=0.1)

    def test_cif_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_cif(TEST_DATA / "nonexistent.cif")
