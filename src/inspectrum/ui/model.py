"""Pure-Python data model for the Inspectrum UI.

Qt-agnostic — can be tested with pytest without any Qt dependency.
Wraps the inspectrum analysis pipeline and manages the list of phases
and experiment parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspectrum.engine import inspect, tof_to_d
from inspectrum.loaders import load_cif, load_gsa, load_instprm
from inspectrum.models import (
    CrystalPhase,
    DiffractionSpectrum,
    EquationOfState,
    ExperimentDescription,
    InspectionResult,
    Instrument,
    PhaseDescription,
    SampleConditions,
)


class InspectrumModel:
    """Manages experiment state and delegates to the inspectrum pipeline.

    Attributes:
        phases: List of phase descriptions (with optional EOS).
        spectrum: Currently loaded spectrum, or None.
        instrument: Currently loaded instrument, or None.
        result: Most recent inspection result, or None.
    """

    def __init__(self) -> None:
        self.phases: list[PhaseDescription] = []
        self.spectrum: DiffractionSpectrum | None = None
        self.instrument: Instrument | None = None
        self.result: InspectionResult | None = None

        # Pipeline parameters
        self.P_min: float = 0.0
        self.P_max: float = 100.0
        self.temperature: float = 295.0

    # ── Phase management ──────────────────────────────────────────

    def add_phase_from_cif(
        self,
        cif_path: str | Path,
        role: str = "sample",
        name: str | None = None,
    ) -> PhaseDescription:
        """Load a CIF file and add it as a phase.

        Args:
            cif_path: Path to the .cif file.
            role: ``"calibrant"`` or ``"sample"``.
            name: Override phase name (uses CIF name if None).

        Returns:
            The created PhaseDescription.

        Raises:
            FileNotFoundError: If CIF file does not exist.
            ValueError: If CIF is missing required fields.
        """
        phase = load_cif(cif_path)
        if name:
            phase.name = name

        desc = PhaseDescription(
            name=phase.name or Path(cif_path).stem,
            cif_path=str(cif_path),
            role=role,
            phase=phase,
        )
        self.phases.append(desc)
        return desc

    def remove_phase(self, index: int) -> None:
        """Remove a phase by index."""
        if 0 <= index < len(self.phases):
            self.phases.pop(index)

    def set_eos(
        self,
        phase_index: int,
        eos_type: str,
        V_0: float,
        K_0: float,
        K_prime: float,
        source: str = "",
    ) -> None:
        """Attach an EOS to a phase.

        Args:
            phase_index: Index into self.phases.
            eos_type: ``"birch-murnaghan"``, ``"vinet"``, or
                ``"murnaghan"``.
            V_0: Reference volume (ų per unit cell).
            K_0: Bulk modulus (GPa).
            K_prime: Pressure derivative of bulk modulus.
            source: Literature citation.
        """
        desc = self.phases[phase_index]
        desc.eos = EquationOfState(
            eos_type=eos_type,
            V_0=V_0,
            K_0=K_0,
            K_prime=K_prime,
            source=source,
        )

    def set_stability_range(
        self,
        phase_index: int,
        P_min: float | None = None,
        P_max: float | None = None,
    ) -> None:
        """Set the pressure stability range for a phase."""
        self.phases[phase_index].stability_pressure = (P_min, P_max)

    # ── Data loading ──────────────────────────────────────────────

    def load_from_files(
        self,
        gsa_path: str | Path,
        instprm_path: str | Path,
        bank: int = 0,
    ) -> None:
        """Load spectrum and instrument from files.

        Args:
            gsa_path: Path to .gsa or .csv file.
            instprm_path: Path to .instprm file.
            bank: Detector bank index (0-based).
        """
        gsa_path = Path(gsa_path)
        if gsa_path.suffix.lower() == ".csv":
            from inspectrum.loaders import load_mantid_csv

            spectra = load_mantid_csv(gsa_path)
        else:
            spectra = load_gsa(gsa_path)

        if bank < len(spectra):
            self.spectrum = spectra[bank]
        else:
            self.spectrum = spectra[0]

        self.instrument = load_instprm(instprm_path)

    def load_from_workspace(self, workspace_name: str, bank: int = 0) -> None:
        """Load spectrum from a Mantid workspace.

        Args:
            workspace_name: Name of the workspace in the Mantid ADS.
            bank: Spectrum index (bank) to extract.

        Raises:
            ImportError: If Mantid is not available.
            KeyError: If workspace is not found in the ADS.
            IndexError: If bank index is out of range.
        """
        import mantid.simpleapi  # noqa: F401
        from mantid.api import AnalysisDataService as ADS

        ws = ADS.retrieve(workspace_name)
        n_spectra = ws.getNumberHistograms()
        if bank >= n_spectra:
            raise IndexError(
                f"Bank {bank} out of range (workspace has {n_spectra} spectra)"
            )

        x = ws.readX(bank).copy()
        y = ws.readY(bank).copy()
        e = ws.readE(bank).copy()

        # Handle histogram data (x has one more bin than y)
        if len(x) == len(y) + 1:
            x = 0.5 * (x[:-1] + x[1:])

        # Determine x-unit
        x_unit_label = ws.getAxis(0).getUnit().unitID()
        if x_unit_label == "dSpacing":
            x_unit = "d-Spacing"
        elif x_unit_label == "TOF":
            x_unit = "TOF"
        else:
            x_unit = x_unit_label

        self.spectrum = DiffractionSpectrum(
            x=x,
            y=y,
            e=e,
            x_unit=x_unit,
            label=workspace_name,
            bank=bank,
        )

    def load_instrument_from_workspace(self, workspace_name: str) -> None:
        """Extract instrument parameters from a Mantid workspace.

        This is a simplification — in practice, the full GSAS-II
        instrument parameters may still need to come from .instprm.
        This method extracts difC and 2-theta from the workspace
        instrument definition.

        Args:
            workspace_name: Name of the workspace in the Mantid ADS.
        """
        from mantid.api import AnalysisDataService as ADS

        ws = ADS.retrieve(workspace_name)
        # Extract basic instrument geometry
        # Full profile params still need .instprm
        det = ws.getDetector(0)
        two_theta = ws.detectorTwoTheta(det) * 180.0 / 3.14159265

        self.instrument = Instrument(two_theta=two_theta)

    # ── Phase-EOS serialization ───────────────────────────────────

    def save_phase(self, phase_index: int, path: str | Path) -> None:
        """Save a phase description (with EOS) to a JSON file.

        Args:
            phase_index: Index into self.phases.
            path: Output file path.
        """
        desc = self.phases[phase_index]
        data: dict[str, Any] = {
            "name": desc.name,
            "cif_path": desc.cif_path,
            "role": desc.role,
        }
        if desc.eos is not None:
            data["eos"] = {
                "type": desc.eos.eos_type,
                "order": desc.eos.order,
                "V_0": desc.eos.V_0,
                "K_0": desc.eos.K_0,
                "K_prime": desc.eos.K_prime,
                "source": desc.eos.source,
            }
        if desc.stability_pressure is not None:
            data["stability_pressure"] = list(desc.stability_pressure)
        if desc.reference_conditions.pressure is not None:
            data.setdefault("reference_conditions", {})[
                "pressure"
            ] = desc.reference_conditions.pressure
        if desc.reference_conditions.temperature is not None:
            data.setdefault("reference_conditions", {})[
                "temperature"
            ] = desc.reference_conditions.temperature

        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def load_phase(self, path: str | Path) -> PhaseDescription:
        """Load a phase description from a JSON file.

        The JSON file should contain a single phase object with
        optional EOS and stability range.  The CIF file referenced
        in ``cif_path`` is loaded automatically.

        Args:
            path: Path to the JSON file.

        Returns:
            The loaded PhaseDescription (also added to self.phases).
        """
        raw = json.loads(Path(path).read_text())

        # Load CIF if path is specified
        cif_path = raw.get("cif_path", "")
        phase: CrystalPhase | None = None
        if cif_path:
            resolved = Path(path).parent / cif_path
            if resolved.exists():
                phase = load_cif(resolved)
            elif Path(cif_path).exists():
                phase = load_cif(cif_path)

        eos: EquationOfState | None = None
        if "eos" in raw:
            e = raw["eos"]
            eos = EquationOfState(
                eos_type=e.get("type", "birch-murnaghan"),
                order=e.get("order", 3),
                V_0=e["V_0"],
                K_0=e["K_0"],
                K_prime=e.get("K_prime", 4.0),
                source=e.get("source", ""),
            )

        ref_cond = SampleConditions()
        if "reference_conditions" in raw:
            rc = raw["reference_conditions"]
            ref_cond = SampleConditions(
                pressure=rc.get("pressure"),
                temperature=rc.get("temperature"),
            )

        stability = None
        if "stability_pressure" in raw:
            sp = raw["stability_pressure"]
            stability = (sp[0], sp[1])

        desc = PhaseDescription(
            name=raw.get("name", ""),
            cif_path=cif_path,
            role=raw.get("role", "sample"),
            reference_conditions=ref_cond,
            eos=eos,
            stability_pressure=stability,
            phase=phase,
        )
        self.phases.append(desc)
        return desc

    # ── Pipeline execution ────────────────────────────────────────

    def run_inspection(self) -> InspectionResult:
        """Run the full inspection pipeline.

        Uses the currently loaded spectrum, instrument, and phases.

        Returns:
            InspectionResult with matched peaks, refined lattice
            parameters, and diagnostic metadata.

        Raises:
            ValueError: If spectrum, instrument, or phases are missing.
        """
        if self.spectrum is None:
            raise ValueError("No spectrum loaded")
        if self.instrument is None:
            raise ValueError("No instrument loaded")
        if not self.phases:
            raise ValueError("No phases defined")

        experiment = ExperimentDescription(
            phases=self.phases,
            global_temperature=self.temperature,
            global_max_pressure=self.P_max,
        )

        self.result = inspect(
            self.spectrum,
            self.instrument,
            experiment,
            P_min=self.P_min,
            P_max=self.P_max,
        )
        return self.result

    # ── Convenience ───────────────────────────────────────────────

    def d_axis(self) -> Any:
        """Get the d-spacing axis for the current spectrum.

        Returns:
            Array of d-spacing values, or None if no data loaded.
        """
        if self.spectrum is None or self.instrument is None:
            return None
        if self.spectrum.x_unit == "TOF":
            return tof_to_d(self.spectrum.x, self.instrument)
        return self.spectrum.x
