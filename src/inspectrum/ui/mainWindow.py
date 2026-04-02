"""Main window — InspectrumWindow dialog.

Assembles the data panel, phase panel, and results panel into a
single QDialog with a splitter layout, toolbar, progress bar, and
status bar.  Wires signals/slots for the complete interactive workflow.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QThread  # type: ignore
from qtpy.QtWidgets import (  # type: ignore
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inspectrum.models import InspectionResult
from inspectrum.ui.dataPanel import DataPanel
from inspectrum.ui.model import InspectrumModel
from inspectrum.ui.phasePanel import PhasePanel
from inspectrum.ui.resultsPanel import ResultsPanel
from inspectrum.ui.worker import InspectionWorker


class InspectrumWindow(QDialog):
    """Top-level dialog for interactive spectrum inspection.

    Layout::

        ┌──────────────┬───────────────────────┐
        │  Phase panel  │                       │
        │               │   Results panel       │
        ├──────────────┤   (plot + table)       │
        │  Data panel   │                       │
        └──────────────┴───────────────────────┘
        [ Run ] [ Clear ]           [progress] status

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inspectrum — Diffraction Inspector")
        self.resize(1200, 720)

        self._model = InspectrumModel()
        self._thread: QThread | None = None
        self._worker: InspectionWorker | None = None

        self._build_ui()
        self._connect_signals()
        self._setRunning(False)

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Main splitter: left panels | right result
        splitter = QSplitter(Qt.Horizontal)

        # -- Left column: phase + data stacked vertically --
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._phasePanel = PhasePanel(self._model)
        left_layout.addWidget(self._phasePanel, stretch=3)

        self._dataPanel = DataPanel()
        left_layout.addWidget(self._dataPanel, stretch=2)

        splitter.addWidget(left)

        # -- Right: results --
        self._resultsPanel = ResultsPanel()
        splitter.addWidget(self._resultsPanel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter, stretch=1)

        # -- Bottom bar: buttons + progress + status --
        bottom = QHBoxLayout()

        self._runBtn = QPushButton("Run Inspection")
        self._runBtn.setMinimumWidth(120)
        bottom.addWidget(self._runBtn)

        self._clearBtn = QPushButton("Clear")
        bottom.addWidget(self._clearBtn)

        bottom.addStretch()

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedWidth(160)
        self._progress.hide()
        bottom.addWidget(self._progress)

        self._status = QLabel("Ready")
        bottom.addWidget(self._status)

        root.addLayout(bottom)

    # ── Signal wiring ─────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._runBtn.clicked.connect(self._onRun)
        self._clearBtn.clicked.connect(self._onClear)

    # ── Slots ─────────────────────────────────────────────────────

    def _onRun(self) -> None:
        """Load data into the model and launch the pipeline worker."""
        try:
            self._loadDataIntoModel()
        except Exception as exc:
            QMessageBox.warning(self, "Data Error", str(exc))
            return

        if not self._model.phases:
            QMessageBox.information(
                self,
                "No Phases",
                "Add at least one CIF phase before running.",
            )
            return

        self._setRunning(True)
        self._status.setText("Running pipeline…")

        self._thread = QThread()
        self._worker = InspectionWorker(self._model)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._onResult)
        self._worker.error.connect(self._onError)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _onResult(self, result: InspectionResult) -> None:
        """Handle successful pipeline completion."""
        self._model.result = result
        self._resultsPanel.show_result(result)
        self._setRunning(False)
        n_ref = len(result.refinements) if result.refinements else 0
        p_str = (
            f"  P ≈ {result.sweep_pressure_gpa:.1f} GPa"
            if result.sweep_pressure_gpa is not None
            else ""
        )
        self._status.setText(f"Done — {n_ref} phase(s) refined.{p_str}")

    def _onError(self, message: str) -> None:
        """Handle pipeline failure."""
        self._setRunning(False)
        self._status.setText("Error")
        QMessageBox.critical(self, "Pipeline Error", message)

    def _onClear(self) -> None:
        """Clear results and reset status."""
        self._model.result = None
        self._resultsPanel.clear()
        self._status.setText("Ready")

    # ── Helpers ───────────────────────────────────────────────────

    def _loadDataIntoModel(self) -> None:
        """Transfer data-panel inputs into the model.

        Raises:
            ValueError: If required fields are missing.
        """
        dp = self._dataPanel

        if dp.source_mode == "file":
            if not dp.gsa_path:
                raise ValueError("No spectrum file selected.")
            if not dp.instprm_path:
                raise ValueError("No instrument parameter file selected.")
            self._model.load_from_files(
                dp.gsa_path,
                dp.instprm_path,
                bank=dp.bank,
            )
        else:
            if not dp.workspace_name:
                raise ValueError("No workspace name specified.")
            self._model.load_from_workspace(
                dp.workspace_name,
                bank=dp.bank,
            )
            if dp.instprm_path:
                from inspectrum.loaders import load_instprm

                self._model.instrument = load_instprm(dp.instprm_path)
            else:
                self._model.load_instrument_from_workspace(dp.workspace_name)

        self._model.P_min = dp.P_min
        self._model.P_max = dp.P_max
        self._model.temperature = dp.temperature

    def _setRunning(self, running: bool) -> None:
        """Toggle UI elements for running/idle state."""
        self._runBtn.setEnabled(not running)
        self._progress.setVisible(running)

    def closeEvent(self, event) -> None:
        """Ensure background thread is stopped before closing."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
