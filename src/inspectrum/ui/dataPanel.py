"""Data input panel — specifies the spectrum to inspect.

Provides controls for:
- Loading data from file (GSA/CSV) or Mantid workspace
- Selecting instrument parameter file
- Setting pressure and temperature ranges
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Signal  # type: ignore
from qtpy.QtWidgets import (  # type: ignore
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class DataPanel(QWidget):
    """Panel for specifying input data and pipeline parameters.

    Signals:
        dataChanged: Emitted when the data source or parameters change.
    """

    dataChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Data source ───────────────────────────────────────────
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)

        # Source selector: file or workspace
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self._sourceCombo = QComboBox()
        self._sourceCombo.addItems(["File", "Workspace"])
        self._sourceCombo.currentIndexChanged.connect(self._onSourceChanged)
        source_row.addWidget(self._sourceCombo)
        source_layout.addLayout(source_row)

        # Stacked widget for file vs workspace inputs
        self._sourceStack = QStackedWidget()

        # -- File page --
        file_page = QWidget()
        file_form = QFormLayout(file_page)
        file_form.setContentsMargins(0, 0, 0, 0)

        self._gsaPathEdit = QLineEdit()
        self._gsaPathEdit.setPlaceholderText("Path to .gsa or .csv file")
        self._gsaPathEdit.setReadOnly(True)
        gsa_row = QHBoxLayout()
        gsa_row.addWidget(self._gsaPathEdit)
        self._gsaBrowseBtn = QPushButton("Browse…")
        self._gsaBrowseBtn.clicked.connect(self._browseGsa)
        gsa_row.addWidget(self._gsaBrowseBtn)
        file_form.addRow("Spectrum:", gsa_row)

        self._instprmPathEdit = QLineEdit()
        self._instprmPathEdit.setPlaceholderText("Path to .instprm file")
        self._instprmPathEdit.setReadOnly(True)
        inst_row = QHBoxLayout()
        inst_row.addWidget(self._instprmPathEdit)
        self._instBrowseBtn = QPushButton("Browse…")
        self._instBrowseBtn.clicked.connect(self._browseInstprm)
        inst_row.addWidget(self._instBrowseBtn)
        file_form.addRow("Instrument:", inst_row)

        self._sourceStack.addWidget(file_page)

        # -- Workspace page --
        ws_page = QWidget()
        ws_form = QFormLayout(ws_page)
        ws_form.setContentsMargins(0, 0, 0, 0)

        self._wsNameEdit = QLineEdit()
        self._wsNameEdit.setPlaceholderText("Workspace name in Mantid ADS")
        ws_form.addRow("Workspace:", self._wsNameEdit)

        self._instprmPathEdit2 = QLineEdit()
        self._instprmPathEdit2.setPlaceholderText("Path to .instprm file")
        self._instprmPathEdit2.setReadOnly(True)
        inst_row2 = QHBoxLayout()
        inst_row2.addWidget(self._instprmPathEdit2)
        self._instBrowseBtn2 = QPushButton("Browse…")
        self._instBrowseBtn2.clicked.connect(self._browseInstprm2)
        inst_row2.addWidget(self._instBrowseBtn2)
        ws_form.addRow("Instrument:", inst_row2)

        self._sourceStack.addWidget(ws_page)

        source_layout.addWidget(self._sourceStack)

        # Bank selector
        bank_row = QHBoxLayout()
        bank_row.addWidget(QLabel("Bank:"))
        self._bankSpin = QSpinBox()
        self._bankSpin.setRange(0, 99)
        self._bankSpin.setValue(0)
        bank_row.addWidget(self._bankSpin)
        bank_row.addStretch()
        source_layout.addLayout(bank_row)

        layout.addWidget(source_group)

        # ── Conditions ────────────────────────────────────────────
        cond_group = QGroupBox("Conditions")
        cond_form = QFormLayout(cond_group)

        # Pressure range
        p_row = QHBoxLayout()
        self._pMinSpin = QDoubleSpinBox()
        self._pMinSpin.setRange(0.0, 500.0)
        self._pMinSpin.setValue(0.0)
        self._pMinSpin.setSuffix(" GPa")
        self._pMinSpin.setDecimals(1)
        p_row.addWidget(self._pMinSpin)
        p_row.addWidget(QLabel("to"))
        self._pMaxSpin = QDoubleSpinBox()
        self._pMaxSpin.setRange(0.0, 500.0)
        self._pMaxSpin.setValue(60.0)
        self._pMaxSpin.setSuffix(" GPa")
        self._pMaxSpin.setDecimals(1)
        p_row.addWidget(self._pMaxSpin)
        cond_form.addRow("Pressure:", p_row)

        # Temperature
        self._tempSpin = QDoubleSpinBox()
        self._tempSpin.setRange(0.0, 5000.0)
        self._tempSpin.setValue(295.0)
        self._tempSpin.setSuffix(" K")
        self._tempSpin.setDecimals(1)
        cond_form.addRow("Temperature:", self._tempSpin)

        layout.addWidget(cond_group)
        layout.addStretch()

    # ── Properties ────────────────────────────────────────────────

    @property
    def source_mode(self) -> str:
        """``'file'`` or ``'workspace'``."""
        return "workspace" if self._sourceCombo.currentIndex() == 1 else "file"

    @property
    def gsa_path(self) -> str:
        return self._gsaPathEdit.text()

    @property
    def instprm_path(self) -> str:
        if self.source_mode == "file":
            return self._instprmPathEdit.text()
        return self._instprmPathEdit2.text()

    @property
    def workspace_name(self) -> str:
        return self._wsNameEdit.text()

    @property
    def bank(self) -> int:
        return self._bankSpin.value()

    @property
    def P_min(self) -> float:
        return self._pMinSpin.value()

    @property
    def P_max(self) -> float:
        return self._pMaxSpin.value()

    @property
    def temperature(self) -> float:
        return self._tempSpin.value()

    # ── Slots ─────────────────────────────────────────────────────

    def _onSourceChanged(self, index: int) -> None:
        self._sourceStack.setCurrentIndex(index)

    def _browseGsa(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select spectrum file",
            "",
            "Spectrum files (*.gsa *.csv);;All files (*)",
        )
        if path:
            self._gsaPathEdit.setText(path)
            # Auto-detect matching .instprm
            instprm = Path(path).with_suffix(".instprm")
            if instprm.exists() and not self._instprmPathEdit.text():
                self._instprmPathEdit.setText(str(instprm))
            self.dataChanged.emit()

    def _browseInstprm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select instrument parameter file",
            "",
            "Instrument files (*.instprm);;All files (*)",
        )
        if path:
            self._instprmPathEdit.setText(path)
            self.dataChanged.emit()

    def _browseInstprm2(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select instrument parameter file",
            "",
            "Instrument files (*.instprm);;All files (*)",
        )
        if path:
            self._instprmPathEdit2.setText(path)
            self.dataChanged.emit()
