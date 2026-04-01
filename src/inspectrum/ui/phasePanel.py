"""Phase input panel — manages crystal phases and their EOS.

Provides:
- Phase list with add/remove
- CIF file loading (browse or drag-and-drop)
- Per-phase EOS editor
- Save/load individual phase-EOS JSON files
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtWidgets import (  # type: ignore
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inspectrum.ui.model import InspectrumModel


class PhasePanel(QWidget):
    """Panel for managing crystal phases and their equations of state.

    Signals:
        phasesChanged: Emitted when the phase list changes.
    """

    phasesChanged = Signal()

    def __init__(
        self,
        model: InspectrumModel,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Phase list ────────────────────────────────────────────
        list_group = QGroupBox("Phases")
        list_layout = QVBoxLayout(list_group)

        self._phaseList = _CIFDropList()
        self._phaseList.setSelectionMode(QListWidget.SingleSelection)
        self._phaseList.currentRowChanged.connect(self._onPhaseSelected)
        self._phaseList.cifDropped.connect(self._addCifFromPath)
        list_layout.addWidget(self._phaseList)

        btn_row = QHBoxLayout()
        self._addCifBtn = QPushButton("+ CIF")
        self._addCifBtn.setToolTip("Add a phase from a CIF file")
        self._addCifBtn.clicked.connect(self._browseCif)
        btn_row.addWidget(self._addCifBtn)

        self._loadJsonBtn = QPushButton("Load JSON")
        self._loadJsonBtn.setToolTip("Load a saved phase-EOS definition")
        self._loadJsonBtn.clicked.connect(self._loadPhaseJson)
        btn_row.addWidget(self._loadJsonBtn)

        self._removeBtn = QPushButton("Remove")
        self._removeBtn.setEnabled(False)
        self._removeBtn.clicked.connect(self._removeSelected)
        btn_row.addWidget(self._removeBtn)

        list_layout.addLayout(btn_row)
        layout.addWidget(list_group)

        # ── Phase detail / EOS editor ─────────────────────────────
        detail_group = QGroupBox("Phase Details")
        self._detailForm = QFormLayout(detail_group)

        self._nameEdit = QLineEdit()
        self._nameEdit.setReadOnly(True)
        self._detailForm.addRow("Name:", self._nameEdit)

        self._sgEdit = QLineEdit()
        self._sgEdit.setReadOnly(True)
        self._detailForm.addRow("Space group:", self._sgEdit)

        self._latticeLabel = QLabel("—")
        self._detailForm.addRow("Lattice:", self._latticeLabel)

        self._roleCombo = QComboBox()
        self._roleCombo.addItems(["sample", "calibrant"])
        self._roleCombo.currentTextChanged.connect(self._onRoleChanged)
        self._detailForm.addRow("Role:", self._roleCombo)

        layout.addWidget(detail_group)

        # ── EOS editor ────────────────────────────────────────────
        eos_group = QGroupBox("Equation of State")
        eos_form = QFormLayout(eos_group)

        self._eosTypeCombo = QComboBox()
        self._eosTypeCombo.addItems([
            "(none)", "birch-murnaghan", "vinet", "murnaghan",
        ])
        self._eosTypeCombo.currentIndexChanged.connect(self._onEosTypeChanged)
        eos_form.addRow("Type:", self._eosTypeCombo)

        self._v0Spin = QDoubleSpinBox()
        self._v0Spin.setRange(0.01, 10000.0)
        self._v0Spin.setDecimals(3)
        self._v0Spin.setSuffix(" ų")
        self._v0Spin.setToolTip("Reference unit-cell volume (ų per cell)")
        eos_form.addRow("V₀:", self._v0Spin)

        self._k0Spin = QDoubleSpinBox()
        self._k0Spin.setRange(0.01, 10000.0)
        self._k0Spin.setDecimals(1)
        self._k0Spin.setSuffix(" GPa")
        eos_form.addRow("K₀:", self._k0Spin)

        self._kpSpin = QDoubleSpinBox()
        self._kpSpin.setRange(0.0, 100.0)
        self._kpSpin.setDecimals(2)
        self._kpSpin.setValue(4.0)
        eos_form.addRow("K′:", self._kpSpin)

        self._eosSourceEdit = QLineEdit()
        self._eosSourceEdit.setPlaceholderText("Literature citation")
        eos_form.addRow("Source:", self._eosSourceEdit)

        eos_btn_row = QHBoxLayout()
        self._applyEosBtn = QPushButton("Apply EOS")
        self._applyEosBtn.clicked.connect(self._applyEos)
        eos_btn_row.addWidget(self._applyEosBtn)

        self._savePhaseBtn = QPushButton("Save JSON")
        self._savePhaseBtn.setToolTip("Save this phase + EOS to a JSON file")
        self._savePhaseBtn.clicked.connect(self._savePhaseJson)
        eos_btn_row.addWidget(self._savePhaseBtn)
        eos_form.addRow(eos_btn_row)

        layout.addWidget(eos_group)
        layout.addStretch()

        # ── Stability range ───────────────────────────────────────
        stab_row = QHBoxLayout()
        stab_row.addWidget(QLabel("Stable:"))
        self._stabMinSpin = QDoubleSpinBox()
        self._stabMinSpin.setRange(0.0, 500.0)
        self._stabMinSpin.setSpecialValueText("any")
        self._stabMinSpin.setSuffix(" GPa")
        stab_row.addWidget(self._stabMinSpin)
        stab_row.addWidget(QLabel("to"))
        self._stabMaxSpin = QDoubleSpinBox()
        self._stabMaxSpin.setRange(0.0, 500.0)
        self._stabMaxSpin.setValue(0.0)
        self._stabMaxSpin.setSpecialValueText("any")
        self._stabMaxSpin.setSuffix(" GPa")
        stab_row.addWidget(self._stabMaxSpin)
        self._applyStabBtn = QPushButton("Set")
        self._applyStabBtn.clicked.connect(self._applyStability)
        stab_row.addWidget(self._applyStabBtn)
        eos_group.layout().addRow("Stability:", stab_row)

    # ── Public ────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the phase list from the model."""
        self._phaseList.clear()
        for desc in self._model.phases:
            eos_tag = f" [{desc.eos.eos_type}]" if desc.eos else ""
            item = QListWidgetItem(
                f"{desc.name} ({desc.role}){eos_tag}"
            )
            self._phaseList.addItem(item)
        self._onPhaseSelected(self._phaseList.currentRow())

    # ── Private ───────────────────────────────────────────────────

    def _addCifFromPath(self, path: str) -> None:
        """Add a phase from a CIF file path."""
        try:
            self._model.add_phase_from_cif(path)
            self.refresh()
            self.phasesChanged.emit()
        except Exception as exc:
            QMessageBox.warning(self, "CIF Error", str(exc))

    def _browseCif(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CIF file(s)",
            "",
            "CIF files (*.cif);;All files (*)",
        )
        for path in paths:
            self._addCifFromPath(path)

    def _loadPhaseJson(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load phase definition",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                self._model.load_phase(path)
                self.refresh()
                self.phasesChanged.emit()
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", str(exc))

    def _removeSelected(self) -> None:
        row = self._phaseList.currentRow()
        if row >= 0:
            self._model.remove_phase(row)
            self.refresh()
            self.phasesChanged.emit()

    def _onPhaseSelected(self, row: int) -> None:
        has_sel = 0 <= row < len(self._model.phases)
        self._removeBtn.setEnabled(has_sel)
        if not has_sel:
            self._nameEdit.clear()
            self._sgEdit.clear()
            self._latticeLabel.setText("—")
            return

        desc = self._model.phases[row]
        self._nameEdit.setText(desc.name)
        phase = desc.phase
        if phase:
            self._sgEdit.setText(
                f"{phase.space_group} (#{phase.space_group_number})"
            )
            self._latticeLabel.setText(
                f"a={phase.a:.4f}  b={phase.b:.4f}  c={phase.c:.4f}  "
                f"α={phase.alpha:.1f}  β={phase.beta:.1f}  γ={phase.gamma:.1f}"
            )
        else:
            self._sgEdit.clear()
            self._latticeLabel.setText("(no CIF loaded)")

        self._roleCombo.blockSignals(True)
        self._roleCombo.setCurrentText(desc.role)
        self._roleCombo.blockSignals(False)

        # Populate EOS fields
        if desc.eos:
            idx = self._eosTypeCombo.findText(desc.eos.eos_type)
            self._eosTypeCombo.setCurrentIndex(max(idx, 0))
            self._v0Spin.setValue(desc.eos.V_0)
            self._k0Spin.setValue(desc.eos.K_0)
            self._kpSpin.setValue(desc.eos.K_prime)
            self._eosSourceEdit.setText(desc.eos.source)
        else:
            self._eosTypeCombo.setCurrentIndex(0)  # (none)
            self._v0Spin.setValue(0.0)
            self._k0Spin.setValue(0.0)
            self._kpSpin.setValue(4.0)
            self._eosSourceEdit.clear()

        # Stability range
        if desc.stability_pressure:
            p_min, p_max = desc.stability_pressure
            self._stabMinSpin.setValue(p_min or 0.0)
            self._stabMaxSpin.setValue(p_max or 0.0)
        else:
            self._stabMinSpin.setValue(0.0)
            self._stabMaxSpin.setValue(0.0)

    def _onRoleChanged(self, role: str) -> None:
        row = self._phaseList.currentRow()
        if 0 <= row < len(self._model.phases):
            self._model.phases[row].role = role
            self.refresh()

    def _onEosTypeChanged(self, _index: int) -> None:
        has_eos = self._eosTypeCombo.currentIndex() > 0
        self._v0Spin.setEnabled(has_eos)
        self._k0Spin.setEnabled(has_eos)
        self._kpSpin.setEnabled(has_eos)
        self._eosSourceEdit.setEnabled(has_eos)

    def _applyEos(self) -> None:
        row = self._phaseList.currentRow()
        if row < 0:
            return

        eos_type = self._eosTypeCombo.currentText()
        if eos_type == "(none)":
            self._model.phases[row].eos = None
        else:
            try:
                self._model.set_eos(
                    row,
                    eos_type=eos_type,
                    V_0=self._v0Spin.value(),
                    K_0=self._k0Spin.value(),
                    K_prime=self._kpSpin.value(),
                    source=self._eosSourceEdit.text(),
                )
            except Exception as exc:
                QMessageBox.warning(self, "EOS Error", str(exc))
                return

        self.refresh()
        self.phasesChanged.emit()

    def _applyStability(self) -> None:
        row = self._phaseList.currentRow()
        if row < 0:
            return
        p_min = self._stabMinSpin.value() or None
        p_max = self._stabMaxSpin.value() or None
        self._model.set_stability_range(row, p_min, p_max)

    def _savePhaseJson(self) -> None:
        row = self._phaseList.currentRow()
        if row < 0:
            return
        desc = self._model.phases[row]
        default_name = f"{desc.name}.json" if desc.name else "phase.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save phase definition",
            default_name,
            "JSON files (*.json);;All files (*)",
        )
        if path:
            try:
                self._model.save_phase(row, path)
            except Exception as exc:
                QMessageBox.warning(self, "Save Error", str(exc))


# ── Drag-and-drop CIF list ───────────────────────────────────────────


class _CIFDropList(QListWidget):
    """QListWidget that accepts CIF files via drag-and-drop."""

    cifDropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".cif"):
                self.cifDropped.emit(path)
        event.acceptProposedAction()
