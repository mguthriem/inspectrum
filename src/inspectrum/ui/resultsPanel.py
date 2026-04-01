"""Results panel — embedded matplotlib plot and summary table.

Displays the spectrum with phase-match overlays and a table of
per-phase lattice parameters after the pipeline runs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy.QtCore import Qt  # type: ignore
from qtpy.QtWidgets import (  # type: ignore
    QGroupBox,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inspectrum.models import InspectionResult
from inspectrum.plotting import plot_phase_matches


class ResultsPanel(QWidget):
    """Panel displaying the inspection results.

    Contains:
    - An embedded matplotlib figure reusing ``plot_phase_matches``.
    - A summary table with per-phase lattice parameters, pressure,
      number of matched peaks, and residual.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)

        # ── Matplotlib canvas ─────────────────────────────────────
        self._figure = Figure(figsize=(8, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas)
        splitter.addWidget(plot_widget)

        # ── Summary table ─────────────────────────────────────────
        table_group = QGroupBox("Phase Summary")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Phase", "a (Å)", "b (Å)", "c (Å)",
            "V (ų)", "P (GPa)", "# peaks",
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # ── Public API ────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear the plot and table."""
        self._figure.clear()
        self._canvas.draw_idle()
        self._table.setRowCount(0)

    def show_result(self, result: InspectionResult) -> None:
        """Render an InspectionResult in the plot and table.

        Args:
            result: Pipeline output from ``engine.inspect()``.
        """
        self._plot_result(result)
        self._fill_table(result)

    # ── Private ───────────────────────────────────────────────────

    def _plot_result(self, result: InspectionResult) -> None:
        """Re-draw the matplotlib figure with phase matches."""
        self._figure.clear()

        if result.match_result is None or result.peak_table is None:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "No match result",
                ha="center", va="center", transform=ax.transAxes,
            )
            self._canvas.draw_idle()
            return

        # We need the spectrum+bg_subtracted signal.  Engine stores them
        # in metadata when available.
        bg_subtracted = result.metadata.get("bg_subtracted")
        spectrum = result.metadata.get("spectrum")

        if bg_subtracted is None or spectrum is None:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "Missing spectrum data in result metadata",
                ha="center", va="center", transform=ax.transAxes,
            )
            self._canvas.draw_idle()
            return

        # Collect per-phase reflections from metadata (if stored)
        phase_reflections = result.metadata.get("phase_reflections")

        ax = self._figure.add_subplot(111)
        plot_phase_matches(
            spectrum,
            bg_subtracted,
            result.match_result,
            observed_positions=result.peak_table,
            phase_reflections=phase_reflections,
            refinements=result.refinements if result.refinements else None,
            ax=ax,
        )
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _fill_table(self, result: InspectionResult) -> None:
        """Populate the summary table with refinement results."""
        refs = result.refinements or []
        self._table.setRowCount(len(refs))

        for row, ref in enumerate(refs):
            self._table.setItem(row, 0, QTableWidgetItem(ref.phase_name))
            self._table.setItem(
                row, 1, QTableWidgetItem(f"{ref.a:.5f}" if ref.success else "—"),
            )
            self._table.setItem(
                row, 2, QTableWidgetItem(f"{ref.b:.5f}" if ref.success else "—"),
            )
            self._table.setItem(
                row, 3, QTableWidgetItem(f"{ref.c:.5f}" if ref.success else "—"),
            )
            vol = f"{ref.volume:.2f}" if ref.success and ref.volume else "—"
            self._table.setItem(row, 4, QTableWidgetItem(vol))
            p = f"{ref.pressure_gpa:.2f}" if ref.pressure_gpa is not None else "—"
            self._table.setItem(row, 5, QTableWidgetItem(p))

            # Number of matched peaks for this phase
            n_matched = "—"
            if result.match_result is not None:
                for pm in result.match_result.phase_matches:
                    if pm.phase_name == ref.phase_name:
                        n_matched = str(pm.n_matched)
                        break
            self._table.setItem(row, 6, QTableWidgetItem(n_matched))
