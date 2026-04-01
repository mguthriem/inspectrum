"""Background worker for running the inspectrum pipeline off the GUI thread."""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal  # type: ignore

from inspectrum.models import InspectionResult
from inspectrum.ui.model import InspectrumModel


class InspectionWorker(QObject):
    """Runs :meth:`InspectrumModel.run_inspection` on a background thread.

    Emits ``finished`` with the result, or ``error`` with a message
    if the pipeline fails.

    Usage::

        thread = QThread()
        worker = InspectionWorker(model)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_result)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.start()
    """

    finished = Signal(object)  # InspectionResult
    error = Signal(str)

    def __init__(self, model: InspectrumModel) -> None:
        super().__init__()
        self._model = model

    def run(self) -> None:
        """Execute the pipeline.  Called by QThread.started."""
        try:
            result = self._model.run_inspection()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
