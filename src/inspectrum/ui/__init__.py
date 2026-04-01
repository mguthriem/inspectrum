"""Inspectrum Qt5 GUI for interactive diffraction spectrum inspection.

Usage from Mantid Workbench script console::

    from inspectrum.ui import show
    show()

The widget can also be launched outside Workbench for development::

    from inspectrum.ui import show_standalone
    show_standalone()
"""

# Module-level reference keeps the dialog alive while it is open.
_active_dialog = None


def show():
    """Open the Inspectrum dialog from Mantid Workbench.

    Safe to call from the Workbench script window.  Uses Mantid's
    ``QAppThreadCall`` to create widgets on the GUI thread.
    """
    from mantidqt.utils.qt.qappthreadcall import QAppThreadCall

    def _open():
        global _active_dialog
        if _active_dialog is not None:
            _active_dialog.raise_()
            _active_dialog.activateWindow()
            return

        from inspectrum.ui.mainWindow import InspectrumWindow

        _active_dialog = InspectrumWindow()
        _active_dialog.show()
        _active_dialog.raise_()
        _active_dialog.activateWindow()

        def _on_destroyed():
            global _active_dialog
            _active_dialog = None

        _active_dialog.destroyed.connect(_on_destroyed)

    QAppThreadCall(_open, blocking=True)()


def show_standalone():
    """Launch the Inspectrum dialog as a standalone application.

    For development and testing without Mantid Workbench.
    Creates its own QApplication if one doesn't exist.
    """
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    own_app = app is None
    if own_app:
        app = QApplication(sys.argv)

    from inspectrum.ui.mainWindow import InspectrumWindow

    window = InspectrumWindow()
    window.show()
    window.raise_()

    if own_app:
        sys.exit(app.exec_())
