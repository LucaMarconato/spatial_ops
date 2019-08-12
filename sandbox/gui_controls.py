from PyQt5.QtWidgets import QWidget
import PyQt5.uic
from spatial_ops.data import PatientSource


class GuiControls(QWidget):
    def __init__(self):
        super().__init__()
        PyQt5.uic.loadUi('sandbox/ui/gui_controls.ui', self)
        self.patient_source_combo_box.addItems([PatientSource(0).name, PatientSource(1).name])
        self.patient_pid_spin_box.setMinimum(1)

