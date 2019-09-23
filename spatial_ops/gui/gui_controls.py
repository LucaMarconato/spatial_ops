from PyQt5.QtWidgets import QWidget
import PyQt5.uic
from spatial_ops.common.data import PatientSource
import os

class GuiControls(QWidget):
    def __init__(self):
        super().__init__()
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        PyQt5.uic.loadUi(os.path.join(this_file_path, 'gui_controls.ui'), self)
        self.patient_source_combo_box.addItems([PatientSource(0).name, PatientSource(1).name])
        self.patient_pid_spin_box.setMinimum(1)

