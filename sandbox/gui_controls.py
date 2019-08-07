from PyQt5.QtWidgets import QPushButton, QSlider, QWidget, QLayout
import PyQt5.uic
from spatial_ops.data import JacksonFischerDataset as jfd, Patient, Plate


class GuiControls(QWidget):
    def __init__(self):
        super().__init__()
        PyQt5.uic.loadUi('sandbox/ui/gui_controls.ui', self)

    def setPatient(self, patient: Patient):
        location_in_array = jfd.patients.index(patient)
        print(location_in_array)
