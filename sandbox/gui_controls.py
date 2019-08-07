from PyQt5.QtWidgets import QWidget
import PyQt5.uic


class GuiControls(QWidget):
    def __init__(self):
        super().__init__()
        PyQt5.uic.loadUi('sandbox/ui/gui_controls.ui', self)

