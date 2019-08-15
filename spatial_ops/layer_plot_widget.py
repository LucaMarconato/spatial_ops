import matplotlib
from pyqtgraph.Qt import QtGui, QtWidgets

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def clear_canvas(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)


class EmptyMplCanvas(MyMplCanvas):
    def compute_initial_figure(self):
        pass


class LayerPlotWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.widget_layout = QtGui.QGridLayout()
        self.setLayout(self.widget_layout)
        self.mpl_canvas = EmptyMplCanvas()
        self.widget_layout.addWidget(self.mpl_canvas)

    def axes(self):
        return self.mpl_canvas.axes

    def fig(self):
        return self.mpl_canvas.fig

    def clear_canvas(self):
        self.mpl_canvas.clear_canvas()


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    widget = LayerPlotWidget()
    widget.show()
    axes = widget.axes()
    fig = widget.fig()
    axes.scatter(list(range(10)), list(range(10)))
    fig.colorbar()
    widget.mpl_canvas.draw()

    sys.exit(app.exec_())

