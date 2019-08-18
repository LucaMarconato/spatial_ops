import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class CrosshairManager:
    def __init__(self, plot_widget: pg.PlotWidget, callback, ome_eda):
        # I would prefer the proxy approach but it does not seem to work
        # proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda x: print('self.mouseMoved'))
        self.plot_widget = plot_widget
        self.callback = callback
        self.enabled = False
        self.plot_widget.scene().sigMouseMoved.connect(lambda x: self.mouse_moved(x))
        # when we call set_disabled(True) we will try to remove the crosshair only if it has been created before
        self.roi_scatter_plot_item = None
        # in pixels
        self.point_diameter = 100
        self.ome_eda = ome_eda

    def mapDistanceSceneToView(self, l: float):
        fake_point0 = QtCore.QPointF(l, l)
        fake_point1 = QtCore.QPointF(0, 0)
        fake_plot_point0 = self.plot_widget.vb.mapSceneToView(fake_point0)
        fake_plot_point1 = self.plot_widget.vb.mapSceneToView(fake_point1)
        l_x = abs(fake_plot_point1.x() - fake_plot_point0.x())
        l_y = abs(fake_plot_point1.y() - fake_plot_point0.y())
        return l_x, l_y

    def mouse_moved(self, event):
        # print(event)
        if not self.enabled:
            return
        # coord = event[0]  ## using signal proxy turns original arguments into a tuple
        coord = event
        plot_coord = self.plot_widget.vb.mapSceneToView(coord)
        plot_coord = (plot_coord.x(), plot_coord.y())

        if self.roi_scatter_plot_item is not None:
            self.plot_widget.removeItem(self.roi_scatter_plot_item)
        self.roi_scatter_plot_item.setData(pos=[plot_coord])
        self.plot_widget.addItem(self.roi_scatter_plot_item)

        l_x, l_y = self.mapDistanceSceneToView(self.point_diameter)
        l = [i for i, (x, y) in enumerate(self.ome_eda.current_points) if
             (x - plot_coord[0]) ** 2 / (l_x * l_x * 0.25) + (y - plot_coord[1]) ** 2 / (l_y * l_y * 0.25) < 1]
        self.callback(l)

    def set_visible(self, visible: bool):
        if not visible:
            if self.roi_scatter_plot_item is not None:
                self.plot_widget.removeItem(self.roi_scatter_plot_item)

    def set_enabled(self, enabled: bool):
        self.set_visible(enabled)
        self.enabled = enabled
        if enabled:
            self.roi_scatter_plot_item = pg.ScatterPlotItem(size=self.point_diameter, pen=pg.mkPen(None),
                                                            brush=pg.mkBrush((255, 255, 255, 100)), pxMode=True)
        else:
            self.callback([])
