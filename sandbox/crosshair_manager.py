from typing import List, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class CrosshairManager:
    def __init__(self, plot_widget: pg.PlotWidget, callback):
        # I would prefer the proxy approach but it does not seem to work
        # proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda x: print('self.mouseMoved'))
        self.plot_widget = plot_widget
        self.callback = callback

        point_diameter = 100.0
        self.plot_widget.disableAutoRange()
        # to hide the auto range button
        self.plot_widget.hideButtons()
        # this hack is used to remove previous connections, and avoid the error that is triggered when still no connection has been made
        try:
            self.plot_widget.scene().sigMouseMoved.disconnect()
        except TypeError:
            print('a')
            pass
        self.plot_widget.scene().sigMouseMoved.connect(lambda x: self.mouse_moved(x))

        self.roi_scatter_plot_item = pg.ScatterPlotItem(size=point_diameter, pen=pg.mkPen(None),
                                                        brush=pg.mkBrush((255, 255, 255, 100)), pxMode=True)
        self.roi_scatter_plot_item.setData(pos=[(0, 0)])
        self.plot_widget.addItem(self.roi_scatter_plot_item)  # , pxMode=True

    def mapDistanceSceneToView(self, l: float):
        fake_point0 = QtCore.QPointF(l, l)
        fake_point1 = QtCore.QPointF(0, 0)
        fake_plot_point0 = self.plot_widget.vb.mapSceneToView(fake_point0)
        fake_plot_point1 = self.plot_widget.vb.mapSceneToView(fake_point1)
        l_x = abs(fake_plot_point1.x() - fake_plot_point0.x())
        l_y = abs(fake_plot_point1.y() - fake_plot_point0.y())
        return l_x, l_y

    def mouse_moved(self, event):
        # coord = event[0]  ## using signal proxy turns original arguments into a tuple
        coord = event
        plot_coord = self.plot_widget.vb.mapSceneToView(coord)
        plot_coord = (plot_coord.x(), plot_coord.y())
        self.roi_scatter_plot_item.setData(pos=[plot_coord])
        pixel_radius = 100.0

        l_x, l_y = self.mapDistanceSceneToView(pixel_radius)
        l = [i for i, (x, y) in enumerate(self.points) if
             (x - plot_coord[0]) ** 2 / (l_x * l_x * 0.25) + (y - plot_coord[1]) ** 2 / (l_y * l_y * 0.25) < 1]
        # print(l)
        # for i in l:
        #     print(self.points[i])
        # print(f'l_x = {l_x}, l_y = {l_y}')
        # print(f'plot_coord = {plot_coord}')
        self.callback(l)

    def set_points(self, points: List[Tuple[int]]):
        self.points = points
