from typing import List, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import random
import string

class LassoManager:
    def __init__(self, plot_item, callback, interactive_plot):
        self.plot_item = plot_item
        self.callback = callback

        self.original_mouseDragEvent = None
        self.lasso_plot_item = None
        self.interactive_plot = interactive_plot

    # def my_mouse_click_event(self, ev):
    #     print(f'{random.choice(string.ascii_letters)} mouse click')
    #     self.interactive_plot.interactive_plots_manager.clear_lassos()
    #     self.callback([])

    def my_mouse_drag_event(self, ev):
        print(f'{random.choice(string.ascii_letters)} mouse drag')
        plot_coord = self.plot_item.vb.mapSceneToView(ev.scenePos())

        def link_first_and_last():
            self.points_x.append(self.points_x[0])
            self.points_y.append(self.points_y[0])

        def remove_last():
            self.points_x.pop(-1)
            self.points_y.pop(-1)

        if ev.isStart():
            self.interactive_plot.interactive_plots_manager.clear_lassos()
            self.plot_item.addItem(self.lasso_plot_item)
            self.points_x = [plot_coord.x()]
            self.points_y = [plot_coord.y()]
            link_first_and_last()
            self.path = QtGui.QPainterPath(plot_coord)
        elif ev.isFinish():
            self.path.closeSubpath()
            contained = []
            for i, point in enumerate(self.interactive_plot.points):
                q_point = QtCore.QPointF(point[0], point[1])
                if self.path.contains(q_point):
                    contained.append(i)
            self.callback(contained)
        else:
            remove_last()
            self.points_x.append(plot_coord.x())
            self.points_y.append(plot_coord.y())
            link_first_and_last()
            self.lasso_plot_item.setData(self.points_x, self.points_y)
            self.path.lineTo(plot_coord)
        ev.accept()

    def set_points(self, points: List[Tuple[int]]):
        self.points = points

    def clear_lasso(self):
        self.plot_item.removeItem(self.lasso_plot_item)
        if self.lasso_plot_item is not None:
            self.lasso_plot_item.clear()

    def set_enabled(self, enabled: bool):
        if enabled:
            self.lasso_plot_item = pg.PlotCurveItem(
                pen=pg.mkPen(width=4, color=(255, 255, 255, 100), style=QtCore.Qt.DashLine))
            self.plot_item.addItem(self.lasso_plot_item)

            def f(ev):
                self.my_mouse_drag_event(ev)

            self.original_mouseDragEvent = self.plot_item.vb.mouseDragEvent
            self.plot_item.vb.mouseDragEvent = f
        else:
            self.clear_lasso()
            if self.original_mouseDragEvent is not None:
                self.plot_item.vb.mouseDragEvent = self.original_mouseDragEvent
            self.callback([])
