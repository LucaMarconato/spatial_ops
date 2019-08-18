from typing import List, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class LassoManager:
    def __init__(self, plot_widget: pg.PlotWidget, callback, ome_eda):
        self.plot_widget = plot_widget
        self.callback = callback
        self.enabled = False

        self.original_mouseDragEvent = None
        self.original_mouseClickEvent = None
        self.lasso_plot_item = None
        self.ome_eda = ome_eda

    def my_mouse_click_event(self, ev):
        self.plot_widget.removeItem(self.lasso_plot_item)

    def my_mouse_drag_event(self, ev):
        plot_coord = self.plot_widget.vb.mapSceneToView(ev.scenePos())

        def link_first_and_last():
            self.points_x.append(self.points_x[0])
            self.points_y.append(self.points_y[0])

        def remove_last():
            self.points_x.pop(-1)
            self.points_y.pop(-1)

        if ev.isStart():
            self.plot_widget.removeItem(self.lasso_plot_item)
            self.lasso_plot_item.clear()
            self.plot_widget.addItem(self.lasso_plot_item)
            self.points_x = [plot_coord.x()]
            self.points_y = [plot_coord.y()]
            link_first_and_last()
            self.path = QtGui.QPainterPath(plot_coord)
        elif ev.isFinish():
            self.path.closeSubpath()
            contained = []
            for i, point in enumerate(self.ome_eda.current_points):
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

    def set_visible(self, visible: bool):
        self.plot_widget.removeItem(self.lasso_plot_item)

        # if visible:
        #     # without the list I get a crash and looking at the source code this fixed the problem
        #     self.roi_scatter_plot_item.setBrush([pg.mkBrush(255, 255, 255, 100)])
        # else:
        #     self.roi_scatter_plot_item.setBrush([pg.mkBrush((255, 255, 255, 0))])

    def set_enabled(self, enabled: bool):
        self.set_visible(enabled)
        # self.plot_widget.setMouseEnabled(x=not enabled, y=not enabled)
        self.enabled = enabled
        if enabled:
            self.lasso_plot_item = pg.PlotCurveItem(
                pen=pg.mkPen(width=4, color=(255, 255, 255, 100), style=QtCore.Qt.DashLine))
            self.plot_widget.addItem(self.lasso_plot_item)

            def f(ev):
                self.my_mouse_drag_event(ev)

            self.original_mouseDragEvent = self.plot_widget.vb.mouseDragEvent
            self.plot_widget.vb.mouseDragEvent = f

            def f(ev):
                self.my_mouse_click_event(ev)

            self.original_mouseClickEvent = self.plot_widget.vb.mouseClickEvent
            self.plot_widget.vb.mouseClickEvent = f
        else:
            if self.original_mouseDragEvent is not None:
                self.plot_widget.vb.mouseDragEvent = self.original_mouseDragEvent
            if self.original_mouseClickEvent is not None:
                self.plot_widget.vb.mouseClickEvent = self.original_mouseClickEvent
            self.callback([])
