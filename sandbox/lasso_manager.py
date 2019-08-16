from typing import List, Tuple

import pyqtgraph as pg


class LassoManager:
    def __init__(self, plot_widget: pg.PlotWidget, callback):
        self.plot_widget = plot_widget
        self.callback = callback
        self.enabled = False
        # self.plot_widget.scene().sigMouseMoved.connect(lambda x: self.mouse_moved(x))

        self.original_mouseDragEvent = None
        self.orignal_mouseClickEvent = None

    def my_mouse_click_event(self, ev):
        self.plot_widget.removeItem(self.lasso_plot_item)

    def my_mouse_drag_event(self, ev):
        plot_coord = self.plot_widget.vb.mapSceneToView(ev.scenePos())
        if ev.isStart():
            self.plot_widget.removeItem(self.lasso_plot_item)
            self.lasso_plot_item.clear()
            self.plot_widget.addItem(self.lasso_plot_item)
            self.points_x = [plot_coord.x()]
            self.points_y = [plot_coord.y()]
        elif ev.isFinish():
            self.points_x.append(self.points_x[0])
            self.points_y.append(self.points_y[0])
            self.lasso_plot_item.setData(self.points_x, self.points_y)
        else:
            self.points_x.append(plot_coord.x())
            self.points_y.append(plot_coord.y())
            self.lasso_plot_item.setData(self.points_x, self.points_y)
        ev.accept()

    def add_to_plot(self):
        self.plot_widget.disableAutoRange()
        # to hide the auto range button
        self.plot_widget.hideButtons()
        self.enabled = True

    def mouse_moved(self, event):
        if not self.enabled:
            return
        # # coord = event[0]  ## using signal proxy turns original arguments into a tuple
        # coord = event
        # plot_coord = self.plot_widget.vb.mapSceneToView(coord)
        # plot_coord = (plot_coord.x(), plot_coord.y())
        # self.roi_scatter_plot_item.setData(pos=[plot_coord], brush=pg.mkBrush((255, 255, 255, 100)))
        # pixel_radius = 100.0
        #
        # l_x, l_y = self.mapDistanceSceneToView(pixel_radius)
        # l = [i for i, (x, y) in enumerate(self.points) if
        #      (x - plot_coord[0]) ** 2 / (l_x * l_x * 0.25) + (y - plot_coord[1]) ** 2 / (l_y * l_y * 0.25) < 1]
        # # print(l)
        # # for i in l:
        # #     print(self.points[i])
        # # print(f'l_x = {l_x}, l_y = {l_y}')
        # print(f'plot_coord = {plot_coord}')
        # self.callback(l)

    def set_points(self, points: List[Tuple[int]]):
        self.points = points

    def set_visible(self, visible: bool):
        pass
        # if visible:
        #     # without the list I get a crash and looking at the source code this fixed the problem
        #     self.roi_scatter_plot_item.setBrush([pg.mkBrush(255, 255, 255, 100)])
        # else:
        #     self.roi_scatter_plot_item.setBrush([pg.mkBrush((255, 255, 255, 0))])

    def set_enabled(self, enabled: bool):
        self.set_visible(enabled)
        self.plot_widget.setMouseEnabled(x=not enabled, y=not enabled)
        self.enabled = enabled

        if enabled:
            self.lasso_plot_item = pg.PlotCurveItem(pen=pg.mkPen(width=2, color=(255, 255, 255, 100)))
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
            if self.orignal_mouseClickEvent is not None:
                self.plot_widget.vb.mouseClickEvent = self.orignal_mouseClickEvent
