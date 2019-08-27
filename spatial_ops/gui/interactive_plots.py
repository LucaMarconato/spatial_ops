import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from spatial_ops.gui.crosshair_manager import CrosshairManager
from spatial_ops.gui.lasso_manager import LassoManager


class CustomViewBox(pg.ViewBox):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.interactive_plot = None

    def set_interactive_plot(self, interactive_plot):
        self.interactive_plot = interactive_plot

    def mouseClickEvent(self, ev):
        if self.interactive_plot is not None:
            self.interactive_plot.lasso_manager.my_mouse_click_event(ev)
        if not ev.accepted:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.interactive_plot is not None:
            self.interactive_plot.lasso_manager.my_mouse_drag_event(ev)
        if not ev.accepted:
            super().mouseDragEvent(ev, axis)


class InteractivePlot:
    def __init__(self, plot_item, interactive_plots_manager):
        self.plot_item = plot_item
        self.already_plotted = False
        self.interactive_plots_manager = interactive_plots_manager
        self.points = []
        self.scatter_plot_item = None
        # callback = self.interactive_plots_manager.ome_viewer.highlight_selected_cells
        callback = self.interactive_plots_manager.points_selected
        self.crosshair_manager = CrosshairManager(self.plot_item, callback, self)
        self.lasso_manager = LassoManager(self.plot_item, callback, self)
        self.some_points_currently_highlighted_from_another_plot = False
        self.brushes = []

    def show_scatter_plot(self, data, brushes):
        self.some_points_currently_highlighted_from_another_plot = False
        self.brushes = brushes
        self.data = data
        if not self.already_plotted:
            self.clear()
            self.already_plotted = True
            self.scatter_plot_item = pg.ScatterPlotItem(size=10,
                                                        pen=pg.mkPen(None))
            self.scatter_plot_item.setData(pos=self.data, brush=self.brushes)
            self.plot_item.setRange(xRange=[min(self.data[:, 0]), max(self.data[:, 0])],
                                    yRange=[min(self.data[:, 1]), max(self.data[:, 1])])
            self.plot_item.addItem(self.scatter_plot_item)

            self.crosshair_manager.set_enabled(
                self.interactive_plots_manager.ome_viewer.gui_controls.crosshair_radio_button.isChecked())

            self.lasso_manager.set_enabled(
                self.interactive_plots_manager.ome_viewer.gui_controls.lasso_radio_button.isChecked())
        else:
            self.scatter_plot_item.setBrush(self.brushes)

        self.points = [[self.data[i, 0], self.data[i, 1]] for i in range(self.data.shape[0])]

    def clear(self):
        # this function must be called because the lasso manager class replaces a default method for dealing with
        # the mouse with a custom one, if when creating a new plot (i.e. switching to a new patient) the default
        # method is not restored, then in the case in which the user returns again to the old patient,
        # the default method would still result overridden, making it impossible to pan in the crosshair mode (a
        # lasso is drawn instead)
        self.lasso_manager.set_enabled(False)
        self.already_plotted = False
        if self.scatter_plot_item is not None:
            self.scatter_plot_item.clear()
        self.plot_item.clear()
        self.plot_item.disableAutoRange()
        # to hide the auto range button
        self.plot_item.hideButtons()
        self.some_points_currently_highlighted_from_another_plot = False

    def highlight_points(self, point_indices):
        if not self.already_plotted:
            return
        points_count = len(self.points)
        not_selected_indices = list(set(range(points_count)).difference(point_indices))
        if self.scatter_plot_item is not None and len(self.brushes) > 0:
            self.some_points_currently_highlighted_from_another_plot = True
            new_brushes = self.brushes.copy()
            # for i in point_indices: new_brushes[i] = pg.mkBrush('r') when we color the non selected points in gray,
            # we want that when nothing is selected, then all points get normally visible, instead of having them gray
            if len(point_indices) == 0:
                not_selected_indices = []
                self.some_points_currently_highlighted_from_another_plot = False
            for i in not_selected_indices:
                c = new_brushes[i].color()
                g = QtGui.qGray(c.red(), c.green(), c.blue())
                new_brushes[i] = pg.mkBrush(g, g, g, 120)
                # new_brushes[i] = pg.mkBrush(new_brushes[i].color())
            self.scatter_plot_item.setBrush(new_brushes)
            # sizes = [10.0] * points_count
            # for i in not_selected_indices:
            #     sizes[i] = 3.0
            # self.scatter_plot_item.setSize(sizes)

    def restore_default_brushes(self):
        if self.some_points_currently_highlighted_from_another_plot:
            self.some_points_currently_highlighted_from_another_plot = False
            # self.scatter_plot_item.setSize(10.0)
            self.scatter_plot_item.setBrush(self.brushes)


class InteractivePlotsManager(pg.GraphicsLayoutWidget):
    def __init__(self, rows, cols, ome_viewer, **kargs):
        super().__init__(**kargs)
        self.ome_viewer = ome_viewer
        # remove padding around subplots, does not seem working
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.ci.layout.setSpacing(0)

        self.interactive_plots = []
        for r in range(rows):
            for c in range(cols):
                plot_item = self.addPlot(row=r, col=c, viewBox=CustomViewBox())
                interactive_plot = InteractivePlot(plot_item, self)
                interactive_plot.plot_item.vb.set_interactive_plot(interactive_plot)
                self.interactive_plots.append(interactive_plot)

    def clear_plots(self):
        for interactive_plot in self.interactive_plots:
            interactive_plot.clear()

    def clear_lassos(self):
        for interactive_plot in self.interactive_plots:
            interactive_plot.lasso_manager.clear_lasso()

    def clear_crosshairs(self):
        for interactive_plot in self.interactive_plots:
            interactive_plot.crosshair_manager.clear_crosshair()

    def highlight_points_in_other_plots(self, point_indices, excluded_plot):
        for interactive_plot in self.interactive_plots:
            if interactive_plot != excluded_plot:
                interactive_plot.highlight_points(point_indices)
            else:
                interactive_plot.restore_default_brushes()

    def points_selected(self, points_indices, in_plot):
        self.ome_viewer.highlight_selected_cells(points_indices)
        if in_plot is not None:
            self.highlight_points_in_other_plots(points_indices, excluded_plot=in_plot)
