import pyqtgraph as pg

from sandbox.crosshair_manager import CrosshairManager
from sandbox.lasso_manager import LassoManager


class InteractivePlot:
    def __init__(self, plot_item, interactive_plots_manager):
        self.plot_item = plot_item
        self.already_plotted = False
        self.interactive_plots_manager = interactive_plots_manager
        self.points = []
        self.scatter_plot_item = None
        callback = self.interactive_plots_manager.ome_viewer.highlight_selected_cells
        self.crosshair_manager = CrosshairManager(self.plot_item, callback, self)
        self.lasso_manager = LassoManager(self.plot_item, callback, self)

    def show_scatter_plot(self, data, brushes):
        self.data = data
        if not self.already_plotted:
            self.clear()
            self.already_plotted = True
            self.scatter_plot_item = pg.ScatterPlotItem(size=10,
                                                        pen=pg.mkPen(None))
            self.scatter_plot_item.setData(pos=self.data, brush=brushes)
            self.plot_item.setRange(xRange=[min(self.data[:, 0]), max(self.data[:, 0])],
                                    yRange=[min(self.data[:, 1]), max(self.data[:, 1])])
            self.plot_item.addItem(self.scatter_plot_item)

            self.crosshair_manager.set_enabled(
                self.interactive_plots_manager.ome_viewer.gui_controls.crosshair_radio_button.isChecked())

            self.lasso_manager.set_enabled(
                self.interactive_plots_manager.ome_viewer.gui_controls.lasso_radio_button.isChecked())
        else:
            self.scatter_plot_item.setBrush(brushes)

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
                plot_item = self.addPlot(row=r, col=c)
                interactive_plot = InteractivePlot(plot_item, self)
                self.interactive_plots.append(interactive_plot)

    def clear_plots(self):
        for interactive_plot in self.interactive_plots:
            interactive_plot.clear()

    def clear_lassos(self):
        for interactive_plot in self.interactive_plots:
            interactive_plot.lasso_manager.clear_lasso()

