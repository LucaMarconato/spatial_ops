import sys

import matplotlib.cm
import numpy as np
from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *
from pyqtgraph.Qt import QtGui, QtCore

from sandbox.crosshair_manager import CrosshairManager
from sandbox.lasso_manager import LassoManager
from sandbox.gui_controls import GuiControls
from sandbox.umap_eda import PlateUMAPLoader
from spatial_ops.data import JacksonFischerDataset as jfd, Patient, PatientSource


class OmeViewer(LayerViewerWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ome viewer')
        self.ome_layer = None
        self.masks_layer = None
        # on the left the layout we have, on the right the one we want to obtain
        # |--------------|      |--------------|
        # |              |      |        ctrl0 |
        # | image  ctrl0 |      | image  ctrl1 |
        # |              |      |        plot  |
        # |--------------|      |--------------|
        if self.gui_style != 'splitter':
            raise Exception('can only insert the plot widget if the gui of layer viewer is using splitters')
        self.inner_container = QtGui.QWidget()
        self.vhbox = QtGui.QVBoxLayout()
        self.splitter.replaceWidget(1, self.inner_container)
        self.inner_container.setLayout(self.vhbox)
        self.inner_splitter = QtGui.QSplitter()
        self.inner_splitter.setOrientation(QtCore.Qt.Vertical)
        self.vhbox.addWidget(self.inner_splitter)
        self.inner_splitter.addWidget(self.m_layer_ctrl_widget)
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.inner_splitter.addWidget(self.graphics_layout_widget)
        self.plot_widget = self.graphics_layout_widget.addPlot()


        self.gui_controls = GuiControls()
        self.inner_splitter.insertWidget(1, self.gui_controls)
        self.inner_splitter.show()

        sizes = self.inner_splitter.sizes()
        a = sizes[0] + sizes[2]
        ratio = 9.0 / 16.0
        sizes[0] = a * ratio
        sizes[2] = a * (1 - ratio)
        self.inner_splitter.setSizes(sizes)

        self.crosshair_manager = CrosshairManager(self.plot_widget, self.highlight_selected_cells)
        self.lasso_manager = LassoManager(self.plot_widget, self.highlight_selected_cells)

        patients_count = len(jfd.patients)
        self.gui_controls.patient_slider.setMaximum(patients_count - 1)
        channels_annotation = jfd.get_channels_annotation()
        channels_count = len(channels_annotation)
        self.gui_controls.channel_slider.setMaximum(channels_count - 1)
        for k, v in channels_annotation.items():
            self.gui_controls.channel_name_combo_box.addItem(v)

        self.gui_controls.patient_slider.valueChanged.connect(lambda x: self.on_patient_slider_value_changed(x))
        self.gui_controls.channel_slider.valueChanged.connect(lambda x: self.on_channel_slider_value_changed(x))
        self.gui_controls.patient_source_combo_box.currentIndexChanged.connect(
            lambda x: self.on_patient_source_combo_box_current_index_changed(x))
        self.gui_controls.patient_pid_spin_box.valueChanged.connect(
            lambda x: self.on_patient_pid_spin_box_value_changed(x))
        self.gui_controls.channel_name_combo_box.currentIndexChanged.connect(
            lambda x: self.on_channel_name_combo_box_current_index_changed(x)
        )
        self.gui_controls.crosshair_radio_button.toggled.connect(lambda x: self.crosshair_toggled(x))
        self.gui_controls.lasso_radio_button.toggled.connect(lambda x: self.lasso_toggled(x))

        self.set_patient(jfd.patients[20])

        def sync_back(new_channel):
            self.gui_controls.channel_slider.setValue(new_channel)

        self.ome_layer.ctrl_widget().channel_selector.valueChanged.connect(sync_back)

        self.ome_layer.ctrl_widget().channel_selector.setValue(47)
        self.load_settings()

    def highlight_selected_cells(self, indices):
        indices_not_selected = list(set(range(len(self.crosshair_manager.points))).difference(indices))
        lut = self.masks_layer.lut
        if len(lut) != len(self.crosshair_manager.points):
            lut_size = len(self.crosshair_manager.points)
            s4 = lut_size * 4
            lut = numpy.random.randint(low=0, high=255, size=s4)
            lut = lut.reshape([lut_size, 4])
            # because the first channel is the background
            lut[0, 3] = 0
            lut = lut.astype('int64')
        lut[:, 3] = 255
        if len(indices_not_selected) > 0:
            lut[indices_not_selected, 3] = 0.1

        self.masks_layer.lut = lut
        self.masks_layer.update_data(self.masks_layer.m_data)

    def load_settings(self):
        settings = QtCore.QSettings('B260', 'spatial_ops')
        patient_source = settings.value('patient_source', PatientSource(0).value, int)
        patient_pid = settings.value('patient_pid', 1, int)
        self.gui_controls.patient_source_combo_box.setCurrentIndex(patient_source)
        self.gui_controls.patient_pid_spin_box.setValue(patient_pid)

    def save_settings(self):
        settings = QtCore.QSettings('B260', 'spatial_ops')
        settings.setValue('patient_source', self.current_patient.source.value)
        settings.setValue('patient_pid', self.current_patient.pid)

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

    def on_patient_source_combo_box_current_index_changed(self, new_value: int):
        patient_source = PatientSource(new_value)
        i = jfd.get_patient_index_by_source_and_pid(patient_source, 1)
        self.gui_controls.patient_slider.setValue(i)

    def on_patient_pid_spin_box_value_changed(self, new_value: int):
        j = self.gui_controls.patient_source_combo_box.currentIndex()
        patient_source = PatientSource(j)
        i = jfd.get_patient_index_by_source_and_pid(patient_source, new_value)
        self.gui_controls.patient_slider.setValue(i)

    def on_patient_slider_value_changed(self, patient_index: int):
        self.current_patient = jfd.patients[patient_index]
        patient_information = f'source: {self.current_patient.source}, pid: {self.current_patient.pid}'
        # when changing the value of the combobox self.current_patient.pid will change
        new_pid = self.current_patient.pid
        # we create one scatterplot of the umap of a specific channel for the current patient, when we change channel
        # we do not need to recreate the scatterplot but just to change the colormap
        self.first_umap_shown = False
        if self.current_patient.source == PatientSource.basel:
            self.gui_controls.patient_source_combo_box.setCurrentIndex(0)
            self.gui_controls.patient_pid_spin_box.setValue(new_pid)
        elif self.current_patient.source == PatientSource.zurich:
            self.gui_controls.patient_source_combo_box.setCurrentIndex(1)
            self.gui_controls.patient_pid_spin_box.setValue(new_pid)
        else:
            raise ValueError(f'self.current_patient_source = {self.current_patient.source}')
        self.gui_controls.patient_pid_spin_box.setMaximum(jfd.patient_count_by_source(self.current_patient.source))
        self.gui_controls.patient_pid_spin_box.setValue(self.current_patient.pid)

        self.current_plate = self.current_patient.plates[0]
        ome = self.current_plate.get_ome()
        self.update_umap()

        if self.ome_layer is None:
            self.ome_layer = MultiChannelImageLayer(name='ome', data=ome[...])
            self.addLayer(layer=self.ome_layer)
        else:
            self.ome_layer.update_data(ome)

        masks = self.current_plate.get_masks()

        if self.masks_layer is None:
            self.masks_layer = ObjectLayer(name='mask', data=masks)
            self.add_layer(layer=self.masks_layer)
            self.masks_layer.ctrl_widget().bar.set_fraction(0.2)
        else:
            self.masks_layer.update_data(masks)

    def on_channel_slider_value_changed(self, new_channel):
        self.ome_layer.ctrl_widget().channel_selector.setValue(new_channel)
        self.update_channel_label()

    def on_channel_name_combo_box_current_index_changed(self, new_channel):
        self.gui_controls.channel_slider.setValue(new_channel)

    def set_patient(self, patient: Patient):
        index_in_list = jfd.patients.index(patient)
        self.gui_controls.patient_slider.setValue(index_in_list)

    def update_channel_label(self):
        self.gui_controls.channel_name_combo_box.setCurrentIndex(self.gui_controls.channel_slider.value())
        self.update_umap()

    def update_umap(self):
        reducer, umap_results, original_data = PlateUMAPLoader(self.current_plate).load_data()
        current_channel = self.gui_controls.channel_slider.value()
        a = min(original_data[:, current_channel])
        b = max(original_data[:, current_channel])
        colormap = matplotlib.cm.viridis
        positions = np.linspace(a, b, len(colormap.colors), endpoint=True)
        q_colormap = pg.ColorMap(pos=positions, color=colormap.colors)
        color_for_points = q_colormap.map(original_data[:, current_channel])
        brushes = [QtGui.QBrush(QtGui.QColor(*color_for_points[i, :].tolist())) for i in
                   range(color_for_points.shape[0])]
        if not self.first_umap_shown:
            self.first_umap_shown = True
            self.scatter_plot_item = pg.ScatterPlotItem(size=10,
                                                        pen=pg.mkPen(None))
            self.scatter_plot_item.setData(pos=umap_results, brush=brushes)
            self.plot_widget.clear()
            self.plot_widget.setRange(xRange=[min(umap_results[:, 0]), max(umap_results[:, 0])],
                                      yRange=[min(umap_results[:, 1]), max(umap_results[:, 1])])
            self.plot_widget.addItem(self.scatter_plot_item)
            self.crosshair_manager.add_to_plot()
            self.lasso_manager.add_to_plot()

            self.crosshair_manager.set_enabled(self.gui_controls.crosshair_radio_button.isChecked())
            self.lasso_manager.set_enabled(self.gui_controls.lasso_radio_button.isChecked())
        else:
            self.scatter_plot_item.setBrush(brushes)

        points = [[umap_results[i, 0], umap_results[i, 1]] for i in range(umap_results.shape[0])]
        self.crosshair_manager.set_points(points)

    def crosshair_toggled(self, state):
        self.crosshair_manager.set_enabled(state)

    def lasso_toggled(self, state):
        self.lasso_manager.set_enabled(state)


# start qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()
        viewer = OmeViewer()
        viewer.show()
        sys.exit(app.exec_())
