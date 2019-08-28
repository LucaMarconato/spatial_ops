import sys

import matplotlib.cm
import numpy as np
from layer_viewer import LayerViewerObject
from layer_viewer.layers import *
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from spatial_ops.gui.gui_controls import GuiControls
from spatial_ops.gui.interactive_plots import InteractivePlotsManager
from spatial_ops.eda.umap_eda import PlateUMAPLoader
from spatial_ops.common.data import JacksonFischerDataset as jfd, Patient, PatientSource
from spatial_ops.nn.vae import VAEUmapLoader


class OmeViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ome viewer')
        self.layer_viewer_object = LayerViewerObject()
        self.ome_layer = None
        self.masks_layer = None

        self.inner_container = QtWidgets.QWidget()
        self.vertical_layout = QtGui.QVBoxLayout()
        self.inner_container.setLayout(self.vertical_layout)
        self.inner_splitter = QtGui.QSplitter()
        self.inner_splitter.setOrientation(QtCore.Qt.Vertical)
        self.vertical_layout.addWidget(self.inner_splitter)
        self.inner_splitter.addWidget(self.layer_viewer_object.layer_ctrl_widget)

        self.gui_controls = GuiControls()
        self.inner_splitter.addWidget(self.gui_controls)

        self.interactive_plots_manager = InteractivePlotsManager(rows=1, cols=2, ome_viewer=self)
        # self.interactive_plots_manager = InteractivePlotsManager(rows=4, cols=4, ome_viewer=self)
        # self.interactive_plots_manager = InteractivePlotsManager(rows=1, cols=3, ome_viewer=self)
        self.inner_splitter.addWidget(self.interactive_plots_manager)

        self.horizontal_layout = QtGui.QHBoxLayout()
        self.setLayout(self.horizontal_layout)
        self.outer_splitter = QtGui.QSplitter()
        self.horizontal_layout.addWidget(self.outer_splitter)
        self.outer_splitter.addWidget(self.layer_viewer_object.layer_view_widget)
        self.outer_splitter.addWidget(self.inner_container)

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
        self.gui_controls.detach_embeddings_check_box.toggled.connect(lambda x: self.detach_embeddings_toggled(x))

        self.set_patient(jfd.patients[20])

        def sync_back(new_channel):
            self.gui_controls.channel_slider.setValue(new_channel)

        self.ome_layer.ctrl_widget().channelSelector.valueChanged.connect(sync_back)

        self.ome_layer.ctrl_widget().channelSelector.setValue(47)
        self.load_settings()

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        self.balance_layout()

    def balance_layout(self):
        sizes = self.inner_splitter.sizes()
        a = sizes[0] + sizes[2]
        ratio = 9.0 / 16.0
        sizes[0] = a * ratio
        sizes[2] = a * (1 - ratio)
        self.inner_splitter.setSizes(sizes)

    def highlight_selected_cells(self, indices):
        # all the plots have the same number of points, because they are different embeddings of the same data
        points_count = len(self.interactive_plots_manager.interactive_plots[0].points)
        not_selected_indices = list(set(range(points_count)).difference(indices))
        if self.masks_layer is None:
            return
        lut = self.masks_layer.lut
        if len(lut) != points_count:
            lut_size = points_count
            s4 = lut_size * 4
            lut = numpy.random.randint(low=0, high=255, size=s4)
            lut = lut.reshape([lut_size, 4])
            # because the first channel is the background
            lut[0, 3] = 0
            lut = lut.astype('int64')
        lut[:, 3] = 255
        if len(not_selected_indices) > 0:
            lut[not_selected_indices, 3] = 0.1

        self.masks_layer.lut = lut
        self.masks_layer.updateData(self.masks_layer.m_data)

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
        # when changing the value of the combobox (possible values basel either zurich),
        # then self.current_patient.pid will change so we then adjust the value of slider which controls the PID
        new_pid = self.current_patient.pid
        if self.current_patient.source == PatientSource.basel:
            self.gui_controls.patient_source_combo_box.setCurrentIndex(0)
        elif self.current_patient.source == PatientSource.zurich:
            self.gui_controls.patient_source_combo_box.setCurrentIndex(1)
        else:
            raise ValueError(f'self.current_patient_source = {self.current_patient.source}')
        self.gui_controls.patient_pid_spin_box.setMaximum(jfd.patient_count_by_source(self.current_patient.source))
        self.gui_controls.patient_pid_spin_box.setValue(new_pid)

        self.current_plate = self.current_patient.plates[0]
        ome = self.current_plate.get_ome()
        self.interactive_plots_manager.clear_plots()
        self.update_embeddings()

        # update the layer viewer to show the ome of the current patient
        if self.ome_layer is None:
            self.ome_layer = MultiChannelImageLayer(name='ome', data=ome[...])
            self.layer_viewer_object.addLayer(layer=self.ome_layer)
        else:
            self.ome_layer.updateData(ome)

        # update the layer viewer to show the masks of the current patient
        masks = self.current_plate.get_masks()
        if self.masks_layer is None:
            self.masks_layer = ObjectLayer(name='mask', data=masks)
            self.layer_viewer_object.addLayer(layer=self.masks_layer)
            self.masks_layer.ctrl_widget().bar.setFraction(0.2)
        else:
            self.masks_layer.updateData(masks)

    def on_channel_slider_value_changed(self, new_channel):
        self.ome_layer.ctrl_widget().channelSelector.setValue(new_channel)
        self.update_channel_label()

    def on_channel_name_combo_box_current_index_changed(self, new_channel):
        self.gui_controls.channel_slider.setValue(new_channel)

    def set_patient(self, patient: Patient):
        index_in_list = jfd.patients.index(patient)
        self.gui_controls.patient_slider.setValue(index_in_list)

    def update_channel_label(self):
        self.gui_controls.channel_name_combo_box.setCurrentIndex(self.gui_controls.channel_slider.value())
        self.update_embeddings()

    def update_embeddings(self):
        _, umap_results, original_data = PlateUMAPLoader(self.current_plate).load_data()
        _, vae_umap_results, _ = VAEUmapLoader(self.current_plate).load_data()
        # _, tsne_results, _ = PlateTSNELoader(self.current_plate).load_data()

        current_channel = self.gui_controls.channel_slider.value()
        a = min(original_data[:, current_channel])
        b = max(original_data[:, current_channel])
        colormap = matplotlib.cm.viridis
        positions = np.linspace(a, b, len(colormap.colors), endpoint=True)
        q_colormap = pg.ColorMap(pos=positions, color=colormap.colors)
        color_for_points = q_colormap.map(original_data[:, current_channel])
        brushes = [QtGui.QBrush(QtGui.QColor(*color_for_points[i, :].tolist())) for i in
                   range(color_for_points.shape[0])]

        self.interactive_plots_manager.interactive_plots[0].show_scatter_plot(umap_results, brushes)
        self.interactive_plots_manager.interactive_plots[1].show_scatter_plot(vae_umap_results, brushes)
        # self.interactive_plots_manager.interactive_plots[1].show_scatter_plot(tsne_results, brushes)

    def crosshair_toggled(self, state):
        for interactive_plot in self.interactive_plots_manager.interactive_plots:
            interactive_plot.crosshair_manager.set_enabled(state)

    def lasso_toggled(self, state):
        for interactive_plot in self.interactive_plots_manager.interactive_plots:
            interactive_plot.lasso_manager.set_enabled(state)

    def detach_embeddings_toggled(self, state):
        class DetachedEmbeddingsWindow(QtGui.QWidget):
            def __init__(self, ome_viewer, parent=None):
                QtGui.QWidget.__init__(self, parent)
                self.ome_viewer = ome_viewer

            def closeEvent(self, event):
                self.ome_viewer.gui_controls.outer_layout.addWidget(
                    self.ome_viewer.gui_controls.embedding_view_properties_group_box)
                self.ome_viewer.inner_splitter.addWidget(self.ome_viewer.interactive_plots_manager)
                self.ome_viewer.balance_layout()
                self.ome_viewer.gui_controls.detach_embeddings_check_box.setChecked(False)
                event.accept()

        if state:
            self.detached_embeddings_window = DetachedEmbeddingsWindow(self)
            l = QtGui.QVBoxLayout()
            self.detached_embeddings_window.setLayout(l)
            l.addWidget(self.gui_controls.embedding_view_properties_group_box)
            l.addWidget(self.interactive_plots_manager)
            self.detached_embeddings_window.show()
        else:
            self.detached_embeddings_window.close()


# start qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()
        viewer = OmeViewer()
        viewer.show()
        sys.exit(app.exec_())
