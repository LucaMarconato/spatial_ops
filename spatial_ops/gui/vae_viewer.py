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
from spatial_ops.nn.vae import VAEEmbeddingAndReconstructionLoader


class VaeViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # place widgets
        self.setWindowTitle('vae viewer')

        self.inner_container_left = QtWidgets.QWidget()
        self.inner_container_center = QtWidgets.QWidget()
        self.inner_container_right = QtWidgets.QWidget()

        self.vertical_layout_left = QtGui.QVBoxLayout()
        self.vertical_layout_center = QtGui.QVBoxLayout()
        self.vertical_layout_right = QtGui.QVBoxLayout()

        self.inner_splitter_left = QtGui.QSplitter()
        self.inner_splitter_center = QtGui.QSplitter()
        self.inner_splitter_right = QtGui.QSplitter()
        self.inner_splitter_left.setOrientation(QtCore.Qt.Vertical)
        self.inner_splitter_center.setOrientation(QtCore.Qt.Vertical)
        self.inner_splitter_right.setOrientation(QtCore.Qt.Vertical)

        self.inner_container_left.setLayout(self.vertical_layout_left)
        self.inner_container_center.setLayout(self.vertical_layout_center)
        self.inner_container_right.setLayout(self.vertical_layout_right)

        self.vertical_layout_left.addWidget(self.inner_splitter_left)
        self.vertical_layout_center.addWidget(self.inner_splitter_center)
        self.vertical_layout_right.addWidget(self.inner_splitter_right)

        self.layer_viewer_object_left = LayerViewerObject()
        self.layer_viewer_object_right = LayerViewerObject()

        self.inner_splitter_left.addWidget(self.layer_viewer_object_left.layer_view_widget)
        self.inner_splitter_left.addWidget(self.layer_viewer_object_left.layer_ctrl_widget)
        self.inner_splitter_right.addWidget(self.layer_viewer_object_right.layer_view_widget)
        self.inner_splitter_right.addWidget(self.layer_viewer_object_right.layer_ctrl_widget)

        self.gui_controls = GuiControls()
        self.interactive_plots_manager = InteractivePlotsManager(rows=2, cols=2, ome_viewer=self)
        self.inner_splitter_center.addWidget(self.interactive_plots_manager)
        self.inner_splitter_center.addWidget(self.gui_controls)
        self.inner_splitter_center.insertWidget(0, self.gui_controls.embedding_view_properties_group_box)

        self.outer_layout = QtGui.QHBoxLayout()
        self.outer_splitter = QtGui.QSplitter()
        self.setLayout(self.outer_layout)
        self.outer_layout.addWidget(self.outer_splitter)
        self.outer_splitter.addWidget(self.inner_container_left)
        self.outer_splitter.addWidget(self.inner_container_center)
        self.outer_splitter.addWidget(self.inner_container_right)

        # connect widgets
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
        self.gui_controls.always_show_cells_check_box.toggled.connect(lambda x: self.always_show_cells_toggled(x))

        self.ome_layer_left = None
        self.ome_layer_right = None
        self.masks_layer_left = None
        self.masks_layer_right = None
        # calling set_patient initializes the four above variables
        self.set_patient(jfd.patients[20])

        def sync_back(new_channel):
            self.gui_controls.channel_slider.setValue(new_channel)

        self.ome_layer_left.ctrl_widget().channelSelector.valueChanged.connect(sync_back)
        self.ome_layer_right.ctrl_widget().channelSelector.valueChanged.connect(sync_back)

        # sync the two layer viewer
        def sync_ome(new_channel):
            self.ome_layer_left.ctrl_widget().channelSelector.setValue(new_channel)
            self.ome_layer_right.ctrl_widget().channelSelector.setValue(new_channel)

        self.ome_layer_left.ctrl_widget().channelSelector.valueChanged.connect(sync_ome)
        self.ome_layer_right.ctrl_widget().channelSelector.valueChanged.connect(sync_ome)

        def sync_masks(new_channel):
            self.masks_layer_left.ctrl_widget().channelSelector.setValue(new_channel)
            self.masks_layer_right.ctrl_widget().channelSelector.setValue(new_channel)

        self.masks_layer_left.ctrl_widget().channelSelector.valueChanged.connect(sync_masks)
        self.masks_layer_right.ctrl_widget().channelSelector.valueChanged.connect(sync_masks)

        self.ome_layer_left.ctrl_widget().channelSelector.setValue(28)
        self.load_settings()

    def highlight_selected_cells(self, indices):
        # all the plots have the same number of points, because they are different embeddings of the same data
        points_count = len(self.interactive_plots_manager.interactive_plots[0].points)
        not_selected_indices = list(set(range(points_count)).difference(indices))

        def f(masks_layer):
            nonlocal not_selected_indices
            if masks_layer is None:
                return
            lut = masks_layer.lut
            if len(lut) != points_count:
                lut_size = points_count
                s4 = lut_size * 4
                lut = np.zeros(s4)
                lut = lut.reshape([lut_size, 4])
                lut = lut.astype('int64')
            lut[:, 3] = 255

            if self.gui_controls.always_show_cells_check_box.isChecked():
                if len(not_selected_indices) == points_count:
                    not_selected_indices = []

            if len(not_selected_indices) > 0:
                lut[not_selected_indices, 3] = 0

            masks_layer.lut = lut
            masks_layer.updateData(masks_layer.m_data)

        f(self.masks_layer_left)
        f(self.masks_layer_right)

    def load_settings(self):
        settings = QtCore.QSettings('B260', 'spatial_ops_vae_viewer')
        patient_source = settings.value('patient_source', PatientSource(0).value, int)
        patient_pid = settings.value('patient_pid', 1, int)
        self.gui_controls.patient_source_combo_box.setCurrentIndex(patient_source)
        self.gui_controls.patient_pid_spin_box.setValue(patient_pid)

    def save_settings(self):
        settings = QtCore.QSettings('B260', 'spatial_ops_vae_viewer')
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
        self.update_embeddings_and_cells()

        # update the layer viewer to show the ome of the current patient
        if self.ome_layer_left is None:
            self.ome_layer_left = MultiChannelImageLayer(name='ome', data=ome[...])
            self.layer_viewer_object_left.addLayer(layer=self.ome_layer_left)
        else:
            self.ome_layer_left.updateData(ome)
        # do the same for the right widget
        if self.ome_layer_right is None:
            self.ome_layer_right = MultiChannelImageLayer(name='ome', data=ome[...])
            self.layer_viewer_object_right.addLayer(layer=self.ome_layer_right)
        else:
            self.ome_layer_right.updateData(ome)

        # update the layer viewer to show the masks of the current patient
        masks = self.current_plate.get_masks()
        if self.masks_layer_left is None:
            self.masks_layer_left = ObjectLayer(name='mask', data=masks)
            self.layer_viewer_object_left.addLayer(layer=self.masks_layer_left)
            # self.masks_layer_left.setOpacity(0.2)
        else:
            self.masks_layer_left.updateData(masks)
        # do the same for the right widget
        if self.masks_layer_right is None:
            self.masks_layer_right = ObjectLayer(name='mask', data=masks)
            self.layer_viewer_object_right.addLayer(layer=self.masks_layer_right)
            # self.masks_layer_right.setOpacity(0.2)
        else:
            self.masks_layer_right.updateData(masks)
        self.highlight_selected_cells([])

    def on_channel_slider_value_changed(self, new_channel):
        self.ome_layer_left.ctrl_widget().channelSelector.setValue(new_channel)
        self.update_channel_label()
        self.highlight_selected_cells([])

    def on_channel_name_combo_box_current_index_changed(self, new_channel):
        self.gui_controls.channel_slider.setValue(new_channel)

    def set_patient(self, patient: Patient):
        index_in_list = jfd.patients.index(patient)
        self.gui_controls.patient_slider.setValue(index_in_list)

    def update_channel_label(self):
        self.gui_controls.channel_name_combo_box.setCurrentIndex(self.gui_controls.channel_slider.value())
        self.update_embeddings_and_cells()

    def color_for_data(self, data, a=None, b=None):
        if a is None:
            a = min(data)
        if b is None:
            b = max(data)
        colormap = matplotlib.cm.viridis
        positions = np.linspace(a, b, len(colormap.colors), endpoint=True)
        q_colormap = pg.ColorMap(pos=positions, color=colormap.colors)
        colors = q_colormap.map(data)
        return colors

    def update_embeddings_and_cells(self):
        _, umap_results, _ = PlateUMAPLoader(self.current_plate).load_data()
        original_data, vae_umap_results, reconstructed_data = VAEEmbeddingAndReconstructionLoader(
            self.current_plate).load_data()
        current_channel = self.gui_controls.channel_slider.value()
        x = original_data[:, current_channel]
        reconstructed_x = reconstructed_data[:, current_channel]
        a = min(min(x), min(reconstructed_x))
        b = max(max(x), max(reconstructed_x))
        color_for_points_left = self.color_for_data(x, a, b)
        color_for_points_right = self.color_for_data(reconstructed_x, a, b)

        # update embeddings
        brushes_left = [QtGui.QBrush(QtGui.QColor(*color_for_points_left[i, :].tolist())) for i in
                        range(color_for_points_left.shape[0])]
        brushes_right = [QtGui.QBrush(QtGui.QColor(*color_for_points_right[i, :].tolist())) for i in
                        range(color_for_points_right.shape[0])]

        self.interactive_plots_manager.interactive_plots[0].show_scatter_plot(umap_results, brushes_left)
        self.interactive_plots_manager.interactive_plots[1].show_scatter_plot(umap_results, brushes_right)
        self.interactive_plots_manager.interactive_plots[2].show_scatter_plot(vae_umap_results, brushes_left)
        self.interactive_plots_manager.interactive_plots[3].show_scatter_plot(vae_umap_results, brushes_right)

        # update cells
        zeros = np.zeros([len(color_for_points_left), 1])
        lut_left = np.append(color_for_points_left, zeros, axis=1)
        if self.masks_layer_left is not None:
            self.masks_layer_left.lut = lut_left
            self.masks_layer_left.updateData(self.masks_layer_left.m_data)

        lut_right = np.append(color_for_points_right, zeros, axis=1)
        if self.masks_layer_right is not None:
            self.masks_layer_right.lut = lut_right
            self.masks_layer_right.updateData(self.masks_layer_right.m_data)

    def crosshair_toggled(self, state):
        for interactive_plot in self.interactive_plots_manager.interactive_plots:
            interactive_plot.crosshair_manager.set_enabled(state)

    def lasso_toggled(self, state):
        for interactive_plot in self.interactive_plots_manager.interactive_plots:
            interactive_plot.lasso_manager.set_enabled(state)

    def always_show_cells_toggled(self, state):
        self.interactive_plots_manager.clear_lassos()
        self.interactive_plots_manager.clear_crosshairs()
        self.highlight_selected_cells([])

    def detach_embeddings_toggled(self, state):
        class DetachedEmbeddingsWindow(QtGui.QWidget):
            def __init__(self, ome_viewer, parent=None):
                QtGui.QWidget.__init__(self, parent)
                self.ome_viewer = ome_viewer

            def closeEvent(self, event):
                self.ome_viewer.inner_splitter_center.insertWidget(0,
                                                                   self.ome_viewer.gui_controls.embedding_view_properties_group_box)
                self.ome_viewer.inner_splitter_center.insertWidget(1, self.ome_viewer.interactive_plots_manager)
                self.ome_viewer.gui_controls.detach_embeddings_check_box.setChecked(False)
                event.accept()

                sizes = self.ome_viewer.outer_splitter.sizes()
                w = sum(sizes) / 3.0
                for i in range(len(sizes)):
                    sizes[i] = w
                self.ome_viewer.outer_splitter.setSizes(sizes)

        if state:
            self.detached_embeddings_window = DetachedEmbeddingsWindow(self)
            l = QtGui.QVBoxLayout()
            self.detached_embeddings_window.setLayout(l)
            l.addWidget(self.gui_controls.embedding_view_properties_group_box)
            l.addWidget(self.interactive_plots_manager)

            sizes = self.outer_splitter.sizes()
            w = sum(sizes) / 2.0
            sizes[0] = w
            sizes[1] = 0.0
            sizes[2] = w
            self.outer_splitter.setSizes(sizes)

            self.detached_embeddings_window.show()
        else:
            self.detached_embeddings_window.close()


# start qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()
        viewer = VaeViewer()
        viewer.show()
        sys.exit(app.exec_())
