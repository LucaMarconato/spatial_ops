from PyQt5 import QtGui
from PyQt5.QtCore import QSettings
from PyQt5.QtCore import Qt
from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *

from sandbox.gui_controls import GuiControls

app = pg.mkQApp()

from spatial_ops.data import JacksonFischerDataset as jfd, Patient, PatientSource
from sandbox.umap_eda import PlateUMAPLoader

from spatial_ops.layer_plot_widget import LayerPlotWidget


class OmeViewer(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.viewer = LayerViewerWidget()
        self.viewer.setWindowTitle('ome viewer')
        self.ome_layer = None
        self.masks_layer = None
        # self.viewer.show()

        # on the left the layout we have, on the right the one we want to obtain
        # |--------------|      |--------------|
        # |              |      |        ctrl0 |
        # | image  ctrl0 |      | image  ctrl1 |
        # |              |      |        plot  |
        # |--------------|      |--------------|
        self.plot_widget = LayerPlotWidget()
        if self.viewer.gui_style != 'splitter':
            raise Exception('can only insert the plot widget if the gui of layer viewer is using splitters')
        self.inner_container = QWidget()
        self.vhbox = QtGui.QVBoxLayout()
        self.viewer.splitter.replaceWidget(1, self.inner_container)
        self.inner_container.setLayout(self.vhbox)
        self.inner_splitter = QtGui.QSplitter()
        self.inner_splitter.setOrientation(Qt.Vertical)
        self.vhbox.addWidget(self.inner_splitter)
        self.inner_splitter.addWidget(self.viewer.m_layer_ctrl_widget)
        self.inner_splitter.addWidget(self.plot_widget)

        self.gui_controls = GuiControls()
        self.inner_splitter.insertWidget(1, self.gui_controls)
        self.inner_splitter.show()

        sizes = self.inner_splitter.sizes()
        a = sizes[0] + sizes[2]
        ratio = 9.0 / 16.0
        sizes[0] = a * ratio
        sizes[2] = a * (1 - ratio)
        self.inner_splitter.setSizes(sizes)

        patients_count = len(jfd.patients)
        self.gui_controls.patient_slider.setMaximum(patients_count - 1)
        channels_annotation = jfd.get_channels_annotation()
        channels_count = len(channels_annotation)
        self.gui_controls.channel_slider.setMaximum(channels_count - 1)
        for k, v in channels_annotation.items():
            self.gui_controls.channel_name_combo_box.addItem(v)
        # size_hint = self.gui_controls.channel_name_combo_box.sizeHint().width()
        # self.gui_controls.channel_name_combo_box.view().setMinimumWidth(size_hint)

        self.gui_controls.patient_slider.valueChanged.connect(lambda x: self.on_patient_slider_value_changed(x))
        self.gui_controls.channel_slider.valueChanged.connect(lambda x: self.on_channel_slider_value_changed(x))
        self.gui_controls.patient_source_combo_box.currentIndexChanged.connect(
            lambda x: self.on_patient_source_combo_box_current_index_changed(x))
        self.gui_controls.patient_pid_spin_box.valueChanged.connect(
            lambda x: self.on_patient_pid_spin_box_value_changed(x))
        self.gui_controls.channel_name_combo_box.currentIndexChanged.connect(
            lambda x: self.on_channel_name_combo_box_current_index_changed(x)
        )

        self.set_patient(jfd.patients[20])

        def sync_back(new_channel):
            self.gui_controls.channel_slider.setValue(new_channel)

        self.ome_layer.ctrl_widget().channel_selector.valueChanged.connect(sync_back)

        self.ome_layer.ctrl_widget().channel_selector.setValue(47)
        self.load_settings()
        pass

    def load_settings(self):
        settings = QSettings('B260', 'spatial_ops')
        patient_source = settings.value('patient_source', PatientSource(0).value, int)
        patient_pid = settings.value('patient_pid', 1, int)
        self.gui_controls.patient_source_combo_box.setCurrentIndex(patient_source)
        self.gui_controls.patient_pid_spin_box.setValue(patient_pid)

    def save_settings(self):
        settings = QSettings('B260', 'spatial_ops')
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

        # self.gui_controls.patient_information_label.setText(patient_information)
        self.current_plate = self.current_patient.plates[0]
        import time
        start = time.time()
        ome = self.current_plate.get_ome()
        self.update_umap()
        print(f'get_ome: {time.time() - start}')

        if self.ome_layer is None:
            start = time.time()
            self.ome_layer = MultiChannelImageLayer(name='ome', data=ome[...])
            self.viewer.addLayer(layer=self.ome_layer)
            print(f'creating ome layer: {time.time() - start}')
        else:
            start = time.time()
            self.ome_layer.update_data(ome)
            print(f'updating ome layer: {time.time() - start}')

        start = time.time()
        masks = self.current_plate.get_masks()
        print(f'get_masks: {time.time() - start}')

        if self.masks_layer is None:
            start = time.time()
            self.masks_layer = ObjectLayer(name='mask', data=masks)
            self.viewer.add_layer(layer=self.masks_layer)
            self.masks_layer.ctrl_widget().bar.set_fraction(0.2)
            print(f'creating masks layer: {time.time() - start}')
        else:
            start = time.time()
            self.masks_layer.update_data(masks)
            print(f'updating masks layer: {time.time() - start}')
        print(f'')

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
        reducer, umap_results = PlateUMAPLoader(self.current_plate).load_data()
        rf = self.current_plate.get_region_features()
        current_channel = self.gui_controls.channel_slider.value()
        self.plot_widget.mpl_canvas.clear_canvas()
        axes = self.plot_widget.axes()
        axes.scatter(umap_results[:, 0], umap_results[:, 1], c=rf.sum[:, current_channel])
        self.plot_widget.mpl_canvas.draw()


# def inspect_plate(plate):
#
#
#     reducer, result = PlateUMAPLoader(plate).load_data()
#     color_channel = 5
#     rf = plate.get_region_features()
#     axes = viewer.axes()
#     axes.scatter(result[:, 0], result[:, 1], c=rf.sum[:, color_channel])
#     # viewer.plot_canvas().colorbar()
#     viewer.draw_plot_canvas()
#
#     # axes.show()
#
#     # layer = MultiChannelImageLayer(name='PCA-IMG', data=Y[...])
#     # viewer.addLayer(layer=layer)
#     # layer.ctrl_widget().toggle_eye.setState(False)
#

#
# plate = jfd.patients[1].plates[0]
# inspect_plate(plate)

# layer = RGBImageLayer(name='img', data=image[...])
# viewer.addLayer(layer=layer)

# labels = numpy.zeros(image.shape[0:2], dtype='uint8')
# label_layer = LabelLayer(name='labels', data=None)
# viewer.addLayer(layer=label_layer)
# viewer.setData('labels',image=labels)

# layer.setOpacity(0.5)

# # viewer.setLayerVisibility('img', False)
# viewer.setLayerOpacity('img', 0.4)
# viewer.setLayerOpacity('img', 0.4)

# start qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # QtGui.QApplication.instance().exec_()
        app = QtGui.QApplication(sys.argv)
        OmeViewer()
        sys.exit(app.exec_())
