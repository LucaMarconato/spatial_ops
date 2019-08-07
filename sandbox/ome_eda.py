from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot

from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *

from spatial_ops.data import JacksonFischerDataset as jfd
from sandbox.umap_eda import PlateUMAPLoader
from sandbox.gui_controls import GuiControls
import vigra

app = pg.mkQApp()

from spatial_ops.data import JacksonFischerDataset as jfd, Patient


class OmeViewer(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.viewer = LayerViewerWidget()
        self.viewer.setWindowTitle('ome viewer')
        self.ome_layer = None
        self.masks_layer = None
        # self.viewer.show()

        self.gui_controls = GuiControls()
        self.viewer.inner_splitter.insertWidget(1, self.gui_controls)

        patients_count = len(jfd.patients)
        self.gui_controls.patient_slider.setMaximum(patients_count - 1)
        channels_count = len(jfd.get_channels_annotation())
        self.gui_controls.channel_slider.setMaximum(channels_count - 1)

        def patient_value_changed(new_patient_index):
            self.set_patient_by_index(new_patient_index)

        def channel_value_changed(new_channel):
            self.ome_layer.ctrl_widget().channelSelector.setValue(new_channel)
            self.update_channel_label()

        self.gui_controls.patient_slider.valueChanged.connect(patient_value_changed)
        self.gui_controls.channel_slider.valueChanged.connect(channel_value_changed)

        self.gui_controls.patient_slider.blockSignals(True)
        self.set_patient_by_index(1)
        self.gui_controls.patient_slider.blockSignals(False)

        def sync_back(new_channel):
            print('syncing back')
            self.gui_controls.channel_slider.blockSignals(True)
            self.gui_controls.channel_slider.setValue(new_channel)
            self.update_channel_label()
            self.gui_controls.channel_slider.blockSignals(False)

        self.ome_layer.ctrl_widget().channelSelector.valueChanged.connect(sync_back)
        pass

    def set_patient_by_index(self, patient_index: int):
        print('b')
        self.current_patient = jfd.patients[patient_index]
        patient_information = f'source: {self.current_patient.source}, pid: {self.current_patient.pid}'
        self.gui_controls.patient_information_label.setText(patient_information)
        self.current_plate = self.current_patient.plates[0]
        ome = self.current_plate.get_ome()
        ome = vigra.taggedView(ome, 'xyc')
        ome = vigra.filters.gaussianSmoothing(ome, 0.5)

        if self.ome_layer is None:
            self.ome_layer = MultiChannelImageLayer(name='ome', data=ome[...])
            self.viewer.addLayer(layer=self.ome_layer)
            self.ome_layer.ctrl_widget().channelSelector.setValue(47)
        else:
            self.ome_layer.update_data(ome)


        print('d')
        if not hasattr(self, 'current_channel'):
            self.current_channel = 0
        channel_information = f'{jfd.get_channels_annotation()[self.current_channel]}'
        self.gui_controls.channel_information_label.setText(channel_information)

        masks = self.current_plate.get_masks()

        if self.masks_layer is None:
            self.masks_layer = ObjectLayer(name='mask', data=masks)
            self.viewer.add_layer(layer=self.masks_layer)
            self.masks_layer.ctrl_widget().bar.set_fraction(0.2)
        else:
            self.masks_layer.update_data(masks)

    def set_patient(self, patient: Patient):
        index_in_list = jfd.patients.index(patient)
        self.set_patient_by_index(index_in_list)

    def update_channel_label(self):
        label = jfd.get_channels_annotation()[self.gui_controls.channel_slider.value()]
        self.gui_controls.channel_information_label.setText(label)


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
