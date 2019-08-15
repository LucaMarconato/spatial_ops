from typing import List

import matplotlib.cm
import numpy as np
from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *
from pyqtgraph.Qt import QtGui, QtCore

from sandbox.gui_controls import GuiControls
from sandbox.umap_eda import PlateUMAPLoader
from spatial_ops.data import JacksonFischerDataset as jfd, Patient, PatientSource


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
        # self.plot_widget = LayerPlotWidget()
        if self.viewer.gui_style != 'splitter':
            raise Exception('can only insert the plot widget if the gui of layer viewer is using splitters')
        self.inner_container = QtGui.QWidget()
        self.vhbox = QtGui.QVBoxLayout()
        self.viewer.splitter.replaceWidget(1, self.inner_container)
        self.inner_container.setLayout(self.vhbox)
        self.inner_splitter = QtGui.QSplitter()
        self.inner_splitter.setOrientation(QtCore.Qt.Vertical)
        self.vhbox.addWidget(self.inner_splitter)
        self.inner_splitter.addWidget(self.viewer.m_layer_ctrl_widget)
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.inner_splitter.addWidget(self.graphics_layout_widget)
        self.plot_widget = self.graphics_layout_widget.addPlot()

        # self.inner_splitter.addWidget(self.plot_widget)

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

        # self.cid = self.plot_widget.mpl_canvas.mpl_connect('motion_notify_event', self)

    def __call__(self, event):
        if event.xdata is None and event.ydata is not None or event.xdata is not None and event.ydata is None:
            raise Exception(f'event.xdata = {event.xdata}, event.ydata = {event.ydata}')

        if event.xdata is None:
            self.mouse_circle_path.set_alpha(0.0)
        else:
            self.mouse_circle_path.set_center((event.xdata, event.ydata))
            self.mouse_circle_path.set_alpha(0.2)

        # self.plot_widget.mpl_canvas.draw()
        indices = self.get_indices_of_selected_points()
        self.highlight_selected_cells(indices)
        # print(event)

    def highlight_selected_cells(self, indices):
        # indices_selected = indices
        indices_not_selected = list(set(range(len(self.current_points))).difference(indices))
        lut = self.masks_layer.lut
        if len(lut) != len(self.current_points):
            lut_size = len(self.current_points)
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
        pass

    def get_indices_of_selected_points(self) -> List[int]:
        # this does not work
        # contained = self.mouse_circle_path.contains_points(self.current_points)
        # to_return = [i for i, b in enumerate(contained) if b is True]
        # print(len(contained))
        # print(len(to_return))
        # print('')
        c = self.mouse_circle_path.center
        l = [i for i, (x, y) in enumerate(self.current_points) if (x - c[0]) ** 2 + (y - c[1]) ** 2 < 1]
        return l

        # def mouseMoved(evt):
        #     pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        #     print(pos)
        #     # if p1.sceneBoundingRect().contains(pos):
        #     #     mousePoint = vb.mapSceneToView(pos)
        #     #     index = int(mousePoint.x())
        #     #     if index > 0 and index < len(data1):
        #     #         label.setText(
        #     #             "<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (
        #     #             mousePoint.x(), data1[index], data2[index]))
        #     #     vLine.setPos(mousePoint.x())
        #     #     hLine.setPos(mousePoint.y())
        #
        # proxy = pg.SignalProxy(self.viewer.m_layer_view_widget.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
        # pass

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
                                                        pen=pg.mkPen(None))  # brush=pg.mkBrush(255, 255, 255, 120)
            self.scatter_plot_item.clear()
            # spots = [{'pos': umap_results[i, :], 'data': original_data[i, current_channel], 'brush': brushes[i]} for i in range(original_data.shape[0])]
            # self.scatter_plot_item.setData(spots)
            self.scatter_plot_item.setData(pos=umap_results, brush=brushes)
            self.plot_widget.clear()
            self.plot_widget.addItem(self.scatter_plot_item)
        else:
            import time
            start = time.time()
            # self.plot_widget.disableAutoRange()
            for i in range(100):
                self.scatter_plot_item.data[i][5] = QtGui.QBrush(QtGui.QColor(255, 0, 0))
                self.scatter_plot_item.data
            self.scatter_plot_item.invalidate()
            self.scatter_plot_item.update()
            # self.scatter_plot_item.setBrush(brushes)
            # self.plot_widget.autoRange()
            # [QtGui.QBrush(QtGui.QColor(255, 0, 0))] * umap_results.shape[0]
            # self.scatter_plot_item.setData(pos=umap_results, brush=brushes)
            # self.scatter_plot_item.setBrush()
            # self.scatter_plot_item.update()
            print(f': {time.time() - start}')

        # self.plot_widget.mpl_canvas.clear_canvas()
        # axes = self.plot_widget.axes()
        # self.current_points = [[umap_results[i, 0], umap_results[i, 1]] for i in range(umap_results.shape[0])]
        # self.scatter_plot = axes.scatter(umap_results[:, 0], umap_results[:, 1], c=original_data[:, current_channel])
        # axes.set_aspect('equal')
        # self.mouse_circle_path = matplotlib.patches.Circle((0.0, 0.0), 1, alpha=0.2, fc='yellow')
        # # axes.add_patch(self.mouse_circle_path)
        # self.plot_widget.mpl_canvas.draw()


# start qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = pg.mkQApp()
        viewer = OmeViewer()
        viewer.setWindowTitle('TODO: avoid the creation of this window')
        viewer.show()
        # QtGui.QApplication.instance().exec_()
        # app = QtGui.QApplication(sys.argv)
        # OmeViewer()
        sys.exit(app.exec_())
