import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.data
import skimage.io
import sklearn
import sklearn.decomposition
import vigra
from layer_viewer import LayerViewerWidget
from layer_viewer.layers import *

from spatial_ops.data import JacksonFischerDataset as jfd
from sandbox.umap_eda import PlateUMAPLoader
from sandbox.gui_controls import GuiControls

app = pg.mkQApp()


def show_mask_histogram(mask: np.ndarray):
    mask = skimage.io.imread(mask)
    plt.figure()
    plt.hist(mask.ravel(), bins=50)
    plt.show()
    print(mask.min(), mask.max())


def inspect_plate(plate):
    img = plate.get_ome()
    mask = plate.get_masks()

    flat_mask = mask.ravel()
    where_non_zero = numpy.where(flat_mask != 0)[0]
    print(where_non_zero)

    print(mask.shape)
    img = img.squeeze()
    shape = img.shape[1:3]
    n_channels = img.shape[0]
    print(f"shape {img.shape}")

    # img = numpy.moveaxis(img, 0, 2)
    img = vigra.taggedView(img, 'xyc')
    img = vigra.filters.gaussianSmoothing(img, 0.5)
    img = numpy.require(img, requirements=['C'])
    X = img.reshape([-1, n_channels])
    maskedX = X[flat_mask, :]
    n_components = 3
    # dim_red_alg = sklearn.decomposition.PCA(n_components=n_components)
    # dim_red_alg.fit(numpy.sqrt(X))
    # Y = dim_red_alg.transform(X)
    # reshape = tuple(shape) + (n_components,)
    # Y = Y.reshape(reshape)

    print(f"Y {img.shape}")

    # for c in range(3):
    #     Yc = Y[..., c]
    #     Yc -= Yc.min()
    #     Yc /= Yc.max()

    viewer = LayerViewerWidget()
    viewer.setWindowTitle(f'{plate.ome_path}')
    viewer.show()
    layer = MultiChannelImageLayer(name='img', data=img[...])
    viewer.addLayer(layer=layer)
    layer.ctrl_widget().channelSelector.setValue(47)
    gui_controls = GuiControls()
    viewer.inner_splitter.insertWidget(1, gui_controls)
    gui_controls.setPatient(plate.patient)

    reducer, result = PlateUMAPLoader(plate).load_data()
    color_channel = 5
    rf = plate.get_region_features()
    axes = viewer.axes()
    axes.scatter(result[:, 0], result[:, 1], c=rf.sum[:, color_channel])
    # viewer.plot_canvas().colorbar()
    viewer.draw_plot_canvas()

    # axes.show()

    # layer = MultiChannelImageLayer(name='PCA-IMG', data=Y[...])
    # viewer.addLayer(layer=layer)
    # layer.ctrl_widget().toggle_eye.setState(False)

    layer = ObjectLayer(name='mask', data=mask)
    viewer.addLayer(layer=layer)
    layer.ctrl_widget().bar.set_fraction(0.2)


plate = jfd.patients[1].plates[0]
inspect_plate(plate)

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
        QtGui.QApplication.instance().exec_()
