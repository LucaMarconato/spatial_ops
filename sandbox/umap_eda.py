import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool

from spatial_ops.data import JacksonFischerDataset as jfd, Plate
from spatial_ops.lazy_loader import PickleLazyLoader


class PlateUMAPLoader(PickleLazyLoader):
    def precompute(self):
        rf = self.associated_instance.get_region_features()
        reducer = umap.UMAP(verbose=True, n_components=2)
        umap_result = reducer.fit_transform(rf.sum)
        data = (reducer, umap_result)
        return data

def precompute_umap_on_single_patient(patient):
    for plate in patient.plates:
        plate_umap_loader = PlateUMAPLoader(plate, 'plate_umap_of_sum')
        plate_umap_loader.load_data()


def parallel_precompute_umap():
    with Pool(processes=4) as pool:
        pool.map(precompute_umap_on_single_patient,
                 iterable=jfd.patients[0:105])


if __name__ == '__main__':
    parallel_precompute_umap()
# iterable=jfd.patients[0:4])
# patient_index = 0
# plate_index = 0
#
# plate = jfd.patients[patient_index].plates[plate_index]
# plate_umap_loader = PlateUMAPLoader(plate, 'plate_umap_of_sum')
# reducer, umap_result = plate_umap_loader.load_data()

# rf = plate.get_region_features()
#
# color_channel = 5
# if umap_result.shape[1] == 3:
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     im = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=rf.sum[:, color_channel])
#     fig.colorbar(im)
# elif umap_result.shape[1] == 2:
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.scatter(umap_result[:, 0], umap_result[:, 1], c=rf.sum[:, color_channel], s=1, cmap=plt.cm.viridis)
#     plt.colorbar()
#     plt.subplot(1, 2, 2)
#     # sns.violinplot(rf.sum[:, color_channel])
#     plt.violinplot(rf.sum[:, color_channel])
# else:
#     raise ValueError(f'n_components = {umap_result.shape[1]}')
# plt.show()
