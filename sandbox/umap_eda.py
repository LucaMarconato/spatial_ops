import os
import numpy as np
import pickle
from multiprocessing import Pool
import pyqtgraph as pg
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.manifold import TSNE

from spatial_ops.data import JacksonFischerDataset as jfd
from spatial_ops.lazy_loader import PickleLazyLoader
from spatial_ops.folders import mem


class PlateUMAPLoader(PickleLazyLoader):
    def get_resource_unique_identifier(self) -> str:
        return 'plate_umap_of_mean'

    def precompute(self):
        rf = self.associated_instance.get_region_features()
        reducer = umap.UMAP(verbose=True, n_components=2)
        original_data = rf.mean
        umap_result = reducer.fit_transform(original_data)
        data = (reducer, umap_result, original_data)
        return data


class PlateTSNELoader(PickleLazyLoader):
    def get_resource_unique_identifier(self) -> str:
        return 'plate_tsne_of_mean'

    def precompute(self):
        rf = self.associated_instance.get_region_features()
        reducer = TSNE(verbose=True, n_components=2)
        original_data = rf.mean
        tsne_result = reducer.fit_transform(original_data)
        data = (reducer, tsne_result, original_data)
        return data


def precompute_umap_on_single_patient(patient):
    for plate in patient.plates:
        PlateUMAPLoader(plate).load_data()


def precompute_tsne_on_single_patient(patient):
    for plate in patient.plates:
        PlateTSNELoader(plate).load_data()


def parallel_precompute_umap():
    with Pool(processes=4) as pool:
        pool.map(precompute_umap_on_single_patient,
                 iterable=jfd.patients)
    with Pool(processes=4) as pool:
        pool.map(precompute_tsne_on_single_patient,
                 iterable=jfd.patients)


def umap_of_list_of_patients(patients):
    temp_file = os.path.expanduser('~/temp/umap.pickle')
    channels_count = len(jfd.get_channels_annotation())
    data = np.empty([0, channels_count])
    plate_color = []
    i = 0
    for patient in patients:
        for plate in patient.plates:
            rf = plate.get_region_features()
            means = rf.mean
            plate_color.extend([i] * rf.mean.shape[0])
            i += 1
            data = np.concatenate([data, means], axis=0)
    print(data.shape)
    # reducer = umap.UMAP(verbose=True, n_components=2, n_neighbors=30, min_dist=0)
    # umap_results = reducer.fit_transform(data)
    # pickle.dump((reducer, umap_results), open(temp_file, 'wb'))
    reducer, umap_results = pickle.load(open(temp_file, 'rb'))
    distinct_colors = max(plate_color) + 1
    colors = np.random.randint(0, 256, distinct_colors * 3)
    colors.shape = (distinct_colors, 3)
    app = pg.mkQApp()
    # pg.setConfigOption('useOpenGL', True)
    brushes = [pg.mkBrush(colors[i]) for i in plate_color]
    pg.plot(x=umap_results[:, 0], y=umap_results[:, 1], symbolBrush=brushes, symbol='o', pen=None)
    app.exec_()
    # pg.plot([0, 1], [1, 2], pen=None, symbolBrush=[pg.mkBrush((100, 100, 0)), pg.mkBrush((255, 0, 0))], symbol='s');
    # app.exec()
    # plt.figure()
    # plt.scatter(umap_results[:, 0], umap_results[:, 1], c=plate_color)
    # plt.show()


def show_umap_embedding(data_points: np.ndarray, instance_ids, joblib_seed):
    @mem.cache
    def get_sample(data_points, joblib_seed):
        p = np.random.permutation(len(data_points))
        sample = data_points[p]
        sample_size = 10000
        if len(data_points) > sample_size:
            sample = data_points[:sample_size]
        assert len(sample) <= sample_size
        return sample, list(p)

    sample, p = get_sample(data_points, joblib_seed)
    instance_ids = [instance_ids[i] for i in p][:len(sample)]

    @mem.cache
    def get_umap(sample, joblib_seed):
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0)
        umap_result = reducer.fit_transform(sample)
        return reducer, umap_result

    _, umap_result = get_umap(sample, joblib_seed)

    random_colormap = np.random.rand(len(data_points))
    c = list(map(lambda x: random_colormap[x], instance_ids))
    plt.figure()
    plt.scatter(umap_result[:, 0], umap_result[:, 1], s=2, c=c, cmap=plt.get_cmap('hsv'))
    plt.show()
    pass
    df = pd.DataFrame(sample)
    sns.pairplot(df)
    # df['instance_id'] = instance_ids
    # sns.pairplot(df, hue='instance_id')



if __name__ == '__main__':
    # parallel_precompute_umap()
    patients = [patient for patient in jfd.patients[0:100]]
    umap_of_list_of_patients(patients)
