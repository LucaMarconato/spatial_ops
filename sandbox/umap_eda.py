from multiprocessing import Pool

import umap
from sklearn.manifold import TSNE

from spatial_ops.data import JacksonFischerDataset as jfd
from spatial_ops.lazy_loader import PickleLazyLoader


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


if __name__ == '__main__':
    parallel_precompute_umap()
