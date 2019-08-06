import os
import pickle
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
import progressbar
import skimage
import vigra
import h5py

from spatial_ops.folders import \
    basel_patient_data_path, \
    zurich_patient_data_path, \
    single_cell_data_path, \
    staining_data_path, \
    whole_image_data_path, \
 \
    get_processed_data_folder, \
    get_ome_files, \
    get_masks_files, \
    get_ome_folder, \
 \
    get_mask_path_associated_to_ome_path, \
    get_region_features_path_associated_to_ome_path, \
    get_pickles_folder
from spatial_ops.unpickler import CustomUnpickler
from spatial_ops.lazy_loader import LazyLoaderAssociatedInstance

database_single_cell = os.path.join(get_processed_data_folder(),
                                    os.path.basename(single_cell_data_path.replace('.csv', '.db')))

basel_patient_data = pd.read_csv(basel_patient_data_path)
zurich_patient_data = pd.read_csv(zurich_patient_data_path)
staining_data = pd.read_csv(staining_data_path)
whole_image_data = pd.read_csv(whole_image_data_path)

remaining_ome_files = set(get_ome_files())
remaining_mask_files = set(get_masks_files())

PatientSource = Enum('PatientSource', 'basel zurich')


class Patient(LazyLoaderAssociatedInstance):
    def __init__(self, source: PatientSource, pid: int):
        super().__init__()
        self.plates = []
        self.source = source
        self.pid = pid
        if self.source == PatientSource.basel:
            self.df = basel_patient_data
        else:
            self.df = zurich_patient_data
        self.initialize_plates()

    def initialize_plates(self):
        plate_rows = self.df[self.df.PID == self.pid]
        for plate_row in plate_rows.itertuples():
            filename = plate_row.FileName_FullStack
            plate = Plate(filename)
            self.plates.append(plate)

    def get_lazy_loader_unique_identifier(self) -> str:
        return f'patient_{self.source}_{self.pid}'


class RegionFeatures:
    def __init__(self, feature_accumulator: vigra.analysis.FeatureAccumulator):
        self.count = feature_accumulator['Count']
        self.max = feature_accumulator['Maximum']
        self.mean = feature_accumulator['Mean']
        self.sum = feature_accumulator['Sum']
        self.variance = feature_accumulator['Variance']
        self.center = feature_accumulator['RegionCenter']


class Plate(LazyLoaderAssociatedInstance):
    def __init__(self, ome_filename: str):
        self.ome_path: str
        self.mask_path: str
        self.region_features_path: str

        self.ome_path = os.path.join(get_ome_folder(), ome_filename)
        if self.ome_path in remaining_ome_files:
            remaining_ome_files.remove(self.ome_path)
        else:
            raise FileNotFoundError(f'file not found: {self.ome_path}')

        self.mask_path = get_mask_path_associated_to_ome_path(self.ome_path)
        if self.mask_path in remaining_mask_files:
            remaining_mask_files.remove(self.mask_path)
        else:
            raise FileNotFoundError(f'file not found {self.mask_path}')

        self.region_features_path = get_region_features_path_associated_to_ome_path(self.ome_path)
        if not os.path.isfile(self.region_features_path):
            self.generate_region_features(self.region_features_path)

    def get_ome(self) -> np.ndarray:
        ome = skimage.io.imread(self.ome_path)
        ome = np.moveaxis(ome, 0, 2)
        ome = np.require(ome, requirements=['C'])
        return ome

    def get_masks(self) -> np.ndarray:
        masks = skimage.io.imread(self.mask_path)
        masks = masks.astype('uint32')
        masks = np.require(masks, requirements=['C'])
        return masks

    def get_region_features(self) -> Dict[str, np.array]:
        region_features = CustomUnpickler(open(self.region_features_path, 'rb')).load()
        return region_features

    def generate_region_features(self, region_features_path: str):
        ome = self.get_ome()
        masks = self.get_masks()

        # plt.figure()
        # cmap = matplotlib.colors.ListedColormap(np.random.rand(masks.max() + 1, 3))
        # cmap.colors[0] = (0, 0, 0)
        # im = plt.imshow(masks, cmap=cmap)
        # # plt.colorbar(im)
        # plt.show()

        # supported_features = vigra.analysis.extractRegionFeatures(ome, labels=masks, features=None,
        #                                                           ignoreLabel=0).supportedFeatures()
        # print(f'supported features: {supported_features}')
        feature_accumulator = vigra.analysis.extractRegionFeatures(ome, labels=masks, ignoreLabel=0,
                                                                   features=['Count', 'Maximum', 'Mean', 'Sum',
                                                                             'Variance', 'RegionCenter'])

        region_features = RegionFeatures(feature_accumulator)
        pickle.dump(region_features, open(region_features_path, 'wb'))

    @staticmethod
    def get_mask_for_specific_cell(masks: np.ndarray, region_number: int):
        return (masks == region_number).astype(int)

    def lazy_loader_unique_identifier(self) -> str:
        return f'plate_{os.path.basename(self.ome_path)}'


def call_the_initializer(cls):
    cls.initialize()
    return cls


@call_the_initializer
class JacksonFischerDataset:
    @classmethod
    def initialize(cls):
        print('initializing JacksonFischerDataset')
        dont_load_from_pickles = False
        pickle_path = os.path.join(get_pickles_folder(), 'JacksonFisherDataset.pickle')
        if os.path.isfile(pickle_path) and not dont_load_from_pickles:
            print('unpickling data... ', end='')
            cls.patients = CustomUnpickler(open(pickle_path, 'rb')).load()
            print('DONE')
        else:
            basel_patient_ids = set(basel_patient_data.PID)
            zurich_patient_ids = set(zurich_patient_data.PID)
            cls.patients = []
            with progressbar.ProgressBar(max_value=len(basel_patient_ids) + len(zurich_patient_ids)) as bar:
                i = 0
                bar.update(0)
                for pid in basel_patient_ids:
                    patient = Patient(PatientSource.basel, pid)
                    cls.patients.append(patient)
                    i += 1
                    bar.update(i)
                for pid in zurich_patient_ids:
                    patient = Patient(PatientSource.zurich, pid)
                    cls.patients.append(patient)
                    i += 1
                    bar.update(i)
            pickle.dump(cls.patients, open(pickle_path, 'wb'))
