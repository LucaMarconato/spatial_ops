import os
import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np

from .folders import get_pickle_lazy_loader_data_path, hdf5_lazy_loader_data_path
from .unpickler import CustomUnpickler


class LazyLoaderAssociatedInstance:
    # when implementing this method in a derived class add an unique suffix (depending on the derived class and not
    # on a specific instance of the derived class) so to ensure uniqueness among different classes which derive this
    # class example, if we have a Dog with name 'Bobby' and a Cat with name 'Bobby', good unique identifiers are
    # 'Cat_Bobby' and 'Dog_Bobby', not just 'Bobby'
    def get_lazy_loader_unique_identifier(self) -> str:
        raise NotImplementedError


class LazyLoader(ABC):
    def __init__(self, associated_instance: LazyLoaderAssociatedInstance):
        self.associated_instance = associated_instance

    @abstractmethod
    def get_resource_unique_identifier(self) -> str:
        pass

    @abstractmethod
    def precompute(self):
        pass

    @abstractmethod
    def delete_precomputation(self):
        pass

    @abstractmethod
    def has_data_already_been_precomputed(self):
        pass

    def precompute_if_needed(self):
        if not self.has_data_already_been_precomputed():
            self.precompute()

    @abstractmethod
    def _load_precomputed_data(self):
        pass

    def load_data(self, store_precomputation_on_disk=True):
        if self.associated_instance is None:
            raise ValueError(f'self.associated_instance = {self.associated_instance}')

        if not self.has_data_already_been_precomputed():
            # print('precomputing')
            data = self.precompute()
            if data is None:
                raise ValueError(f'data = {data}')
            if store_precomputation_on_disk:
                self._save_data(data)
            return data
        else:
            # print('loading')
            return self._load_precomputed_data()

    @abstractmethod
    def _save_data(self, data):
        pass


class PickleLazyLoader(LazyLoader, ABC):
    def get_pickle_path(self):
        path = os.path.join(get_pickle_lazy_loader_data_path(),
                            self.associated_instance.get_lazy_loader_unique_identifier())
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, self.get_resource_unique_identifier() + '.pickle')
        return path

    def _load_precomputed_data(self):
        pickle_path = self.get_pickle_path()
        try:
            data = CustomUnpickler(open(pickle_path, 'rb')).load()
        # if the pickle is corrupted because a previous execution of the program was terminated while pickling a file
        # then we want to delete the pickle file and recompute it
        except EOFError:
            self.delete_precomputation()
            data = self.load_data()
        return data

    def has_data_already_been_precomputed(self):
        pickle_path = self.get_pickle_path()
        return os.path.isfile(pickle_path)

    def delete_precomputation(self):
        os.remove(self.get_pickle_path())

    def _save_data(self, data):
        pickle.dump(data, open(self.get_pickle_path(), 'wb'))


if not os.path.isfile(hdf5_lazy_loader_data_path):
    f = h5py.File(hdf5_lazy_loader_data_path, 'w')
    f.close()


class HDF5LazyLoader(LazyLoader, ABC):
    # just for convenience, to have the path ready when deriving the subclass
    def get_hdf5_file_path(self) -> str:
        return hdf5_lazy_loader_data_path

    def get_hdf5_resource_internal_path(self):
        return self.associated_instance.get_lazy_loader_unique_identifier() \
               + '/' + self.get_resource_unique_identifier()

    # using a singleton and opening the file only once in r+ mode would lead to better performance, but I will wait
    # to see if the current performance are bad before implementing it
    def _load_precomputed_data(self):
        with h5py.File(self.get_hdf5_file_path(), 'r') as f:
            data = np.array(f[self.get_hdf5_resource_internal_path()][...])
            return data

    def has_data_already_been_precomputed(self):
        with h5py.File(self.get_hdf5_file_path(), 'r') as f:
            return self.get_hdf5_resource_internal_path() in f

    def delete_precomputation(self):
        with h5py.File(self.get_hdf5_file_path(), 'r+') as f:
            del f[self.get_hdf5_resource_internal_path()]

    def _save_data(self, data):
        with h5py.File(self.get_hdf5_file_path(), 'r+') as f:
            f[self.get_hdf5_resource_internal_path()] = data


if __name__ == '__main__':
    from spatial_ops.data import JacksonFischerDataset as jfd
    from spatial_ops.data import Patient

    patient = jfd.patients[15]


    class NumberOfPlatesLoader0(PickleLazyLoader):
        def get_resource_unique_identifier(self) -> str:
            return 'example_quantity_pickle'

        def precompute(self):
            # just to enable the autocompletion within the ide
            p: Patient = self.associated_instance
            data = f'len = {len(p.plates)}'
            return data


    derived_quantity = NumberOfPlatesLoader0(patient)
    print(derived_quantity.load_data())


    class NumberOfPlatesLoader1(HDF5LazyLoader):
        def get_resource_unique_identifier(self) -> str:
            return 'example_quantity_hdf5'

        def precompute(self):
            p: Patient = self.associated_instance
            data = f'len = {len(p.plates)}'
            # data = np.zeros((2, 3))
            return data


    derived_quantity = NumberOfPlatesLoader1(patient)
    # derived_quantity.delete_precomputation()
    print(derived_quantity.load_data())
