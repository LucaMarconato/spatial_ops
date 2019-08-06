import pickle
import h5py
import os
import numpy as np
from abc import ABC, abstractmethod

from spatial_ops.folders import get_pickle_lazy_loader_data_path, hdf5_lazy_loader_data_path
from spatial_ops.unpickler import CustomUnpickler


class LazyLoaderAssociatedInstance:
    # when implementing this method in a derived class add an unique suffix (depending on the derived class and not
    # on a specific instance of the derived class) so to ensure uniqueness among different classes which derive this
    # class example, if we have a Dog with name 'Bobby' and a Cat with name 'Bobby', good unique identifiers are
    # 'Cat_Bobby' and 'Dog_Bobby', not just 'Bobby'
    def get_lazy_loader_unique_identifier(self) -> str:
        raise NotImplementedError


class LazyLoader(ABC):
    # while it is not compulsory, it is convenient to implement __init__ in the child class to specify the
    # associated_instance type, see the examples below for clarification
    def __init__(self, associated_instance: LazyLoaderAssociatedInstance, resource_unique_identifier: str):
        self.associated_instance = associated_instance
        self.resource_unique_identifier = resource_unique_identifier

    @abstractmethod
    def precompute(self):
        pass

    @abstractmethod
    def delete_precomputation(self):
        pass

    @abstractmethod
    def has_data_already_been_precomputed(self):
        pass

    @abstractmethod
    def _load_precomputed_data(self):
        pass

    def load_data(self):
        if self.associated_instance is None:
            raise ValueError(f'self.associated_instance = {self.associated_instance}')

        if not self.has_data_already_been_precomputed():
            # print('precomputing')
            return self.precompute()
        else:
            # print('loading')
            return self._load_precomputed_data()

    @abstractmethod
    def _get_resource_id(self) -> str:
        pass


class PickleLazyLoader(LazyLoader, ABC):
    def _get_resource_id(self) -> str:
        if self.resource_unique_identifier is None:
            raise ValueError(f'self.resource_unique_identifier = {self.resource_unique_identifier}')
        resource_id = self.associated_instance.get_lazy_loader_unique_identifier().replace('-', '--') \
                      + '-' + self.resource_unique_identifier.replace('-', '--')
        return resource_id

    def get_pickle_path(self):
        return os.path.join(get_pickle_lazy_loader_data_path(), self._get_resource_id() + '.pickle')

    def _load_precomputed_data(self):
        pickle_path = self.get_pickle_path()
        data = CustomUnpickler(open(pickle_path, 'rb')).load()
        return data

    def has_data_already_been_precomputed(self):
        pickle_path = self.get_pickle_path()
        return os.path.isfile(pickle_path)

    def delete_precomputation(self):
        os.remove(self.get_pickle_path())


if not os.path.isfile(hdf5_lazy_loader_data_path):
    f = h5py.File(hdf5_lazy_loader_data_path, 'w')
    f.close()


class HDF5LazyLoader(LazyLoader, ABC):
    def _get_resource_id(self) -> str:
        if self.resource_unique_identifier is None:
            raise ValueError(f'self.resource_unique_identifier = {self.resource_unique_identifier}')
        resource_id = self.associated_instance.get_lazy_loader_unique_identifier() \
                      + '/' + self.resource_unique_identifier
        return resource_id

    # just for convenience, to have the path ready when deriving the subclass
    def get_hdf5_file_path(self) -> str:
        return hdf5_lazy_loader_data_path

    def get_hdf5_resource_internal_path(self):
        return self._get_resource_id()

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


if __name__ == '__main__':
    from spatial_ops.data import JacksonFischerDataset as jfd
    from spatial_ops.data import Patient

    patient = jfd.patients[0]


    class NumberOfPlatesLoader0(PickleLazyLoader):
        def __init__(self, associated_instance: Patient, resource_unique_identifier: str):
            super().__init__(associated_instance, resource_unique_identifier)

        def precompute(self):
            # just to enable the autocompletion within the ide
            p: Patient = self.associated_instance
            data = f'len = {len(p.plates)}'
            pickle.dump(data, open(self.get_pickle_path(), 'wb'))
            return data


    derived_quantity = NumberOfPlatesLoader0(patient, 'example_quantity')
    print(derived_quantity.load_data())


    class NumberOfPlatesLoader1(HDF5LazyLoader):
        def __init__(self, associated_instance: Patient, resource_unique_identifier: str):
            super().__init__(associated_instance, resource_unique_identifier)

        def precompute(self):
            p: Patient = self.associated_instance
            data = f'len = {len(p.plates)}'
            # data = np.zeros((2, 3))
            with h5py.File(self.get_hdf5_file_path(), 'r+') as f:
                f[self.get_hdf5_resource_internal_path()] = data
            return data


    derived_quantity = NumberOfPlatesLoader1(patient, 'example_quantity')
    # derived_quantity.delete_precomputation()
    print(derived_quantity.load_data())
