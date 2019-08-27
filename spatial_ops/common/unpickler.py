import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Patient':
            from spatial_ops.common.data import Patient
            return Patient
        if name == 'Plate':
            from spatial_ops.common.data import Plate
            return Plate
        if name == 'PatientSource':
            from spatial_ops.common.data import PatientSource
            return PatientSource
        if name == 'RegionFeatures':
            from spatial_ops.common.data import RegionFeatures
            return RegionFeatures
        return super().find_class(module, name)
