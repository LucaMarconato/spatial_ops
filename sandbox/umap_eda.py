import spatial_ops.data
import umap
import numpy as np
from spatial_ops.data import JacksonFischerDataset as jfd
from spatial_ops.folders import get_pickles_folder
import pickle
import os

patient_index = 0
plate_index = 0

plate = jfd.patients[patient_index].plates[plate_index]
ome = plate.get_ome()
masks = plate.get_masks()
rf = plate.get_region_features()

pickle_path = os.path.join(get_pickles_folder(), 'umap_eda.pickle')
if os.path.isfile(pickle_path):
    reducer, umap_result = pickle.load(open(pickle_path, 'rb'))
else:
    reducer = umap.UMAP(verbose=True)
    umap_result = reducer.fit_transform(rf.sum)
    pickle.dump((reducer, umap_result), open(pickle_path, 'wb'))
pass
