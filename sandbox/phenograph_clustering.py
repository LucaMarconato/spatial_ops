import phenograph
import numpy as np
import random

from spatial_ops.common.data import JacksonFischerDataset as jfd
to_concat = []

for patient in jfd.patients:
    for plate in patient.plates:
        rf = plate.get_region_features()
        to_concat.append(rf.mean)

x = np.concatenate(to_concat, axis=0)
random.shuffle(x)
x = x[:100000]
communities, graph, Q = phenograph.cluster(x, k=20)
print('done')
pass
