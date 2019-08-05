import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from mpl_toolkits.mplot3d import Axes3D

from spatial_ops.data import JacksonFischerDataset as jfd
from spatial_ops.folders import get_pickles_folder

patient_index = 0
plate_index = 0

plate = jfd.patients[patient_index].plates[plate_index]
ome = plate.get_ome()
masks = plate.get_masks()
rf = plate.get_region_features()

dont_use_pickle = False
dont_use_pickle = True
pickle_path = os.path.join(get_pickles_folder(), 'umap_eda.pickle')
if os.path.isfile(pickle_path) and not dont_use_pickle:
    reducer, umap_result = pickle.load(open(pickle_path, 'rb'))
else:
    reducer = umap.UMAP(verbose=True, n_components=2)
    umap_result = reducer.fit_transform(rf.sum)
    pickle.dump((reducer, umap_result), open(pickle_path, 'wb'))
print('umap done')
color_channel = 5
if umap_result.shape[1] == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    im = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=rf.sum[:, color_channel])
    fig.colorbar(im)
elif umap_result.shape[1] == 2:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=rf.sum[:, color_channel], s=1, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    # sns.violinplot(rf.sum[:, color_channel])
    plt.violinplot(rf.sum[:, color_channel])
else:
    raise ValueError(f'n_components = {umap_result.shape[1]}')
plt.show()
print('done')
