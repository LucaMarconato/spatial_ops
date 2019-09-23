import umap
import os

from spatial_ops.common.data import JacksonFischerDataset as jfd, Plate
from spatial_ops.common.lazy_loader import PickleLazyLoader
from spatial_thorsten.ds.dataset import SpatialTranscriptomicsDs
from spatial_thorsten.nn.model import Model
import torch
from multiprocessing import Pool

root_folder = '/data/l989o/spatial_zurich/data'
model = Model(in_channels=38, k=3)
ds_train = SpatialTranscriptomicsDs(root_folder=root_folder, k=model.k, training=False,
                                    divideable_by=model.divideable_by)
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=1, num_workers=0)

predictions_folder = '/data/l989o/deployed/spatial_thorsten/out'
h5_files = os.listdir(predictions_folder)

class ThorstenPredictionsLoader(PickleLazyLoader):

    def get_resource_unique_identifier(self) -> str:
        return 'thorsten_predictions'

    def compute(self):
        from spatial_ops.common.data import Plate
        plate: Plate = self.associated_instance
        ome_path = plate.ome_path
        associated_h5_file = os.path.basename(ome_path).replace('.h5', '')
        if associated_h5_file in h5_files:
            print('found')
        else:
            print('not found')
        # dataset = get_standardized_dataset()
        # model = VAE(dataset.channels_count())
        # model.load_state_dict(torch.load(self.torch_model_path))
        # rf = self.associated_instance.get_region_features()
        # x = dataset.move_to_torch_and_scale(rf.mean)
        # x_reconstructed, means, log_vars = model(x)
        # x = dataset.inverse_scale(x)
        # x = x.detach().numpy()
        # x_reconstructed = dataset.inverse_scale(x_reconstructed)
        # x_reconstructed = x_reconstructed.detach().numpy()
        # means = means.detach().numpy()
        # log_vars = log_vars.detach().numpy()
        # reducer = umap.UMAP(verbose=True, n_components=2)
        # umap_result = reducer.fit_transform(means)
        # # data = (reducer, umap_result, means, log_vars, x_reconstructed)
        # # x = np.concatenate([np.zeros((1, x.shape[1])), x], axis=0)
        # # x_reconstructed = np.concatenate([np.zeros((1, x_reconstructed.shape[1])), x_reconstructed], axis=0)
        # data = (x, umap_result, x_reconstructed)
        return (x, umap_result, x_reconstructed)


def parallel_precompute_thorsten_predictions_on_single_patient(patient):
    for plate in patient.plates:
        ThorstenPredictionsLoader(plate).load_data()


def parallel_precompute_thorsten_predictions():
    with Pool(processes=4) as pool:
        pool.map(parallel_precompute_thorsten_predictions_on_single_patient,
                 iterable=jfd.patients)

# parallel_precompute_thorsten_predictions()
