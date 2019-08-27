import argparse
import pickle
import os
from multiprocessing import Pool
import random
from typing import List
import progressbar
import umap
from spatial_ops.data import PatientSource

import numpy as np
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from torch import autograd
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from spatial_ops.data import JacksonFischerDataset as jfd, Plate
from spatial_ops.folders import get_processed_data_folder

from spatial_ops.folders import mem
from sandbox.umap_eda import show_umap_embedding
from spatial_ops.lazy_loader import PickleLazyLoader


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.plates: List[Plate] = []
        for patient in jfd.patients:
            if patient.pid == 25 and patient.source == PatientSource.basel:
                self.plates.append(patient.plates[0])
            # self.plates.extend(patient.plates)
        self.biologically_relevant_channels = [k for k, v in jfd.get_biologically_relevant_channels().items()]

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, item):
        data = self.plates[item].get_region_features().mean
        # x = torch.from_numpy(data[:, self.biologically_relevant_channels]).float()
        x = torch.from_numpy(data).float()
        if self.mean is not None:
            x = x - self.mean
        if self.std is not None:
            x = x / self.std
        return x

    def channels_count(self):
        return self.__getitem__(0).shape[1]

    def move_to_torch_and_scale(self, data):
        x = torch.from_numpy(data).float()
        x = x - self.mean
        x = x / self.std
        return x


class VAEUmapBasel25Loader(PickleLazyLoader):
    def get_resource_unique_identifier(self) -> str:
        return 'umap_of_vae_basel25'

    def precompute(self):
        torch_model_path = os.path.join(get_processed_data_folder(), 'vae_torch.model_basel25')
        dataset = get_scaled_dataset()
        model = VAE(dataset.channels_count())
        model.load_state_dict(torch.load(torch_model_path))
        rf = self.associated_instance.get_region_features()
        x = dataset.move_to_torch_and_scale(rf.mean)
        means, logvars = model.encode(x)
        means = means.detach().numpy()
        logvars = logvars.detach().numpy()
        reducer = umap.UMAP(verbose=True, n_components=2)
        umap_result = reducer.fit_transform(means)
        data = (reducer, umap_result, means)
        return data


@mem.cache
def get_scaled_dataset():
    # ehi
    dataset = AutoEncoderDataset()
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    for data in dataset:
        scaler.partial_fit(data)

    mean = torch.from_numpy(scaler.mean_, ).float()
    mean = mean.view(-1, len(mean))
    std = torch.from_numpy(scaler.var_ ** 0.5).float()
    std = std.view(-1, len(std))
    dataset = AutoEncoderDataset(mean=mean, std=std)
    return dataset


# class WeightedDropout(torch.nn.Module):
#     def __init__(self, n_groups, probs=None):
#         super().__init__()
#         if probs is None:
#             probs = [1.0 - (1.0 / (10.0 ** i)) for i in range(n_groups)]
#         self.n_groups = n_groups
#         self.probs = probs
#
#         self.dropouts = nn.ModuleList([nn.Dropout(probs[i]) for i in range(n_groups)])
#
#     def forward(self, x):
#         x_tuple = torch.split(x, self.n_groups, dim=1)
#         res = []
#         for cx, dropout in zip(x_tuple, self.dropouts):
#             res.append(dropout(cx))
#         return torch.stack(res, dim=1)


class VAE(nn.Module):
    def __init__(self, features_count):
        super(VAE, self).__init__()
        self.features_count = features_count
        self.hidden_layer_dimensions = 5
        self.encoder0 = nn.Linear(self.features_count, 20)
        self.encoder1 = nn.Linear(20, 15)
        self.encoder2 = nn.Linear(15, 10)
        self.encoder3_mean = nn.Linear(10, self.hidden_layer_dimensions)
        self.encoder3_log_var = nn.Linear(10, self.hidden_layer_dimensions)
        self.decoder0 = nn.Linear(self.hidden_layer_dimensions, 10)
        self.decoder1 = nn.Linear(10, 15)
        self.decoder2 = nn.Linear(15, 20)
        self.decoder3 = nn.Linear(20, self.features_count)

    def encode(self, x):
        # print(x.shape)
        x = F.relu(self.encoder0(x))
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        mean = F.relu(self.encoder3_mean(x))
        log_var = F.relu(self.encoder3_log_var(x))
        # identity = nn.Identity(self.hidden_layer_dimensions)
        # dropout = WeightedDropout(self.hidden_layer_dimensions)
        # NO!!!!
        # f = lambda x: dropout(identity(x))
        # dropped_out_mean = f(mean)
        # dropped_out_log_var = f(log_var)
        # return dropped_out_mean, dropped_out_log_var
        return mean, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decoder0(z))
        z = F.relu(self.decoder1(z))
        z = F.relu(self.decoder2(z))
        decoded = torch.sigmoid(self.decoder3(z))
        return decoded

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.features_count))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # bce = F.binary_cross_entropy(recon_x, x.view(-1, dataset.channels_count()), reduction='sum')
    x = x.view(-1, dataset.channels_count())
    err = torch.norm(recon_x - x).mean()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    beta = 0.000001
    # beta = 1
    # magic_number = logvar.numel() * 11.0
    magic_number = 1
    kld = -0.5 / magic_number * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    regularizator = 1
    # print(f'err = {err}, kld = {kld}, err / regularizator = {err / regularizator}')
    return err / regularizator + kld


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(training_loader):
        data = data.to(device)
        optimizer.zero_grad()
        with autograd.detect_anomaly():
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            loss_item = loss.item()
            train_loss += loss_item
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(training_loader.dataset),
                           100. * batch_idx / len(training_loader),
                           loss_item / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(training_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(validation_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def get_data_points():
    # to later color the points matching the instance of origin
    instance_ids = []
    l = []
    with progressbar.ProgressBar(max_value=len(dataset)) as bar:
        data_points = torch.empty(size=[0, model.hidden_layer_dimensions])
        for i, data in enumerate(dataset):
            bar.update(i)
            means, logvars = model.encode(data)
            instance_ids.extend([i] * means.shape[0])
            l.append(means)
            # for j in range(means.shape[0]):
            #     mean = means[j, :]
            #     logvar = logvars[j, :]
            #     var = torch.exp(logvar)
            #     m = MultivariateNormal(mean, torch.diag(var))
            #     sample = m.sample()
            #     sample = sample.view(-1, len(sample))
            #     l.append(sample)
        data_points = torch.cat(l, dim=0)
        data_points = data_points.detach().numpy()
        a = 1
    return data_points, instance_ids


# def generate(epoch: int, digit: int):
#     with torch.no_grad():
#         sample = torch.randn(100, model.hidden_layer_dimensions).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#                    'results/sample_' + str(epoch) + '.png')

class VAEUmapLoader(PickleLazyLoader):
    def get_resource_unique_identifier(self) -> str:
        return 'umap_of_vae'

    def precompute(self):
        torch_model_path = os.path.join(get_processed_data_folder(), 'vae_torch.model_small_beta')
        dataset = get_scaled_dataset()
        model = VAE(dataset.channels_count())
        model.load_state_dict(torch.load(torch_model_path))
        rf = self.associated_instance.get_region_features()
        x = dataset.move_to_torch_and_scale(rf.mean)
        means, logvars = model.encode(x)
        means = means.detach().numpy()
        logvars = logvars.detach().numpy()
        reducer = umap.UMAP(verbose=True, n_components=2)
        umap_result = reducer.fit_transform(means)
        data = (reducer, umap_result, means)
        return data


def parallel_precompute_vae_umap_on_single_patient(patient):
    for plate in patient.plates:
        VAEUmapLoader(plate).load_data()


def parallel_precompute_vae_umap():
    with Pool(processes=4) as pool:
        pool.map(parallel_precompute_vae_umap_on_single_patient,
                 iterable=jfd.patients[:50])


# parallel_precompute_vae_umap()
# os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nn test')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if torch.cuda.is_available():
        print('cuda available')
        if use_cuda:
            print('using cuda')
        else:
            print('not using cuda')
    else:
        print('cuda not available')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')

    dataset = get_scaled_dataset()

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    training_data_fraction = 0.8
    split = int(np.floor(training_data_fraction * len(dataset)))
    training_indices, validation_indices = indices[:split], indices[split:]

    training_sampler = torch.utils.data.SubsetRandomSampler(training_indices)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)

    batch_size = 1
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=training_sampler, **kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=validation_sampler, **kwargs)

    model = VAE(dataset.channels_count()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # torch_model_path = os.path.join(get_processed_data_folder(), 'vae_torch.model_0.1_trick')
    # torch_model_path = os.path.join(get_processed_data_folder(), 'vae_torch.model_small_beta')
    torch_model_path = os.path.join(get_processed_data_folder(), 'vae_torch.model_basel25')
    rebuild_model = False
    rebuild_model = True
    if not os.path.isfile(torch_model_path) or rebuild_model:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            # generate(epoch, 5)
        print('saving the model')
        torch.save(model.state_dict(), torch_model_path)
    else:
        if not use_cuda or not torch.cuda.is_available():
            kwargs = {'map_location': 'cpu'}
        else:
            kwargs = {}
        model.load_state_dict(torch.load(torch_model_path))
    data_points, instance_ids = get_data_points()
    show_umap_embedding(data_points, instance_ids, joblib_seed=1)
print('done')
