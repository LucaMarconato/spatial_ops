import argparse
import os
import random
from typing import List

import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import autograd

from spatial_ops.data import JacksonFischerDataset as jfd, Plate

parser = argparse.ArgumentParser(description='nn test')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if torch.cuda.is_available():
    print('cuda available')
    if args.cuda:
        print('using cuda')
    else:
        print('not using cuda')
else:
    print('cuda not available')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True):
        self.plates: List[Plate] = []
        for patient in jfd.patients:
            self.plates.extend(patient.plates)
        if shuffle:
            random.shuffle(self.plates)
        self.biologically_relevant_channels = [k for k, v in jfd.get_biologically_relevant_channels().items()]

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, item):
        data = self.plates[item].get_region_features().sum
        x = torch.from_numpy(data[:, self.biologically_relevant_channels]).float()
        return x

    def channels_count(self):
        return self.__getitem__(0).shape[1]


dataset = AutoEncoderDataset()

indices = list(range(len(dataset)))
random.shuffle(indices)
training_data_fraction = 0.8
split = int(np.floor(training_data_fraction * len(dataset)))
training_indices, validation_indices = indices[:split], indices[split:]

training_sampler = torch.utils.data.SubsetRandomSampler(training_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)

batch_size = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=training_sampler, **kwargs)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=validation_sampler, **kwargs)


# print(training_loader.__iter__().__next__().shape)

class VAE(nn.Module):
    def __init__(self, hidden_layer_dim: int):
        super(VAE, self).__init__()
        self.hidden_layer_dimensions = hidden_layer_dim
        self.encoder0 = nn.Linear(dataset.channels_count(), 20)
        self.encoder1 = nn.Linear(20, 15)
        self.encoder2 = nn.Linear(15, 10)
        self.encoder3_mean = nn.Linear(10, self.hidden_layer_dimensions)
        self.encoder3_log_var = nn.Linear(10, self.hidden_layer_dimensions)
        self.decoder0 = nn.Linear(self.hidden_layer_dimensions, 10)
        self.decoder1 = nn.Linear(10, 15)
        self.decoder2 = nn.Linear(15, 20)
        self.decoder3 = nn.Linear(20, dataset.channels_count())

    def encode(self, x):
        x = F.relu(self.encoder0(x))
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        mean = F.relu(self.encoder3_mean(x))
        log_var = F.relu(self.encoder3_log_var(x))
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
        mu, logvar = self.encode(x.view(-1, dataset.channels_count()))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(hidden_layer_dim=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # bce = F.binary_cross_entropy(recon_x, x.view(-1, dataset.channels_count()), reduction='sum')
    x = x.view(-1, dataset.channels_count())
    err = torch.norm(recon_x - x).mean()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    regularizator = 1
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


def get_embedding():
    for data in dataset:
        mean, logvar = model.encode(data)
        pass


# def generate(epoch: int, digit: int):
#     with torch.no_grad():
#         sample = torch.randn(100, model.hidden_layer_dimensions).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#                    'results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    torch_model_path = 'torch_model'
    rebuild_model = False
    rebuild_model = True
    if not os.path.isfile(torch_model_path) or rebuild_model:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            # generate(epoch, 5)
        print('saving the model')
        torch.save(model, torch_model_path)
    else:
        if not args.cuda:
            kwargs = {'map_location': 'cpu'}
        else:
            kwargs = {}
        model = torch.load(torch_model_path, **kwargs)
print('done')
