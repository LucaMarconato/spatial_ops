import argparse
import random
from typing import List

import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

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
        return data[:, self.biologically_relevant_channels]

    def channels_count(self):
        return self.__getitem__(0).shape[1]


dataset = AutoEncoderDataset()

indices = list(range(len(dataset)))
random.shuffle(indices)
training_data_fraction = 0.8
split = int(np.floor(training_data_fraction * len(dataset)))
training_indices, validation_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(training_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)

batch_size = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, **kwargs)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, **kwargs)


# print(train_loader.__iter__().__next__().shape)

class VAE(nn.Module):
    def __init__(self, hidden_layer_dim: int):
        super(VAE, self).__init__()
        self.hidden_layer_dim = hidden_layer_dim
        self.encoder0 = nn.Linear(dataset.channels_count(), 20)
        self.encoder1 = nn.Linear(20, 15)
        self.encoder2 = nn.Linear(15, 10)
        self.encoder3_mean = nn.Linear(5, self.hidden_layer_dim)
        self.encoder3_log_var = nn.Linear(5, self.hidden_layer_dim)
        self.decoder0 = nn.Linear(self.hidden_layer_dim, 5)
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

    def forward(self, x, condition):
        mu, logvar = self.encode(x.view(-1, dataset.channels_count()))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(hidden_layer_dim=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, dataset.channels_count()), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    regularizator = 1
    return bce / regularizator + kld


# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, condition) in enumerate(train_loader):
#         data = data.to(device)
#         condition = condition.to(device)
#         condition = condition.view(-1, 1)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data, condition)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader),
#                        loss.item() / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(train_loader.dataset)))
#
#
# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, condition) in enumerate(test_loader):
#             data = data.to(device)
#             condition = condition.to(device)
#             condition = condition.view(-1, 1)
#             recon_batch, mu, logvar = model(data, condition)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                         recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 os.makedirs('results', exist_ok=True)
#                 save_image(comparison.cpu(),
#                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#
#
# def generate(epoch: int, digit: int):
#     with torch.no_grad():
#         sample = torch.randn(64, hidden_layer_dimension).to(device)
#         condition = torch.ones(64)
#         condition = condition.to(device)
#         condition *= digit
#         condition = condition.view(-1, 1)
#         condition = condition.long()
#         sample = model.decode(sample, condition).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#                    'results/sample_' + str(epoch) + '.png')
#
#
# def display_image_from_pandas_row(row: torch.Tensor):
#     x = np.array(row)
#     x.shape = (28, 28)
#     x = x.astype(np.uint8)
#     img = Image.fromarray(x)
#     # img.save('my.png')
#     img.show()
#
#
# def tsne_plot():
#     pixels_columns = ['pixel' + str(i) for i in range(28 * 28)]
#     mnist_dataset = datasets.MNIST('./data', train=True, download=True,
#                                    transform=transforms.ToTensor())
#     data = mnist_dataset.data.view(60000, 28 * 28).cpu().detach().numpy()
#     data = data.astype(np.float)
#     data /= 255.0
#     df = pd.DataFrame(data)
#     df.columns = pixels_columns
#     df['digit'] = mnist_dataset.targets.cpu().detach().numpy()
#     df['original_data'] = True
#
#     n = 10000
#     seed = 42
#     np.random.seed(seed)
#     seed_pickle_path = 'random_seed.pickle'
#     force_umap_computation = True
#     if os.path.isfile(seed_pickle_path):
#         old_seed = pickle.load(open(seed_pickle_path, 'rb'))
#         if old_seed == seed:
#             force_umap_computation = False
#     pickle.dump(seed, open(seed_pickle_path, 'wb'))
#     random_permutation = np.random.permutation(n)
#     df = df.iloc[random_permutation[:n],].copy()
#     df.reset_index(drop=True, inplace=True)
#
#     # for each digit, generate new digits, then map them into the t-SNE space
#     n = 750
#
#     def generate_new_digits(digit: int, count: int = n) -> torch.Tensor:
#         with torch.no_grad():
#             sample = torch.randn(count, hidden_layer_dimension).to(device)
#             condition = torch.ones(count)
#             condition = condition.to(device)
#             condition *= digit
#             condition = condition.view(-1, 1)
#             condition = condition.long()
#             sample = model.decode(sample, condition).cpu()
#             # save_image(sample.view(n, 1, 28, 28),
#             #            f'cpu_generated_{i}.png',
#             #            nrow=int(np.sqrt(n)))
#             sample = sample.numpy()
#             return sample
#
#     for i in range(10):
#         generated = generate_new_digits(digit=i, count=n)
#         df_generated = pd.DataFrame(generated)
#         df_generated.columns = pixels_columns
#         df_generated['digit'] = i
#         df_generated['original_data'] = False
#         df = pd.concat([df, df_generated], axis=0)
#
#     df.reset_index(drop=True, inplace=True)
#
#     # small_df = df_merged.loc[random_permutation[:n],:].copy()
#     # small_data = small_df[pc_columns].values
#     pca_dimentions = 50
#     pca = PCA(n_components=pca_dimentions)
#     pca_result_original_data = pca.fit_transform(df.loc[df['original_data'] == True, pixels_columns])
#     pca_result_generated_data = pca.transform(df.loc[df['original_data'] == False, pixels_columns])
#     print(f'explained variance ratio = {np.sum(pca.explained_variance_ratio_)}')
#     pc_columns = ['pc' + str(i) for i in range(pca_dimentions)]
#     pca_original_df = pd.DataFrame(pca_result_original_data, columns=pc_columns,
#                                    index=df[df['original_data'] == True].index)
#     pca_generated_df = pd.DataFrame(pca_result_generated_data, columns=pc_columns,
#                                     index=df[df['original_data'] == False].index)
#     df = pd.merge(df, pca_original_df, how='outer', left_index=True, right_index=True)
#     df.fillna(pca_generated_df, inplace=True)
#     # df = pd.merge(df, pca_generated_df, how='outer', left_index=True, right_index=True)
#     # df = pd.concat([df, pca_df], axis=1, join_axes=[df.index])
#     data_for_tsne = df.loc[df['original_data'] == True, pc_columns].values
#
#     umap_pickle_path = 'umap.pickle'
#     if not os.path.isfile(umap_pickle_path) or force_umap_computation:
#         print('computing umap from scratch')
#         # reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#         reducer = umap.UMAP(verbose=True)
#         umap_result_original_data = reducer.fit_transform(data_for_tsne)
#         pickle.dump((reducer, umap_result_original_data), open(umap_pickle_path, 'wb'))
#     else:
#         print('loading umap from pickle')
#         reducer, umap_result_original_data = pickle.load(open(umap_pickle_path, 'rb'))
#
#     df['umap0'] = np.nan
#     df['umap1'] = np.nan
#     df.loc[df['original_data'] == True, ['umap0', 'umap1']] = umap_result_original_data
#     umap_result_generated_data = reducer.transform(df.loc[df['original_data'] == False, pc_columns].values)
#     df.loc[df['original_data'] == False, ['umap0', 'umap1']] = umap_result_generated_data
#     # df_umap_original = df.loc[df['original_data'] == True]
#     # df_umap_generated = df.loc[df['original_data'] == False]
#     data_to_save = df[pixels_columns + ['umap0', 'umap1', 'original_data', 'digit']]
#     pickle.dump(data_to_save, open('bokeh_data_to_show.pickle', 'wb'))
#     use_bokeh = False
#     if not use_bokeh:
#         plt.figure(figsize=(16, 9))
#         sns.scatterplot(
#             x='umap0', y='umap1',
#             hue='digit',
#             palette=sns.color_palette('hls', 10),
#             data=df[df['original_data'] == True],
#             legend='full',
#             alpha=0.15,
#         )
#         sns.scatterplot(
#             x='umap0', y='umap1',
#             hue='digit',
#             palette=sns.color_palette('hls', 10),
#             data=df[df['original_data'] == False],
#             # marker='+',
#             alpha=1
#         )
#         plt.show()
#     else:
#         from bokeh_plotting import display_bokeh
#         display_bokeh()
#
#
# if __name__ == "__main__":
#     torch_model_path = 'torch_model'
#     if not os.path.isfile(torch_model_path):
#         for epoch in range(1, args.epochs + 1):
#             train(epoch)
#             test(epoch)
#             generate(epoch, 5)
#         print('saving the model')
#         torch.save(model, torch_model_path)
#     else:
#         if not args.cuda:
#             kwargs = {'map_location': 'cpu'}
#         else:
#             kwargs = {}
#         model = torch.load(torch_model_path, **kwargs)
#     # for i in range(10):
#     #     generate(20 + i, i)
#     tsne_plot()
# print('done')
