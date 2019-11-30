import sys
sys.path.append('../..')

from exercise_4.pca import PCA
from exercise_4.diffusion_maps import DiffusionMap
from exercise_4.vae import VAE
import pandas as pd
from scipy.misc import face
import matplotlib.pyplot as plt
import numpy as np
from codecs import decode
from sklearn.datasets import make_swiss_roll
from functools import reduce
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch


def task1_1():
    x = pd.read_csv("data/pca_dataset.txt", delimiter=" ", header=None).values
    pca = PCA().fit(x)
    plt.figure()
    plt.scatter(*x.T)
    v_t = pca.V_T
    plt.quiver([0], [0], *v_t, scale=5,color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Principal Components of Dataset 1')
    plt.show()
    print('energy of pc1:', pca.energy(1), 'energy of pc2', pca.energy(2) - pca.energy(1))


def task1_2():
    image = face().astype(np.float64)

    for compression in [120]:
        reconstructed = np.zeros_like(image)
        for row_index, column_rgb in enumerate(image):
            pca = PCA()
            compressed = pca.fit_transform(column_rgb.T)[:, :compression]
            reconstructed[row_index] = pca.inverse_transform(compressed).T
            print('processed row', row_index)
        plt.title('Reconstructed image for compression'+str(compression))
        plt.imshow(reconstructed.astype(int))


def task1_3():
    x = pd.read_csv("data/data_DMAP_PCA_vadere.txt", delimiter=" ", header=None).values

    plt.figure()
    plt.plot(*x[:, :2].T, label='Pedestrian 1')
    plt.plot(*x[:, 2:4].T, label='Pedestrian 2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of the first 2 pedestrians')
    plt.show()

    pca = PCA()
    transformed = pca.fit_transform(x)[:, :2]
    print('energy of pc1:', pca.energy(1), 'energy of pc2', pca.energy(2) - pca.energy(1))

    reconstructed = pca.inverse_transform(transformed)
    plt.figure()
    plt.plot(*reconstructed[:, :2].T, label='Pedestrian 1')
    plt.plot(*reconstructed[:, 2:4].T, label='Pedestrian 2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reconstructed trajectories of the first 2 pedestrians')
    plt.show()


def task2_1():
    N = 1000
    X = np.empty((N, 2))
    t = (2 * np.pi * np.arange(1, N+1)) / (N + 1)
    for k in range(N):
        X[k] = [np.cos(t[k]), np.sin(t[k])]

    dm = DiffusionMap()
    dm.fit(X, 5)
    transformed = dm.transform(X)

    plt.figure()
    for l in range(5):
        plt.plot(t, transformed[:, l], label='\u03d5'+decode(r'\u208{}'.format(l+1), 'unicode_escape'))
    plt.legend(loc='upper left')
    plt.xlabel('t\u2096')
    plt.ylabel('\u03d5')
    plt.show()


def task2_2():
    N = 1000
    X = make_swiss_roll(N, random_state=7)[0]

    dm = DiffusionMap()
    dm.fit(X, 10)
    transformed = dm.transform(X)

    fig, axes = plt.subplots(3, 3, sharex='all')
    fig.suptitle('Eigenfunctions by the first eigenfunction')

    for l, ax in enumerate(axes.flat, 2):
        ax.scatter(transformed[:, 0], transformed[:, l-1], label='\u03d5' + decode(r'\u208{}'.format(l),
                                                                                   'unicode_escape'), marker='o', s=1)
        ax.set_ylabel('\u03d5' + decode(reduce(lambda a, b: a+b, [r'\u208'+c for c in str(l)]), 'unicode_escape'))
        ax.set_xlabel('\u03d5\u2081')
    fig.show()

    pca = PCA()
    transformed = pca.fit_transform(X)
    print('pc3', pca.energy(3) - pca.energy(2), '\npc2', pca.energy(2) - pca.energy(1), '\npc1', pca.energy(1))
    print(pca.energy(2))


def task2_3():
    x = pd.read_csv("data/data_DMAP_PCA_vadere.txt", delimiter=" ", header=None).values

    dm = DiffusionMap()
    dm.fit(x, 2)


def task3():
    batch_size = 128
    latent_vector_sizes = [2, 32]
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_set = MNIST(root='./mnist', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = MNIST(root='./mnist', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    for latent_vector_size in latent_vector_sizes:
        configs = {'latent_vector_size': latent_vector_size,
                   'print_output': True,
                   'batch_size': batch_size,
                   'learning_rate': 0.001,
                   'epochs': 50,
                   'train_dataloader': train_dataloader,
                   'test_dataloader': test_dataloader,
                   'dataset_dims': 784}

        vae = VAE(**configs)
        vae.train()


def task4():
    batch_size = 128

    train_set = torch.tensor(np.load("./data/FireEvac_train_set.npy"), dtype=torch.float).cuda()
    test_set = torch.tensor(np.load("./data/FireEvac_test_set.npy"), dtype=torch.float).cuda()

    train_dataset = TensorDataset(train_set, torch.zeros(size=(train_set.shape[0], )))
    test_dataset = TensorDataset(test_set, torch.zeros(size=(test_set.shape[0], )))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    configs = {'latent_vector_size': 32,
               'print_output': False,
               'batch_size': batch_size,
               'learning_rate': 0.001,
               'epochs': 50,
               'train_dataloader': train_dataloader,
               'test_dataloader': test_dataloader,
               'dataset_dims': 2}

    vae = VAE(**configs)
    vae.train()


if __name__== "__main__":
    task4()
