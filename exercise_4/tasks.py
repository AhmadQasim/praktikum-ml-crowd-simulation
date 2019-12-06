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
    plt.quiver([0], [0], *pca.V, scale=5,color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Principal Components of Dataset 1')
    plt.show()
    print('energy of pc1:', pca.energy(1), 'energy of pc2', pca.energy(2) - pca.energy(1))


def task1_2():
    image = face().astype(np.float64)
    flattened_representation = np.zeros((image.shape[1], image.shape[0] * image.shape[2]))

    for column_index in range(image.shape[1]):
        flattened_representation[column_index] = image[:, column_index, :].flatten()

    pca = PCA()
    compressed = pca.fit_transform(flattened_representation)

    for L in [10, 50, 120, image.shape[1]]:
        reconstructed_flattened_representation = pca.inverse_transform(compressed[:, :L])
        reconstructed_image = np.zeros_like(image)
        for i, col in enumerate(reconstructed_flattened_representation):
            reconstructed_image[:, i, :] = col.reshape(-1, 3)

        plt.figure()
        plt.title(f'Reconstructed image with first {L} principal components')
        plt.axis('off')
        plt.imshow(reconstructed_image.astype(np.int))


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
    transformed = dm.fit_transform(X, 5)

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
    transformed = dm.fit_transform(X, 10)

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
                   'dataset_dims': 784,
                   'mode': 'mnist',
                   'test_count': 16}

        vae = VAE(**configs)
        vae.train()


def task4_plot(data, set_type):
    fig = plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("{} set scatter plot".format(set_type))
    plt.savefig('plots/task4/{}_set_scatter'.format(set_type))
    plt.close(fig)


def task4():
    batch_size = 1024

    train_set = np.load("./data/FireEvac_train_set.npy")
    test_set = np.load("./data/FireEvac_test_set.npy")

    # task4_plot(train_set, "training")
    # task4_plot(test_set, "test")

    train_set = torch.tensor(train_set, dtype=torch.float).cuda()
    test_set = torch.tensor(test_set, dtype=torch.float).cuda()

    train_dataset = TensorDataset(train_set, torch.zeros(size=(train_set.shape[0], )))
    test_dataset = TensorDataset(test_set, torch.zeros(size=(test_set.shape[0], )))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    configs = {'latent_vector_size': 4,
               'print_output': False,
               'batch_size': batch_size,
               'learning_rate': 0.0001,
               'epochs': 1000,
               'train_dataloader': train_dataloader,
               'test_dataloader': test_dataloader,
               'dataset_dims': 2,
               'mode': 'mi',
               'test_count': 100}

    vae = VAE(**configs)
    vae.train()
    reconstructed_data = vae.test()
    task4_plot(reconstructed_data, "reconstructed")

    generated_data = vae.test(reconstructed=False)
    task4_plot(generated_data, "generated")


if __name__ == "__main__":
    task2_1()
