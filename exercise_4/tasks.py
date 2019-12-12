import sys

sys.path.append('../..')

from exercise_4.pca import PCA
from exercise_4.diffusion_maps import DiffusionMap
from exercise_4.vae import VAE
import pandas as pd
from scipy.misc import face
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from codecs import decode
from sklearn.datasets import make_swiss_roll
from functools import reduce
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
from sklearn.preprocessing import normalize


def task1_1():
    x = pd.read_csv("data/pca_dataset.txt", delimiter=" ", header=None).values
    pca = PCA().fit(x)
    plt.figure()
    plt.scatter(*x.T)
    plt.quiver([0], [0], *pca.V, scale=5, color='red')
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
    t = (2 * np.pi * np.arange(1, N + 1)) / (N + 1)
    for k in range(N):
        X[k] = [np.cos(t[k]), np.sin(t[k])]

    dm = DiffusionMap()
    transformed = dm.fit_transform(X, 5)

    plt.figure()
    for l in range(5):
        plt.plot(t, transformed[:, l], label='\u03d5' + decode(r'\u208{}'.format(l + 1), 'unicode_escape'))
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
        ax.scatter(transformed[:, 0], transformed[:, l - 1], label='\u03d5' + decode(r'\u208{}'.format(l),
                                                                                     'unicode_escape'), marker='o', s=1)
        ax.set_ylabel('\u03d5' + decode(reduce(lambda a, b: a + b, [r'\u208' + c for c in str(l)]), 'unicode_escape'))
        ax.set_xlabel('\u03d5\u2081')
    fig.show()

    pca = PCA()
    transformed = pca.fit_transform(X)
    print('pc3', pca.energy(3) - pca.energy(2), '\npc2', pca.energy(2) - pca.energy(1), '\npc1', pca.energy(1))
    print(pca.energy(2))


def task2_3():
    x = pd.read_csv("data/data_DMAP_PCA_vadere.txt", delimiter=" ", header=None).values
    N, dim = x.shape
    L = 3

    dm = DiffusionMap()
    transformed = dm.fit_transform(x, L)

    plt.figure()
    for l, col in enumerate(transformed.T):
        plt.scatter(np.arange(N), col, label='\u03d5' + decode(r'\u208{}'.format(l + 1), 'unicode_escape'), s=1)
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('mapping of pedestrian coordinates')
    plt.title('Mappings of Eigenfunctions')
    plt.show()

    for i in range(L):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=transformed[:, i], cmap='plasma')
        ax.set_title(f'Eigenfunction {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.show()


def task3():
    # initializations
    batch_size = 128
    latent_vector_sizes = [2, 32]

    # initialize the transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0], std=[1])]
    )

    # load the mnist dataset's training and test sets
    train_set = MNIST(root='./mnist', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = MNIST(root='./mnist', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # run for each latent vector size
    for latent_vector_size in latent_vector_sizes:
        # initialize the configurations
        configs = {'latent_vector_size': latent_vector_size,
                   'print_output': True,
                   'batch_size': batch_size,
                   'learning_rate': 0.001,
                   'epochs': 50,
                   'train_dataloader': train_dataloader,
                   'test_dataloader': test_dataloader,
                   'dataset_dims': 784,
                   'test_count': 16,
                   'kl_annealing': False,
                   'beta_limit': 1,
                   'reconstruction_loss_scale': 1}

        vae = VAE(**configs)
        vae.train()


def task4_plot(data, set_type, lower_left_critical_region, height, width):
    fig, ax = plt.subplots(1)
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("{} set scatter plot".format(set_type))
    rect = patches.Rectangle(lower_left_critical_region, height=height, width=width,
                             linewidth=1,
                             edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    plt.savefig('plots/task4/{}_set_scatter'.format(set_type))
    plt.close(fig)


def task4():
    batch_size = 1024

    # the critical region coordinates
    critical_region = np.array([[130, 70], [150, 50]])

    # load the train and test set of FireEvac dataset
    train_set = np.load("./data/FireEvac_train_set.npy")
    test_set = np.load("./data/FireEvac_test_set.npy")

    # concatenate the critical region coordinates with training set so that they can be normalized
    train_set_critical_region = np.vstack([train_set, critical_region])

    # normalize the training, test and critical region coordinates
    train_set_critical_region = normalize(train_set_critical_region, axis=0, norm="max")
    critical_region = train_set_critical_region[-2:, :]

    # separate the training set from the critical region coordinates
    train_set = train_set_critical_region[:train_set.shape[0], :]
    test_set = normalize(test_set, axis=0, norm="max")

    # calculate the height and width of the critical region, along with the lower left corner's coordinates
    height = np.abs(critical_region[0, 0] - critical_region[1, 0])
    width = np.abs(critical_region[0, 1] - critical_region[1, 1])
    lower_left_critical_region = (critical_region[1, 0] - width, critical_region[1, 1])

    task4_plot(train_set, "training", lower_left_critical_region, height, width)
    task4_plot(test_set, "test", lower_left_critical_region, height, width)

    train_set = torch.tensor(train_set, dtype=torch.float).cuda()
    test_set = torch.tensor(test_set, dtype=torch.float).cuda()

    # initializing the pytorch dataset and data loader
    train_dataset = TensorDataset(train_set, torch.zeros(size=(train_set.shape[0],)))
    test_dataset = TensorDataset(test_set, torch.zeros(size=(test_set.shape[0],)))

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
               'test_count': 1000,
               'kl_annealing': True,
               'beta_limit': 1,
               'reconstruction_loss_scale': 100}

    vae = VAE(**configs)
    # vae.train()
    vae.model.load_state_dict(torch.load(vae.model_path))

    reconstructed_data = vae.test()
    task4_plot(reconstructed_data, "reconstructed", lower_left_critical_region, height, width)

    generated_data = vae.generate().cpu().detach().numpy()
    task4_plot(generated_data, "generated", lower_left_critical_region, height, width)

    # task 4 part 5, sample from the approximate posterior distribution
    num = 0
    samples = 0
    generated_data = np.array(np.zeros(shape=(1, 2)))

    while num < 100:
        data = np.squeeze(vae.generate(count=1).cpu().detach().numpy())
        if (lower_left_critical_region[0] + height) > data[0] > (lower_left_critical_region[0]) \
                and (lower_left_critical_region[1] + width) > data[1] > lower_left_critical_region[1]:
            num += 1
        samples += 1
        generated_data = np.vstack([generated_data, data])

    task4_plot(generated_data, "sampled", lower_left_critical_region, height, width)


if __name__ == "__main__":
    task4()
