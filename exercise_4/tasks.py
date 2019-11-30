from pca import PCA
import pandas as pd
from scipy.misc import face
import matplotlib.pyplot as plt
import numpy as np

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


if __name__== "__main__":
    task1_3()