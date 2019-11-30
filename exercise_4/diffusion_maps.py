import numpy as np
from scipy.linalg import sqrtm

class DiffusionMap:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, x: np.ndarray, L) -> 'DiffusionMap':
        if len(x.shape) != 2:
            raise ValueError

        N, dim = x.shape

        # create distance matrix D
        matrix = self.create_distance_matrix(x)

        epsilon = 0.01 * matrix.max()

        # create W
        matrix: np.ndarray = np.exp(-np.square(matrix) / epsilon)  # (N, N)

        # create K
        P_inv = np.linalg.inv(np.diag(matrix.sum(axis=1)))  # (N, N)
        matrix: np.ndarray = P_inv @ matrix @ P_inv  # (N, N)

        # create T hat
        Q_inv_sqrt = np.linalg.inv(np.diag(matrix.sum(axis=1)) ** 0.5) # (N, N)
        matrix = Q_inv_sqrt @ matrix @ Q_inv_sqrt

        # get eigenvalues of T hat and sort them
        eigenvalues, eigenvectors = np.linalg.eig(matrix)  # (N,), (N, N)
        indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[indices][:L+1]  # (L+1,)
        eigenvectors = eigenvectors[:, indices][:, :L+1]  # (N, L+1)

        self.eigenvalues = np.sqrt(eigenvalues ** (1 / epsilon))[1:]  # (L+1,) -> (L,)

        self.eigenvectors = (Q_inv_sqrt @ eigenvectors)[:, 1:]  # (N, N) x (N, L+1) = (N, L+1) -> (N, L)

        return self

    def transform(self, x):
        return (np.diag(self.eigenvalues) @ ((self.create_distance_matrix(x) ** 0.5) @ self.eigenvectors).T).T

    def fit_transform(self, x, L):
        return self.fit(x, L).transform(x)

    @staticmethod
    def create_distance_matrix(x):
        N = x.shape[0]
        matrix = np.empty((N, N))  # (N, N)
        for i in range(N):
            for j in range(i, N):
                matrix[i, j] = np.linalg.norm(x[i] - x[j])
                matrix[j, i] = matrix[i, j]
        return matrix