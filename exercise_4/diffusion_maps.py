import numpy as np
from scipy.linalg import sqrtm


class DiffusionMap:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def fit_transform(self, x: np.ndarray, L) -> np.ndarray:
        """

        :param x: points of multidimensional represented as a 2D matrix
        :param L: number of eigenfunctions to take
        :return: first L eigenfunction mappings of x
        """
        if len(x.shape) != 2:
            raise ValueError('x hast to be a 2 dimensional numpy array')


        # create distance matrix D
        matrix = self.create_distance_matrix(x)

        epsilon = 0.05 * matrix.max()

        # create W
        matrix: np.ndarray = np.exp(-np.square(matrix) / epsilon)  # (N, N)

        # create K
        P_inv = np.linalg.inv(np.diag(matrix.sum(axis=1)))  # (N, N)
        matrix: np.ndarray = P_inv @ matrix @ P_inv  # (N, N)
        del P_inv

        # create T hat
        Q_inv_sqrt = np.linalg.inv(sqrtm(np.diag(matrix.sum(axis=1)))) # (N, N)
        matrix = Q_inv_sqrt @ matrix @ Q_inv_sqrt

        # get eigenvalues of T hat and sort them
        eigenvalues, eigenvectors = np.linalg.eig(matrix)  # (N,), (N, N)
        indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[indices][:L+1]  # (L+1,)
        eigenvectors = eigenvectors[:, indices][:, :L+1]  # (N, L+1)

        self.eigenvalues = np.sqrt(eigenvalues ** (1 / epsilon))[1:]  # (L+1,) -> (L,)

        self.eigenvectors = (Q_inv_sqrt @ eigenvectors)[:, 1:]  # (N, N) x (N, L+1) = (N, L+1) -> (N, L)

        return self.eigenvectors * self.eigenvalues.reshape(1, -1)

    @staticmethod
    def create_distance_matrix(x):
        """

        :param x:  points of multidimensional represented as a 2D matrix
        :return: The distance matrix of each pair of points in x
        """
        N = x.shape[0]
        matrix = np.empty((N, N))  # (N, N)
        for i in range(N):
            for j in range(i, N):
                matrix[i, j] = np.linalg.norm(x[i] - x[j])
                matrix[j, i] = matrix[i, j]
        return matrix
