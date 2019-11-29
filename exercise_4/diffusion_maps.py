import numpy as np
from scipy.linalg import sqrtm

class DiffusionMap:
    def __init__(self):
        pass

    def fit(self, x: np.ndarray) -> 'DiffusionMap':
        if len(x.shape) != 2:
            raise ValueError
        N, dim = x.shape

        D = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                D[i, j] = np.abs(x[i] - x[j])
                D[j, i] = D[i, j]

        epsilon = 0.05 * D.max()

        W: np.ndarray = np.exp(-np.square(D) / epsilon)

        P = np.diag(W.sum(axis=1))

        P_inv = np.linalg.inv(P)
        K: np.ndarray = P_inv @ W @ P_inv

        Q = np.diag(K.sum(axis=1))

        Q_inv_sqrt = sqrtm(np.linalg.inv(Q))
        T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eig(T_hat)
        # TODO: unfinished

        return self

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        return self.fit(x).transform(x)
