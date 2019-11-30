import numpy as np
from scipy.linalg import sqrtm

class DiffusionMap:
    def __init__(self):
        self.eigenvalues = None
        self.eigenfunctions = None

    def fit(self, x: np.ndarray, L) -> 'DiffusionMap':
        if len(x.shape) != 2:
            raise ValueError

        N, dim = x.shape

        D = np.empty((N, N))  # (N, N)
        for i in range(N):
            for j in range(i, N):
                D[i, j] = np.abs(x[i] - x[j])
                D[j, i] = D[i, j]

        epsilon = 0.05 * D.max()

        W: np.ndarray = np.exp(-np.square(D) / epsilon)  # (N, N)

        P = np.diag(W.sum(axis=1))  # (N, N)

        P_inv = np.linalg.inv(P)  # (N, N)
        K: np.ndarray = P_inv @ W @ P_inv  # (N, N)

        Q = np.diag(K.sum(axis=1))  # (N, N)

        Q_inv_sqrt = sqrtm(np.linalg.inv(Q))  # (N, N)
        T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eig(T_hat)  # (N,), (N, N)
        indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[indices][:L+1]  # (L+1,)
        eigenvectors = eigenvectors[:, indices][:, :L+1]  # (N, L+1)

        eigenvalues = np.sqrt(eigenvalues ** (1 / epsilon))  # (L+1,)

        eigenvectors = Q_inv_sqrt @ eigenvectors  # (N, N) x (N, L+1) = (N, L+1)
        # TODO: unfinished

        return self

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        return self.fit(x).transform(x)
