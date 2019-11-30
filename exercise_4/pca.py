import numpy as np

class PCA:
    def __init__(self):
        self.S: [None, np.ndarray] = None
        self.V: [None, np.ndarray] = None
        self.is_fitted: bool = False
        self.max_dimensions: int = -1
        self.mean = None

    def fit(self, x: np.ndarray) -> 'PCA':
        if len(x.shape) != 2:
            raise ValueError

        x = x.copy()
        N, self.max_dimensions = x.shape

        self.mean = x.mean(axis=0).reshape(1, -1)
        x -= self.mean

        # set V
        eigenvalues_V, self.V = np.linalg.eig(x.T @ x)
        indices = eigenvalues_V.argsort()[::-1]
        eigenvalues_V = eigenvalues_V[indices]
        self.V = self.V[:, indices]

        # set S, only save the diagonal
        if self.max_dimensions > N:
            eigenvalues_V = eigenvalues_V[:N]
        self.S = eigenvalues_V

        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError
        if len(x.shape) != 2 or x.shape[1] != self.max_dimensions:
            raise ValueError
        return (x - self.mean) @ self.V

    def inverse_transform(self, US: np.ndarray) -> np.ndarray:
        if US.shape[1] > self.V.shape[0]:
            raise ValueError
        return US @ self.V.T[:US.shape[1], :] + self.mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def energy(self, L: int):
        if L > self.max_dimensions:
            raise ValueError
        return self.S[:L].sum() / self.S.sum()