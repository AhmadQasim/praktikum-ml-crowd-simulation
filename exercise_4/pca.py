import numpy as np

class PCA:
    def __init__(self):
        self.S: [None, np.ndarray] = None
        self.V: [None, np.ndarray] = None
        self.is_fitted: bool = False
        self.max_dimensions: int = -1
        self.mean: [None, np.ndarray] = None

    def fit(self, x: np.ndarray) -> 'PCA':
        """

        :param x: input to be fitted
        :return: the object itself
        """
        if len(x.shape) != 2:
            raise ValueError('Input has to be 2 dimensional.')

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
        """

        :param x: input to be transformed
        :return: the transformed space with all of its principal components
        """
        if not self.is_fitted:
            raise RuntimeError('PCA is not fitted yet.')
        if len(x.shape) != 2 or x.shape[1] != self.max_dimensions:
            raise ValueError(f'Input has invalid shape {x.shape}. A shape of (N, {self.max_dimensions}) is needed.')

        return (x - self.mean) @ self.V

    def inverse_transform(self, US: np.ndarray) -> np.ndarray:
        """

        :param US:
        :return: the reconstructed space
        """
        if not self.is_fitted:
            raise RuntimeError('PCA is not fitted yet.')
        if len(US.shape) != 2 or US.shape[1] > self.max_dimensions:
            raise ValueError(f'Input has invalid shape {US.shape}. A shape of (N, \u2264{self.max_dimensions}) is needed.')

        return US @ self.V.T[:US.shape[1], :] + self.mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """

        :param x: input to be fitted and transformed
        :return: the transformed space with all of its principal components
        """
        return self.fit(x).transform(x)

    def energy(self, L: int):
        """

        :param L: the first L principal components
        :return: the 'energy' of the first L components
        """
        if not self.is_fitted:
            raise RuntimeError('PCA is not fitted yet.')
        if L > self.max_dimensions:
            raise ValueError(f'L={L} is too large. The condition L\u2264{self.max_dimensions} has to be satisfied.')

        return self.S[:L].sum() / self.S.sum()
