import numpy as np
from scipy.linalg import lstsq


class NonlinearApproximator:

    def __init__(self, L, epsilon):
        self.center_points = None
        self.coefficients = None
        self.fitted = False

        self.L = L
        self.epsilon_square = epsilon ** 2

    def fit(self, x, y) -> 'NonlinearApproximator':
        if self.fitted:
            raise RuntimeError
        if not (len(x.shape) == len(y.shape) == 2):
            raise ValueError

        indices = np.random.choice(x.shape[0], size=self.L, replace=False)
        self.center_points = x[indices].copy()

        radial_basis_functions = self._create_radial_basis_functions(x)

        self.coefficients, *_ = lstsq(radial_basis_functions, y)

        self.fitted = True
        return self

    def predict(self, x) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError
        return self._create_radial_basis_functions(x) @ self.coefficients

    def fit_predict(self, x, y) -> np.ndarray:
        return self.fit(x, y).predict(x)

    def _create_radial_basis_functions(self, x) -> np.ndarray:
        radial_basis_functions = np.empty((x.shape[0], self.L))
        for l in range(self.L):
            square_norm = np.linalg.norm(x - self.center_points[l, :], axis=1) ** 2
            radial_basis_functions[:, l] = np.exp(-square_norm / self.epsilon_square)
        return radial_basis_functions

