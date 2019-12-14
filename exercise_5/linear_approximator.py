import numpy as np
from scipy.linalg import lstsq


class LinearApproximator:
    def __init__(self):
        self.coefficients = None
        self.fitted = False

    def fit(self, x, y) -> 'LinearApproximator':
        if self.fitted:
            raise RuntimeError
        self.coefficients, *_ = lstsq(x, y)

        self.fitted = True
        return self

    def predict(self, x) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError
        return x @ self.coefficients

    def fit_predict(self, x, y) -> np.ndarray:
        return self.fit(x, y).predict(x)
