import numpy as np


def mean_squared_error(x, x_hat):
    return np.square(x - x_hat).sum()


def time_delay_embedding(data: np.ndarray, delta_t: int, delay: int) -> np.ndarray:
    p_matrix = []

    for t in range(data.shape[0] - delay):
        p_vector = []
        for i in range(delay):
            p_vector.append(data[t + i * delta_t])
        p_matrix.append(p_vector)

    return np.array(p_matrix)
