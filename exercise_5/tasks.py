import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from exercise_5.linear_approximator import LinearApproximator
from exercise_5.nonlinear_approximator import NonlinearApproximator
from exercise_5.utils import mean_squared_error
import matplotlib.pyplot as plt


def task2():
    delta_t = 0.1

    # initialization
    x0_f = "./data/linear_vectorfield_data_x0.txt"
    x1_f = "./data/linear_vectorfield_data_x1.txt"
    x0 = pd.read_csv(x0_f, header=None, delimiter=" ").values
    x1 = pd.read_csv(x1_f, header=None, delimiter=" ").values

    vector = (x1 - x0) / delta_t

    # first part
    linear_approximator = LinearApproximator()
    v_hat = linear_approximator.fit_predict(x0, vector)

    # second part
    x1_hat = v_hat * delta_t + x0
    error = mean_squared_error(x1, x1_hat)
    print("Mean Squared Error: ", error)

    # third part
    T = 100
    x0 = np.array([[10, 10]])
    x_hat_pred = [x0]

    for _ in np.arange(T, step=delta_t):
        v_hat = linear_approximator.predict(x_hat_pred[-1])
        x_hat = v_hat * delta_t + x_hat_pred[-1]
        x_hat_pred.append(x_hat)

    x_hat_pred = np.vstack(x_hat_pred)

    plt.figure()
    plt.plot(x_hat_pred[:, 0], x_hat_pred[:, 1])
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.show()

def task3():
    x0_f = "./data/nonlinear_vectorfield_data_x0.txt"
    x1_f = "./data/nonlinear_vectorfield_data_x1.txt"
    x0 = pd.read_csv(x0_f, header=None, delimiter=" ").values
    x1 = pd.read_csv(x1_f, header=None, delimiter=" ").values
    delta_t = 0.1

    vector = (x1 - x0) / delta_t

    # first part
    linear_approximator = LinearApproximator()
    v_hat = linear_approximator.fit_predict(x0, vector)
    x1_hat = v_hat * delta_t + x0
    error = mean_squared_error(x1, x1_hat)
    print("Mean Squared Error: ", error)

    # Second part
    epsilon = max(np.linalg.norm(x0[i] - x0[j]) for i in range(x0.shape[0]) for j in range(i, x0.shape[0]))
    l = 10
    nonlinear_approximator = NonlinearApproximator(l, epsilon)
    v_hat = nonlinear_approximator.fit_predict(x0, vector)
    x1_hat = v_hat * delta_t + x0
    error = mean_squared_error(x1, x1_hat)
    print("Mean Squared Error: ", error)

    # Third part
    T = 100
    x_hat_pred = [x0]
    v_hat_pred = [v_hat]

    for _ in np.arange(T, step=delta_t):
        v_hat = nonlinear_approximator.predict(x_hat_pred[-1])
        x_hat = v_hat * delta_t + x_hat_pred[-1]
        x_hat_pred.append(x_hat)
        v_hat_pred.append(v_hat)

    x_hat_pred = np.dstack(x_hat_pred)

    plt.figure(figsize=(15, 15))
    for i in range(x_hat_pred.shape[0]):
        plt.plot(x_hat_pred[i, 0, :], x_hat_pred[i, 1, :], c='blue', lw=0.2, alpha=0.5)
    plt.show()

def task4():
    data_f = "./data/takens_1.txt"
    data = pd.read_csv(data_f, header=None, delimiter=" ").values
    fig = plt.figure()
    plt.plot(np.arange(0, data.shape[0]), data[:, 0])
    plt.show()

    fig2 = plt.figure()
    plt.plot(data[70:, 0], data[:-70, 0])
    plt.show()



if __name__ == "__main__":
    task4()
