import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from exercise_3.task5.utils import parse_trajectories
from exercise_3.task4.task4_lorenz import lorenz_attractor
from exercise_4.pca import PCA
from exercise_5.linear_approximator import LinearApproximator
from exercise_5.nonlinear_approximator import NonlinearApproximator
from exercise_5.utils import mean_squared_error, time_delay_embedding
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler


def task1():
    # Part 1
    linear_f = "./data/linear_function_data.txt"
    nonlinear_f = "./data/nonlinear_function_data.txt"

    linear = pd.read_csv(linear_f, header=None, delimiter=" ").values
    nonlinear = pd.read_csv(nonlinear_f, header=None, delimiter=" ").values

    linear_approximator = LinearApproximator()
    # Add ones for bias
    x_linear = np.vstack([linear[:, 0], np.ones(len(linear[:, 0]))]).T
    linear_predictions_A = linear_approximator.fit_predict(x_linear, linear[:, 1])

    fig = plt.figure()
    plt.plot(linear[:, 0], linear[:, 1], 'o', label='Original data')
    plt.plot(linear[:, 0], linear_predictions_A, label="Linear approximation of linear data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of linear data and its linear approximation')
    plt.legend()
    plt.show()

    # Part 2
    linear_approximator = LinearApproximator()
    # Add ones for bias
    x_nonlinear = np.vstack([nonlinear[:, 0], np.ones(len(nonlinear[:, 0]))]).T
    linear_predictions_B = linear_approximator.fit_predict(x_nonlinear, nonlinear[:, 1])
    fig = plt.figure()
    plt.plot(nonlinear[:, 0], nonlinear[:, 1], 'o', label='Original data')
    plt.plot(nonlinear[:, 0], linear_predictions_B, label='Linear approximation of nonlinear data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Plot of nonlinear data and its linear approximation")
    plt.legend()
    plt.show()


    # Part 3
    epsilon = max(np.linalg.norm(nonlinear[i, 0] - nonlinear[j, 0]) for i in range(nonlinear.shape[0]) for j in range(i, nonlinear.shape[0]))
    l = 15
    y = np.expand_dims(nonlinear[:, 1], axis=1)
    nonlinear_approximator = NonlinearApproximator(l, epsilon)
    nonlinear_predictions_B = nonlinear_approximator.fit_predict(x_nonlinear, y)

    fig = plt.figure()
    plt.plot(nonlinear[:, 0], nonlinear[:, 1], ' o', label='Original data')
    plt.plot(nonlinear[:, 0], nonlinear_predictions_B, 'o', label='Nonlinear approximation of nonlinear data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nonlinear data and its nonlinear approximation')
    plt.legend()
    plt.show()


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

    # second part
    dt = 0.01

    trajectory = solve_ivp(lorenz_attractor, [0, 1000], y0=[10, 10, 10], t_eval=np.linspace(0, 1000, num=int(1000/dt))).y

    x = trajectory[0, :]
    skip_index = int((0.75 / 4) / dt)

    x1, x2, x3 = x[:-2*skip_index], x[skip_index:-skip_index], x[2*skip_index:]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x1, x2, x3, c='darkcyan', lw=0.2, alpha=0.75)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    z = trajectory[2, :]
    skip_index = int((0.81 / 4) / dt)

    z1, z2, z3 = z[2 * skip_index:], z[skip_index:-skip_index], z[:-2 * skip_index]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('trajectory only from z')
    ax.plot(z1, z2, z3, c='darkcyan', lw=0.2, alpha=0.75)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    # second part bonus task
    dx, dy, dz = lorenz_attractor(ts=None, point=(x1, x2, x3))
    print("Lorenz Attractor")
    vector = np.hstack([dx.reshape(-1, 1), dy.reshape(-1, 1), dz.reshape(-1, 1)])
    print("X1: ", x1.shape)
    epsilon = 10 # max(np.linalg.norm(x1[i] - x1[j]) for i in range(x1.shape[0]) for j in range(i, x1.shape[0]))
    print("Epsilon")
    l = 15
    nonlinear_approximator = NonlinearApproximator(l, epsilon)
    v_hat = nonlinear_approximator.fit_predict(np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)]), vector)
    error = mean_squared_error(vector, v_hat)
    print("Mean Squared Error: ", error)

    def derivative_func(t, point):
        return nonlinear_approximator.predict(point.reshape(-1, 1)).reshape(-1)

    predicted_trajectory = solve_ivp(derivative_func, [0, 1000],
                                     y0=np.array([x1[0], x2[0], x3[0]]),
                                     t_eval=np.linspace(0, 1000, num=int(1000/dt))).y
    training_trajectory = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*training_trajectory.T, label="training")
    ax.plot(*predicted_trajectory.T, label='predicted')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.show()

    exit(1)

    # third part
    trajectory_path = './data/postvis.trajectories'
    coordinates = parse_trajectories(trajectory_path)

    # Take x coordinates from first pedestrian (Shape of 3751)
    xs = coordinates[1][0, :]
    p_matrix = time_delay_embedding(xs, delta_t=1, delay=200)

    pca = PCA().fit(p_matrix)
    transformed = pca.transform(p_matrix)[:, :2]

    # Plot first two principal components
    plt.figure()
    plt.plot(*transformed.T)
    plt.title('First 2 Principal Components')
    plt.show()


def task5():
    # Part 1
    data = pd.read_csv("./data/MI_timesteps.txt", sep=' ', dtype=np.float64)
    data = data.iloc[1000:, 1:].values  # ignore the first 1000 time steps, do not include the time step column
    delay = 350
    delta_t = 1
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    """
    embedding_area_1 = time_delay_embedding(data[:, 0], delta_t=delta_t, delay=delay)
    embedding_area_2 = time_delay_embedding(data[:, 1], delta_t=delta_t, delay=delay)
    embedding_area_3 = time_delay_embedding(data[:, 2], delta_t=delta_t, delay=delay)

    embedding = np.hstack([embedding_area_1, embedding_area_2, embedding_area_3])

    pca = PCA().fit(embedding)
    embedding_transformed_2pc = pca.transform(embedding)[:, :2]
    print(f'Energy of the first 2 PC is {pca.energy(2) * 100}%')

    # Part 2
    fig, axes = plt.subplots(3, 3, sharex='all', sharey='all')  # create 9 subplots
    fig.suptitle('Colouring of the PCA Space by the Values of 9 Measurement Areas')

    plots = []
    for i, ax in enumerate(axes.flat):  # plot the same principal component space but colour it for each measurement area
        ax.set_title(f'Measurement Area {i+1}')
        p = ax.scatter(*embedding_transformed_2pc.T, c=data[:-delay, i], cmap='viridis', alpha=0.7, s=1.)
        plots.append(p)

    vmin = min(plot.get_array().min() for plot in plots)  # get min value
    vmax = max(plot.get_array().max() for plot in plots)  # get max value
    norm = colors.Normalize(vmin=vmin, vmax=vmax)  # set up a normalizer
    for plot in plots:
        plot.set_norm(norm)  # make the colour map same for all subplots
    fig.colorbar(plots[0], ax=axes, fraction=0.1)  # put a colour bar

    fig.show()


    # Part 3
    epsilon = 1000

    # print("e: ", epsilon)

    l = 100
    v = (embedding_transformed_2pc[1:] - embedding_transformed_2pc[:-1]) / delta_t
    nonlinearapprox = NonlinearApproximator(l, epsilon)
    nonlinearapprox.fit_predict(embedding_transformed_2pc[:-1], v)

    x_hats = [embedding_transformed_2pc[0, :].reshape(1, -1)]
    arc_length = 0

    for _ in range(len(embedding_transformed_2pc)):
        v_hat = nonlinearapprox.predict(x_hats[-1])
        x_hat = v_hat * delta_t + x_hats[-1]
        x_hats.append(x_hat)
        arc_length += np.linalg.norm(v_hat) * delta_t

    x_hats = np.array(x_hats).reshape(-1, 2)
    fig = plt.figure()
    plt.plot(x_hats[:, 0], x_hats[:, 1])
    plt.show()
    plt.close(fig)
    print("True Arc Length: ", np.sum(np.linalg.norm(v, axis=1)) * delta_t)
    print("Predicted Arc length: ", arc_length)


    # Part 4
    generated_embedding = pca.inverse_transform(x_hats)
    x = generated_embedding[:, 0]  # the first measurement area, without time delay
    plt.figure()
    plt.title('Students at measurement area 1 over time')
    plt.xlabel('Time steps')
    plt.ylabel('Number of students')
    plt.plot(x)
    plt.show()
    """l = 500
    epsilon = 15

    timesteps_prediction = 28000

    v = np.abs(data[delay+1:, 0] - data[delay:-1, 0]).reshape(-1, 1) / delta_t
    nonlinearapprox = NonlinearApproximator(l, epsilon)
    nonlinearapprox.fit(embedding_area_1[:-1], v)

    x_hats = [embedding_area_1[0, :].reshape(1, -1)]

    for _ in range(timesteps_prediction):
        v_hat = nonlinearapprox.predict(x_hats[-1])
        x_hat = v_hat * delta_t + x_hats[-1][0, -1]
        x_hats.append(np.hstack([x_hats[-1][0, 1:].reshape(1, -1), x_hat]))

    x_hats = np.array(x_hats).reshape(-1, delay)

    fig = plt.figure()
    plt.plot(data[:, 0])
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x_hats[:, -1])
    plt.show()
    plt.close(fig)"""


if __name__ == "__main__":
    task1()
