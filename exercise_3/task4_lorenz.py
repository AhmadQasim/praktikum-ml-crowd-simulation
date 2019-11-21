from scipy.integrate import solve_ivp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz_attractor(ts, point, alpha=10, beta=8/3, rho=28):
    x, y, z = point

    dx = alpha * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return dx, dy, dz


def plot(a, b=None, c='indianred', title='', lw=0.2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*a, c=c, lw=lw)
    if b is not None:
        ax.plot(*b, c='darkcyan', lw=lw, alpha=0.75)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    dt = 0.01

    params = {
        't_span': [0, 1000],
        't_eval': np.linspace(0, 1000, num=int(1000/dt)),
        'fun': lorenz_attractor
    }

    mode = 1

    if mode == 0:
        print("1", dt)
        sol1 = solve_ivp(**params, y0=[10, 10, 10]).y
        print('2')
        sol2 = solve_ivp(**params, y0=[10+10**-8, 10, 10]).y

        print('3')
        diff = np.sqrt(np.sum(np.square(sol1.T - sol2.T), axis=1))
        print(np.where(diff[::int(1/dt)] <= 1.0)[0])

        #plot(sol1, title='Trajectory from (10, 10, 10)')
        #plot(sol2, c='darkcyan', title='Trajectory from(10+10\u207b\u2078, 10, 10)')
        #plot(sol1, sol2, title='2 trajectories with slightly different starting points')

    elif mode == 1:
        params['fun'] = partial(lorenz_attractor, rho=0.5)
        print("1", dt)
        sol1 = solve_ivp(**params, y0=[10, 10, 10]).y
        print('2')
        sol2 = solve_ivp(**params, y0=[10 + 10 ** -8, 10, 10]).y
        diff = np.sqrt(np.sum(np.square(sol1.T - sol2.T), axis=1))
        print(np.where(diff[::int(1 / dt)] <= 1.0)[0])

        plot(sol1, sol2, lw=1, title='2 trajectories with slightly different starting points (\u03c1=0.5)')
