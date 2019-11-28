import matplotlib.pyplot as plt
import numpy as np
import fire
from scipy.integrate import solve_ivp
from functools import partial


def func1(t, x, a):
    return a - np.square(x)


def func2(t, x, a):
    return a - 2*np.square(x) - 2


def func1_stable(a):
    return np.sqrt(a)


def func1_unstable(a):
    return -np.sqrt(a)


def bifurcation_diagram(func):
    a_lower = -4
    a_upper = 4
    a_series = np.linspace(a_lower, a_upper, 10000)

    a = []
    x = []
    for alpha in a_series:
        x_curr = np.random.uniform(-2, 2, 1)
        if func == 1:
            func_ = partial(func1, a=alpha)
            x_curr = solve_ivp(func_, (0, 100), x_curr, t_eval=[0, 100]).y[:, -1].reshape(-1)
        else:
            func_ = partial(func2, a=alpha)
            x_curr = solve_ivp(func_, (0, 100), x_curr, t_eval=[0, 100]).y[:, -1].reshape(-1)

        a.append(np.full_like(x_curr, fill_value=alpha))
        x.append(x_curr)

    if func == 1:
        x_stable = list(map(func1_stable, a_series))
        x_unstable = list(map(func1_unstable, a_series))
        plt.plot(a_series, x_stable, label="Stable")
        plt.plot(a_series, x_unstable, label="Unstable")

    for a_, x_ in zip(a, x):
        plt.plot(a_, x_, ls='', marker=',')
    plt.title("Bifurcation Diagram")
    plt.legend()
    plt.xlabel('α')
    plt.ylabel('x')
    plt.xlim(a_lower, a_upper)
    plt.show()


if __name__ == "__main__":
    fire.Fire()
