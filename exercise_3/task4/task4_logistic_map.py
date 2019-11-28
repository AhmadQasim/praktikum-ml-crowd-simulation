#from scipy.integrate import solve_ivp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from itertools import product


def logistic(x, r=2):
    """
    logistic map
    :param x: location
    :param r: r parameter
    :return: next step's location
    """
    return r * x * (1 - x)


def iterate(x0, r_space=(0, 4), sample_count=60):
    """
    iterates in the logistic map.
    :param x0: the starting coordinate
    :param r_space: the r parameter to iterate through
    :param sample_count: the number of coordinates to save after 120 steps
    :return: 2 iterables r values and x values of size sample_count * 10000.
    Their value order are in parallel
    """
    out = []

    for r in np.linspace(*r_space, num=10000):
        x = x0
        for i in range(120 + sample_count):
            x = logistic(x, r)
            if i >= 100:
                out.append((r, x))

    return zip(*out)


def plot(r_vals, x_vals, title='Bifurcation Diagram', x_lim=(0, 4), vertical_lines=None):
    """
    plots the bifurcation diagram
    :param r_vals: r parameter values
    :param x_vals: x coordinates
    :param title: title of the graph
    :param x_lim:
    :param vertical_lines: list of tuples of r value and colour to put vertical lines
    """
    fig, ax = plt.subplots(1, 1)

    ax.scatter(r_vals, x_vals, c='darkcyan', marker='.', s=0.1)

    if vertical_lines is not None:
        for x, c in vertical_lines:
            ax.axvline(x=x, c=c, label=f'line at r={x}', alpha=0.6)
        ax.legend(loc='upper left')

    ax.set_title(title)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_xlim(*x_lim)
    ax.set_ylim(-0.01, 1)

    fig.show()


if __name__ == '__main__':
    R, X = iterate(np.random.rand(), r_space=(0, 4))
    plot(R, X, vertical_lines=[(3.567, 'red'), (3.63, 'blue'), (3.74, 'magenta'), (3.836, 'orange')])
