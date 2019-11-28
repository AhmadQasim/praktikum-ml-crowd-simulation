from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def computePhaseDiagram(alpha, x1, x2):
    X1, X2 = np.meshgrid(x1, x2)
    # Andronov Hopf equations
    dx1 = alpha * X1 - X2 - X1 * ((X1**2+X2)**2)
    dx2 = X1 + alpha * X2 - X2 * ((X1**2+X2)**2)
    plt.figure()
    a1 = plt.subplot()
    a1.streamplot(X1, X2, dx1, dx2)
    a1.set_title('Andronov-Hopf phase diagram for alpha: '+str(a))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def f(y,z):
    return z**3 - y*z

def plot3d(f):
    y = np.linspace(-1, 1, 30)
    z = np.linspace(-1, 1, 30)
    Y, Z = np.meshgrid(y, z)
    X = f(Y, Z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X,Y,Z,50,cmap='binary')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('\u03b1\u2081')
    ax.set_ylabel('\u03b1\u2082')
    ax.set_zlabel('x')
    ax.set_title('Cusp bifurcation')
    plt.show()

def computeHopfBifurcation(x1, x2, alpha):
    dx1 = alpha * x1 - x2 - x1 * (x1**2 + x2)**2
    dx2 = x1 + alpha * x2 - x2 * (x1**2 + x2)**2
    return dx1, dx2

def computeOrbit(start, stepsize, alpha=1.0, delta = 0.1):
    s = np.transpose(np.array(start))
    orbit = np.empty(shape=(2, stepsize + 1))
    orbit[:, 0] = s
    for i in range(0, stepsize):
        x1_prev = orbit.item((0, i))
        x2_prev = orbit[1, i]
        dx1, dx2 = computeHopfBifurcation(x1_prev, x2_prev, alpha)
        # Euler's method
        x1_new = x1_prev + delta * dx1
        x2_new = x2_prev + delta * dx2
        x_new = np.transpose(np.array([x1_new, x2_new]))
        orbit[:, i+1] = x_new
    x1s = orbit[0, :]
    x2s = orbit[1, :]
    plt.figure()
    plt.plot(x1s, x2s)
    plt.title('Trajectory for Euler method with start point at: '+str(start)+'and delta t = '+str(delta))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(x1s[0], x2s[0], 'og')
    plt.plot(x1s[-1], x2s[-1], 'ob')
    plt.show()

if __name__ == "__main__":
    """
    different flag options:
    0 = Andronov Hopf phase diagram
    1 = cusp bifurcation diagram
    2 = Computation and visualization of trajectories for given start point and Euler approximation
    """
    flag = 1
    if flag == 0:
        # Representative values for alpha
        alphas = [-1.8, 0.0, 1.3]
        x1 = np.arange(-3, 3.1, 0.1)
        x2 = np.arange(-3, 3.1, 0.1)
        for a in alphas:
            computePhaseDiagram(a, x1, x2)
    elif flag == 1:
        plot3d(f)
    else:
        starts = [[2, 0], [0.5, 0]]
        for start in starts:
            start = np.transpose(np.array(start))
            computeOrbit(start=start, stepsize=10000, delta=0.001)


