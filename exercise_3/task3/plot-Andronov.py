from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

"""
What is cusp bifurcation?

"""



def compute(alpha, x1, x2):
    X1, X2 = np.meshgrid(x1, x2)
    dx1 = alpha * X1 - X2 - X1 * ((X1**2+X2)**2)
    dx2 = X1 + alpha * X2 - X2 * ((X1**2+X2)**2)
    plt.figure()
    a1 = plt.subplot()
    a1.streamplot(X1, X2, dx1, dx2)
    a1.set_title('Phase Diagram for alpha: '+str(a))
    plt.show()

#  Generate 3d plot for given function fn,
def plot_cusp(fn,bbox=(-1, 1)):
    xmin, xmax, ymin, ymax,zmin, zmax = bbox *3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100)
    B = np.linspace(xmin, xmax, 15)
    A1, A2 = np.meshgrid(A, A)
    for z in B:
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')

    for y in B:
        X, Z = A1, A2
        Y = fn(X, y, Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B:
        Y, Z = A1, A2
        X = fn(x, Y, Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_xlabel('alpha1')
    ax.set_ylabel('alpha2')
    ax.set_zlabel('x')
    ax.set_title('cusp bifurcation')
    plt.show()

# Defining the function which shall be zero and plotted
def computeFunction(x,y,z):
    return x + y*z - z**3

def f(y,z):
    return z**3 - y*z

def plot3d(f):
    y = np.linspace(-1,1,30)
    z = np.linspace(-1,1,30)
    Y,Z = np.meshgrid(y,z)
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
    2 = cusp bifurcation diagram
    3 = Computation and visualization of trajectories for given start point and Euler approximation
    """
    flag = 2
    if flag == 0:
        # Representative values for alpha
        alphas = [-1.8, 0.0, 1.3]
        x1 = np.arange(-1, 1.1, 0.1)
        x2 = np.arange(-1, 1.1, 0.1)
        for a in alphas:
            compute(a, x1, x2)
    elif flag == 1:
        plot_cusp(computeFunction)
    elif flag == 2:
        plot3d(f)
    else:
        starts = [[2, 0], [0.5, 0]]
        for start in starts:
            start = np.transpose(np.array(start))
            computeOrbit(start=start, stepsize=10000, delta=0.001)


