import matplotlib.pyplot as plt
import numpy as np


def visualizePhase(alpha, x1, x2):
    X1, X2 = np.meshgrid(x1, x2)
    dx1 = alpha * X1 + alpha *X2
    dx2 = -0.25 * X1
    plt.figure()
    a1 = plt.subplot()
    a1.streamplot(X1, X2, dx1, dx2)
    a1.set_title('Phase Diagram for alpha: '+str(a))
    plt.show()

#alphas = [0.1, 0.5, 2.0, 10.0]
# alpha -0,1 leads to a positive and a negative eigenvalue (saddle)
# alpha 0.5 leads to positive eigenvalue with positive and negative imaginary part (focus)
# alpha 2 leads to two positive eigenvalues with no imaginary part (node)
# All hyperbolic equilibrium are unstable
# The unstable focus and the unstable node are topologically equivalent
alphas = [-0.1, 0.5, 2]
x1 = np.arange(-1, 1.1, 0.1)
x2 = np.arange(-1, 1.1, 0.1)
for a in alphas:
    visualizePhase(a, x1, x2)




