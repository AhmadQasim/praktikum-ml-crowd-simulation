import numpy as np
import fire
import matplotlib.pyplot as plt

random_seed = 1  # used throughout the example
np.random.seed(random_seed)


class EntropyMetric:
    def __init__(self):
        # Initial state of the true and simulated data
        x0 = np.array([1, 0]).reshape(1, -1)
        self.x0 = x0 / np.linalg.norm(x0)
        # Number of time steps for the given simulation
        self.NT = 100
        # Physical time that passes in the given number of time steps
        self.T = 5
        self.dt = self.T / self.NT
        self.time = np.linspace(0, self.T, self.NT)

        # this 'parameter' essentially defines how well the model approximates the true dynamics for this example.
        # Values below 0.05 yield essentially no visible difference to the truth, while values above 0.1
        # make the trajectories almost indistinguishable from noise, and only ensemble runs make it possible to evaluate
        # the error and estimate the true state.
        self.model_error = 1e-3
        # this is the error in the true model, and also in the observations. You do not need to change this.
        self.true_error = 1e-4
        self.m = 10  # ensemble runs
        self.n = 1  # number of agents. Note that the models f_true and f_model only work
        self.d = 2  # dimension per "agent" (we only have one here) self.nd = n*d # total state dimension

        self.nd = self.n * self.d  # total state dimension
        self.xt = np.zeros((self.NT, (self.nd * self.m)))
        # this is the initial guess for the entropy matrix. can be pretty arbitrary
        self.M = np.identity(self.n)
        # this is the guess for the true error in the observations. should be small here
        self.Q = np.identity(self.n) * self.true_error ** 2
        self.N_ITER = 5  # number of iterations of algorithm1_enks and max_likelihood
        self.Mhat = self.M
        self.zk = self.observation(self.xt[1:, :])
        self.xm_hat = 0
        self.xm_hat_prev = 0
        self.xm = None

    @staticmethod
    def f_true(x, data_t):
        xc0 = x[:, 0] + 1j * x[:, 1]
        angle = np.angle(xc0)
        xc1 = (1 + np.sin(5 * angle) / 3) * np.exp(1j * (angle + data_t))
        return np.column_stack([np.real(xc1), np.imag(xc1)])

    def f_model(self, x, data_t, m_error):
        return self.f_true(x, data_t) + np.random.randn(x.shape[0], x.shape[1]) * m_error

    @staticmethod
    def normal_draw(cov):
        """draw an n-dimensional point from a Gaussian distribution with givencovariance."""
        return np.random.multivariate_normal(0 * cov[:, 0], cov, 1)

    @staticmethod
    def observation(x):
        # relatively simple observation function z=h(x), also no change in dimension
        return -x / 2 + np.cos(x) / 3

    # compute ensemble Kalman smoothing
    def algorithm1_enks(self, z_data, error_covariance_m, error_covariance_q, observation, fhat_model):
        t = z_data.shape[0]
        m_l = error_covariance_m
        q_l = error_covariance_q

        # initialize the initial guess for the model state with random numbers
        # normally, one would probably have a better guess
        xk = np.random.randn(z_data.shape[0], z_data.shape[1]) / 10000
        for k in range(1, t):
            zk = np.zeros((z_data.shape[1],))
            for i in range(self.m):
                mkm1 = self.normal_draw(m_l)
                xk[k, (i * self.nd):((i + 1) * self.nd)] = \
                    fhat_model(xk[k - 1, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1)) + mkm1
                qk = self.normal_draw(q_l)
                zk[(i * self.nd):((i + 1) * self.nd)] = \
                    observation(xk[k, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1)) + qk
            zkhat = 1 / self.m * np.sum([zk[i::self.nd] for i in range(self.nd)], axis=1)
            zdiff = np.row_stack([(zk[(i * self.nd):((i + 1) * self.nd)] - zkhat) for i in range(self.m)])
            z_k = np.cov(zdiff.T)

            for j in range(1, k + 1):
                xjbar = np.array(1 / self.m * np.sum([xk[j, i::self.nd] for i in range(self.nd)], axis=1))
                xdiff = np.row_stack([(xk[j, (i * self.nd):((i + 1) * self.nd)] - xjbar) for i in range(self.m)])
                zdiff = np.row_stack([(zk[(i * self.nd):((i + 1) * self.nd)] - zkhat) for i in range(self.m)])
                sigmaj = 1 / (self.m - 1) * (xdiff.T @ zdiff)
                matk = sigmaj @ np.linalg.pinv(z_k, rcond=1e-10)
                for i in range(self.m):
                    xk[j, (i * self.nd):((i + 1) * self.nd)] = xk[j, (i * self.nd):((i + 1) * self.nd)] + matk @ (
                            z_data[k, (i * self.nd):((i + 1) * self.nd)] - zk[(i * self.nd):((i + 1) * self.nd)])
        return xk

    def max_likelihood(self, xk, fhat_model):
        t = xk.shape[0]
        data = []
        for k in range(1, t - 1):
            for i in range(self.m):
                fhat = fhat_model(xk[k, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1))
                xhat = xk[k + 1, (i * self.nd):((i + 1) * self.nd)]
                data.append((xhat - fhat))
        data = np.row_stack(data)
        # note that we do not compute it "per agent", as in the paper guy-2019b,
        # but for all coordinates of the state (we only consider one "agent" in this code)
        return np.cov(data.T)

    def entropy(self, m_dist):
        return 1 / 2 * self.n * np.log((2 * np.pi * np.exp(1)) ** self.d * np.linalg.det(m_dist))

    def initial_run(self):
        self.xt[0, :] = np.column_stack([self.x0 + np.random.randn(1) / 1000 for _ in range(self.m * self.n)])
        self.xm = self.xt.copy()
        for k in range(1, self.NT):
            for i in range(self.m):
                # using this as "true dynamics" means there is no noise in the true states(which is also ok)
                # xt[k,(i*n):((i+1)*n)]=(f_true(xt[k-1,(i*n):((i+1)*n)].reshape(1,-1), dt))
                # test what happens if the "true dynamics" are just a less noisy version of the model
                self.xt[k, (i * self.nd):((i + 1) * self.nd)] = (self.f_model(
                    self.xt[k - 1, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1), self.dt, self.true_error))
                self.xm[k, (i * self.nd):((i + 1) * self.nd)] = (self.f_model(
                    self.xm[k - 1, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1), self.dt, self.model_error))

    def plot_initial_run(self):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='all')
        ax[0].scatter(self.xt[:, 0], self.xt[:, 1], s=1, label='true state')

        ax[0].scatter(self.xm[:, 0], self.xm[:, 1], s=1, label='model state')
        ax[0].set_xlabel('space 1')
        ax[0].set_ylabel('space 2')
        ax[0].set_title('trajectories in space')
        ax[0].legend()
        ax[1].plot(self.time, self.xt)
        ax[1].plot(self.time, self.xm)
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('space 1,2')
        ax[1].set_title('trajectories in time')
        plt.show()

    def run_em(self):
        for k in range(self.N_ITER):
            self.xm_hat = self.algorithm1_enks(self.zk, self.Mhat, self.Q, self.observation,
                                               lambda x: self.f_model(x, self.dt, self.model_error))
            self.Mhat = self.max_likelihood(self.xm_hat, lambda x: self.f_model(x, self.dt, self.model_error))
            print('current det(M)', np.linalg.det(self.Mhat))
            print('error change ', np.linalg.norm(self.xm_hat - self.xm_hat_prev))
            self.xm_hat_prev = self.xm_hat

    def plot_em(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 4), sharey='all')
        ax[0].plot(self.time, self.xt[:, 0], label='true state 1')
        ax[0].plot(self.time, self.xt[:, 1], label='true state 2')
        ax[0].plot(self.time[1:], self.zk[:, 0], 'g:', label='observation 1')
        ax[0].plot(self.time[1:], self.zk[:, 1], 'g.', label='observation 2')
        ax[0].set_title('true states and observations')
        ax[0].legend()

        # this second plot shows the estimated MODEL (!) state, but since the model here is just the truth plus noise,
        # the estimated model state should be the true state.
        # We also skip the first state in xm_hat, since it was chosen at random.
        ax[1].plot(self.time[2:], self.xm_hat[1:, 0], '-', label='estimated model state 1')
        ax[1].plot(self.time[2:], self.xm_hat[1:, 1], '-', label='estimated model state 2')
        ax[1].set_title('estimated states from observations and model')
        ax[1].legend()
        plt.show()

    def find_entropy(self):
        self.initial_run()
        self.plot_initial_run()
        self.run_em()
        self.plot_em()


if __name__ == "__main__":
    fire.Fire(EntropyMetric)
