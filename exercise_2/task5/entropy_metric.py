import numpy as np
import matplotlib.pyplot as plt
from utils import predict, parse_trajectory
import datetime

random_seed = 1  # used throughout the example
np.random.seed(random_seed)


class EntropyMetric:
    def __init__(self, pedestrian_num, model_error, source, target):
        # Initial state of the true and simulated data
        x0 = np.array([1, 0]).reshape(1, -1)
        self.x0 = x0 / np.linalg.norm(x0)
        # Number of time steps for the given simulation
        self.NT = 30
        # Physical time that passes in the given number of time steps
        self.T = self.NT
        self.dt = self.T / self.NT
        self.time = np.linspace(0, self.T, self.NT)

        # this 'parameter' essentially defines how well the model approximates the true dynamics for this example.
        # Values below 0.05 yield essentially no visible difference to the truth, while values above 0.1
        # make the trajectories almost indistinguishable from noise, and only ensemble runs make it possible to evaluate
        # the error and estimate the true state.
        self.model_error = model_error
        self.true_error = 1e-4
        self.m = 10  # ensemble runs
        self.n = pedestrian_num  # number of agents. Note that the models f_true and f_model only work
        self.d = 4  # dimension per "agent" (we only have one here) self.nd = n*d

        self.nd = self.n * self.d  # total state dimension
        self.xt = np.zeros((self.NT, (self.nd * self.m)))
        # this is the initial guess for the entropy matrix. can be pretty arbitrary
        self.M = np.identity(self.nd)
        # this is the guess for the true error in the observations. should be small here
        self.Q = np.identity(self.nd) * self.true_error ** 2
        self.N_ITER = 5  # number of iterations of algorithm1_enks and max_likelihood
        self.Mhat = self.M
        self.zk = None
        self.xm_hat = None
        self.xm_hat_prev = 0
        self.xm = None

        self.source = source
        self.target = target

        # paths
        self.vadere_root = '/home/ahmad/praktikum/vadere/'
        self.path_scenario = f'./bottleneck/scenarios/bottleneck_{self.target}.json'
        self.output_path = f'./bottleneck/output/{self.target.upper()}/prediction/'
        self.source_trajectory = f"./bottleneck/output/{self.source.upper()}" \
            f"/model/bottleneck_{self.source.upper()}_{self.n}"
        self.dynamic_scenario_path = self.path_scenario[:-5] + '_edited.json'

        self.true_data = parse_trajectory(path=self.source_trajectory) + np.random.randn(self.n,
                                                                                         self.d) * self.true_error

    def f_model(self, x, error):
        """
        :param x: the true data for pedestrians at a timestep
        :param error: the standard deviation of the gaussian noise to be added to the model predictions
        """
        """
            For the given true trajectories for the pedestrians over a timestep. Predict the trajectories according to
            the prediction model
        """
        return predict(x, self.path_scenario, self.output_path, self.dynamic_scenario_path, self.vadere_root)\
               + np.random.randn(x.shape[0], x.shape[1]) * error

    def initial_run(self):
        """
        """
        """
            A function which populates the xt array with true data and then calculates xm without applying the EM
            algorithm. The resulting predicted trajectories are then plotted. 
        """
        self.xt[0, :] = np.column_stack([self.true_data[0, :].reshape(1, -1) for _ in range(self.m)])
        self.xm = self.xt.copy()
        for k in range(1, self.NT):
            print("Timestep: ", k)
            for i in range(self.m):
                self.xt[k, (i * self.nd):((i + 1) * self.nd)] = self.true_data[k, :].reshape(1, -1)
                self.xm[k, (i * self.nd):((i + 1) * self.nd)] = \
                    (self.f_model(self.xm[k - 1, (i * self.nd):((i + 1) * self.nd)].reshape(self.n, self.d),
                                  self.model_error).reshape(1, -1))

    def plot_initial_run(self):
        """
        """
        """
            A function for plotting the initial run data i.e. xt and xm
        """
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='all')
        ax[0].scatter(self.xt[:, 0], self.xt[:, 1], s=1, label='true state')

        ax[0].scatter(self.xm[:, 0], self.xm[:, 1], s=1, label='model state')
        ax[0].set_xlabel('space 1')
        ax[0].set_ylabel('space 2')
        ax[0].set_title('trajectories in space')
        ax[1].plot(self.time, self.xt)
        ax[1].plot(self.time, self.xm)
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('space 1,2')
        ax[1].set_title('trajectories in time')
        plt.show()

    @staticmethod
    def normal_draw(cov):
        """
        :param cov: the per-agent covariance matrix M
        """
        """
            draw an n-dimensional point from a Gaussian distribution with given covariance.
        """
        return np.random.multivariate_normal(0 * cov[:, 0], cov, size=1)

    @staticmethod
    def observation(x):
        """
        :param x: the true data zk
        """
        """
            the observation function h is an identity function
        """
        return x

    def algorithm1_enks(self, z_data, error_covariance_m, error_covariance_q, observation, fhat_model):
        """
        :param z_data: the true data of the pedestrian trajectories
        :param error_covariance_m: the per agent covariance matrix M
        :param error_covariance_q: the true model error covariance matrix Q
        :param observation: the observation function h
        :param fhat_model: the model predictions function f_model
        """
        """
            the enks computation function
        """
        t = z_data.shape[0]
        m_l = error_covariance_m
        q_l = error_covariance_q

        # initialize the initial guess for the model state with random numbers
        # normally, one would probably have a better guess
        xk = np.random.randn(z_data.shape[0], z_data.shape[1]) / 10000
        xk[0, :] = np.column_stack([self.true_data[0, :].reshape(1, -1) for _ in range(self.m)])
        for k in range(1, t):
            print("Timestep: ", k)
            zk = np.zeros((z_data.shape[1],))
            for i in range(self.m):
                mkm1 = self.normal_draw(m_l)
                xk[k, (i * self.nd):((i + 1) * self.nd)] = \
                    fhat_model(xk[k - 1, (i * self.nd):((i + 1) * self.nd)].reshape(self.n, self.d)).reshape(1, -1)\
                    + mkm1
                qk = self.normal_draw(q_l)
                zk[(i * self.nd):((i + 1) * self.nd)] = \
                    observation(xk[k, (i * self.nd):((i + 1) * self.nd)].reshape(1, -1)) + qk
            zkhat = 1 / self.m * (np.sum([zk[i::self.nd] for i in range(self.nd)], axis=1))
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

    def max_likelihood_multi(self, xk, fhat_model):
        """
        :param xk: the model state which is equal to zk as observation function is identity
        :param fhat_model: the model predictions function f_model
        """
        """
            the maximum likelihood computation function
        """
        t = xk.shape[0]
        m = 0
        for k in range(0, t - 1):
            for i in range(self.m):
                fhat = fhat_model(xk[k, (i * self.nd):((i + 1) * self.nd)].reshape(self.n, self.d)).reshape(1, -1)
                xhat = xk[k + 1, (i * self.nd):((i + 1) * self.nd)]
                mul = xhat - fhat
                for j in range(self.n):
                    m += mul[0, (j*self.d):(j*self.d) + self.d] @ mul[0, (j*self.d):(j*self.d) + self.d].T

        return np.array([[m / (t * self.m * self.n)]])

    def entropy(self, m_dist):
        """
        :param m_dist: the per-agent covariance matrix M
        """
        """
            the entropy computation function
        """
        return (1 / 2) * (self.n * np.log((2 * np.pi * np.exp(1)) ** self.d * np.linalg.det(m_dist)))

    def run_em(self):
        """
        """
        """
            iterate the EM algorithm through different steps
        """
        self.zk = self.observation(self.xt[1:, :])
        for k in range(self.N_ITER):
            self.xm_hat = self.algorithm1_enks(self.zk, self.Mhat, self.Q, self.observation,
                                               lambda x: self.f_model(x, self.model_error))
            self.Mhat = self.max_likelihood_multi(self.xm_hat, lambda x: self.f_model(x, self.model_error))
            print('current det(M)', np.linalg.det(self.Mhat))
            print('error change ', np.linalg.norm(self.xm_hat - self.xm_hat_prev))
            self.xm_hat_prev = self.xm_hat

    def plot_em(self):
        """
        """
        """
            plot the result of the EM algorithm i.e. xt and xm_hat
        """
        fig, ax = plt.subplots(1, 2, figsize=(16, 4), sharey='all')
        for i in range(self.nd):
            ax[0].plot(self.time, self.xt[:, i], label='true state {}'.format(i))
        ax[0].set_title('true states/observations')

        # this second plot shows the estimated MODEL (!) state, but since the model here is just the truth plus noise,
        # the estimated model state should be the true state.
        # We also skip the first state in xm_hat, since it was chosen at random.
        for i in range(self.nd):
            ax[1].plot(self.time[2:], self.xm_hat[1:, i], '-', label='estimated model state {}'.format(i))
        ax[1].set_title('estimated states from observations and model')
        plt.show()

    def find_entropy(self):
        self.initial_run()
        self.run_em()
        self.plot_em()

        with open('./results', 'a') as fp:
            print('entropy(M estimated) ', self.entropy(self.Mhat))
            fp.writelines('***************************************\n')
            fp.writelines('Time: {}\n'.format(datetime.datetime.now()))
            fp.writelines('Pedestrians Count: {}\n'.format(self.n))
            fp.writelines('Dimensions Count: {}\n'.format(self.d))
            fp.writelines('Ensembles Count: {}\n'.format(self.m))
            fp.writelines('Timesteps Count: {}\n'.format(self.NT))
            fp.writelines('Model Error: {}\n'.format(self.model_error))
            fp.writelines('True Error: {}\n'.format(self.true_error))
            fp.writelines('entropy(M estimated) {}\n'.format(self.entropy(self.Mhat)))
            fp.writelines('\n\n\n')

        return self.entropy(self.Mhat)


if __name__ == "__main__":
    # the number of pedestrians to run the algorithm for
    pedestrians = [15, 20, 25, 30]
    # the different model errors
    model_errors = [0.1]
    entropies = []

    for pedestrian in pedestrians:
        for error in model_errors:
            entropy_metric = EntropyMetric(pedestrian, error, source='osm', target='gnm')
            entropies.append(entropy_metric.find_entropy())

    plt.plot(model_errors, entropies)
    plt.show()
