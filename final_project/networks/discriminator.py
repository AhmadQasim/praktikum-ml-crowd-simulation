import torch.nn as nn
from networks.Encoder import Encoder
from networks.PoolingModule import PoolingModule


class Discriminator(nn.Module):

    def __init__(self, in_dim, latent_dim, activation_encoder=None):
        """

        :param in_dim:
        :param latent_dim:
        :param activation_encoder:
        """
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_dim, latent_dim, activation=activation_encoder)

        self.fake_real_discriminator = nn.Sequential(nn.Linear(latent_dim, 1),
                                                     nn.BatchNorm1d(1),
                                                     nn.ReLU())

        self.pool_net = PoolingModule(in_dim, latent_dim)

    def forward(self, trajectory, relative_trajectory, seq_start_end):
        """

        :param trajectory:
        :param relative_trajectory:
        :param seq_start_end:
        :return:
        """
        encoded_relative_trajectory = self.encoder(relative_trajectory)

        pooled = self.pool_net(encoded_relative_trajectory.squeeze(), seq_start_end, trajectory[0])

        scores = self.fake_real_discriminator(pooled)
        return scores
