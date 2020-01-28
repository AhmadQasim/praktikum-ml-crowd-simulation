import torch.nn as nn
import os
os.chdir("../..")
from networks.Encoder import Encoder
from networks.PoolingModule import PoolingModule


class Discriminator(nn.Module):

    def __init__(self, in_dim, latent_dim, activation_encoder=False):
        """

        :param in_dim: input dimension
        :param latent_dim: dimension of the embedding
        :param activation_encoder: whether the encoder should have activation or not
        """
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_dim, latent_dim, activation=activation_encoder)

        self.fake_real_discriminator = nn.Sequential(nn.Linear(latent_dim, 1024),
                                                     nn.BatchNorm1d(1024),
                                                     nn.ReLU(),
                                                     nn.Linear(1024, 1),
                                                     nn.BatchNorm1d(1),
                                                     nn.ReLU())

    def forward(self, trajectory, relative_trajectory, seq_start_end):
        """

        :param trajectory: trajectory data
        :param relative_trajectory: relative trajectory data, i.e. changes of coordinates
        :param seq_start_end: start and end indices for each sequence in trajectory data
        :return: return the classification scores of trajectories
        """
        encoded_relative_trajectory = self.encoder(relative_trajectory)

        scores = self.fake_real_discriminator(encoded_relative_trajectory.squeeze())
        return scores
