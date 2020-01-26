import torch
import torch.nn as nn
import os
os.chdir("../..")
from networks.PoolingModule import PoolingModule


class Decoder(nn.Module):
    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.ffnn_dim = 1024
        self.bottleneck_dim = 1024
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        self.pool_net = PoolingModule(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )

        self.fc_0 = nn.Linear(hidden_dim + self.bottleneck_dim, self.ffnn_dim)
        self.bn_0 = nn.BatchNorm1d(self.ffnn_dim)
        self.relu_0 = nn.ReLU()
        self.fc_1 = nn.Linear(self.ffnn_dim, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.relu_1 = nn.ReLU()

        self.embedding_0 = nn.Linear(2, embedding_dim)
        self.fc_2 = nn.Linear(hidden_dim, 2)

    def forward(self, last_position, last_pos_rel, state_tuple, seq_start_end):
        batch = last_position.size(0)
        pred_traj_fake_relative = []
        decoder_input = self.embedding_0(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            relative_position = self.fc_2(output.view(-1, self.hidden_dim))
            current_position = relative_position + last_position

            hidden_decoder = state_tuple[0]
            pool_h = self.pool_net(hidden_decoder, seq_start_end, current_position)
            hidden_decoder = torch.cat([hidden_decoder.view(-1, self.hidden_dim), pool_h], dim=1)

            hidden_decoder = self.fc_0(hidden_decoder)
            hidden_decoder = self.bn_0(hidden_decoder)
            hidden_decoder = self.relu_0(hidden_decoder)
            hidden_decoder = self.fc_1(hidden_decoder)
            hidden_decoder = self.bn_1(hidden_decoder)
            hidden_decoder = self.relu_1(hidden_decoder)

            hidden_decoder = torch.unsqueeze(hidden_decoder, 0)
            state_tuple = (hidden_decoder, state_tuple[1])

            embedding_input = relative_position

            decoder_input = self.embedding_0(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_relative.append(relative_position.view(batch, -1))
            last_position = current_position

        pred_traj_fake_relative = torch.stack(pred_traj_fake_relative, dim=0)
        return pred_traj_fake_relative, state_tuple[0]