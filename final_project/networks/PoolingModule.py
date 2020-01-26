import torch
import torch.nn as nn
import os
os.chdir("../..")


class PoolingModule(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        hidden_dim=64
    ):
        super(PoolingModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = 1024
        self.embedding_dim = embedding_dim

        self.fc_0 = nn.Linear(self.embedding_dim + self.hidden_dim, 512)
        self.bn_0 = nn.BatchNorm1d(512)
        self.relu_0 = nn.ReLU()
        self.fc_1 = nn.Linear(512, self.bottleneck_dim)
        self.bn_1 = nn.BatchNorm1d(self.bottleneck_dim)
        self.relu_1 = nn.ReLU()

        self.embedding_0 = nn.Linear(2, embedding_dim)

    def forward(self, h_states, seq_start_end, end_pos):
        hidden_pool_all = []
        for (start, end) in seq_start_end:
            start = start.item()
            end = end.item()

            num_ped = end - start
            hidden_state = h_states.view(-1, self.hidden_dim)[start:end]
            end_positions = end_pos[start:end]

            hidden_interleaved = hidden_state.repeat(num_ped, 1)
            end_interleaved = end_positions.repeat(num_ped, 1)
            end_repeated = end_positions.unsqueeze(dim=1).repeat(1, num_ped, 1)
            end_repeated = end_repeated.view(-1, end_positions.size(1))
            relative_positions = end_interleaved - end_repeated
            relative_embeddings = self.embedding_0(relative_positions)
            ffnn_h_input = torch.cat([relative_embeddings, hidden_interleaved], dim=1)

            hidden_pool = self.fc_0(ffnn_h_input)
            hidden_pool = self.bn_0(hidden_pool)
            hidden_pool = self.relu_0(hidden_pool)
            hidden_pool = self.fc_1(hidden_pool)
            hidden_pool = self.bn_1(hidden_pool)
            hidden_pool = self.relu_1(hidden_pool)

            hidden_pool = hidden_pool.view(num_ped, num_ped, -1).max(1)[0]
            hidden_pool_all.append(hidden_pool)
        hidden_pool_all = torch.cat(hidden_pool_all, dim=0)
        return hidden_pool_all
