import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, activation=False):
        """
        Encoder is used for the Generator and Discriminator
        :param embedding_dim: dimension of the embedding_layer
        :param hidden_dim: dimension of hidden state
        :param num_layers: number of LSTM layers
        :param activation: boolean value which determines if ReLU
        nonlinearity will be added after linear embedding layer
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.embedLayer = nn.Linear(2, embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers)

    def forward(self, input: torch):
        """
        Forward method of the encoder
        :param input: observed trajectories in the shape of
        [length of trajectory, batch, 2]
        batch equals the number of pedestrians,
        2 is because of x- and y-coordinate
        :return: h_t of LSTM layer in the shape of
        [num_layers, batch, hidden_size]
        """
        # Input is in the shape of [observed_trajectory, batch, 2]
        batch = input.size(1)
        input = input.view(-1, 2)
        # Creating embedding of input trajectory
        input = self.embedLayer(input)
        # Adding Relu activation of the embedding if desired
        if self.activation:
            input = F.relu(input)
        # Initializing the initial cell and hidden state with zeros
        h_0 = torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        c_0 = torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        inital_state = (h_0, c_0)
        # Forward through lstm
        output, state = self.lstm(input.view(-1, batch, self.embedding_dim), inital_state)
        return state[0]
