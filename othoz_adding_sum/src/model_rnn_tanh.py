import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, layer_dim=1, output_dim=1, nonlinearity='tanh'):
        super(RNNModel, self).__init__()
        # Input dimensions
        self.input_dim = input_dim

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        #Non linearity value. If it TANH or RELU
        self.nonlinearity = nonlinearity

        # Output dimension
        self.output_dim = output_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim,
                          batch_first=False, nonlinearity=self.nonlinearity)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x)

        # Index hidden state of last time step
        # out.size() --> 100, sequence length, 100
        # out[:, -1, :] --> 100, 100 --> We want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
