import torch
import torch.nn as nn


class ConvRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ConvRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size)
        self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, stride=1, kernel_size=2)

    def forward(self, x, hidden):
        x2h = self.x2h(x)
        new_hidden = torch.stack([x2h, hidden], dim=2)
        new_hidden = self.conv(new_hidden)
        new_hidden = new_hidden.squeeze()
        return new_hidden


class ConvRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, batch_first=False):
        super(ConvRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_first = batch_first
        self.rnn = []
        for i in range(layer_dim):
            if i == 0:
                self.rnn.append(ConvRNNCell(input_dim, hidden_dim))
                continue
            self.rnn.append(ConvRNNCell(hidden_dim, hidden_dim))

        self.rnn = nn.ModuleList(self.rnn)

    def forward(self, x):
        # x shape = (seq_length, batch_size, input_dim)
        if self.batch_first:
            # x shape =  (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
            x = x.permute(1, 0, 2)

        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)
        for idx, layer in enumerate(self.rnn):
            hn = h0[idx, :, :]
            outs = []

            for seq in range(x.size(0)):
                hn = layer(x[seq, :, :], hn)
                # print(hn.shape)
                outs.append(hn)

            x = torch.stack(outs, 0)

        if self.batch_first:
            x = x.permute(1, 0, 2)

        return x, x[-1]