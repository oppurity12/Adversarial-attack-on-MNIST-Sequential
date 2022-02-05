import torch
import torch.nn as nn
import math


class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(ResLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.W_p = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        # x_shape = (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        if hidden is None:
            hx = torch.zeros(batch_size, self.hidden_size).to(x.device)
            cx = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        c_t = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        r_t = torch.tanh(c_t)
        m_t = self.W_p(r_t)
        if self.input_size == self.hidden_size:
            h_t = outgate * (m_t + x)
        else:
            h_t = outgate * (m_t + self.W_h(x))

        return h_t, (h_t, c_t)


class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.cell = ResLSTMCell(input_size, hidden_size)
        self.cells = []
        for i in range(num_layers):
            if i == 0:
                self.cells.append(ResLSTMCell(input_size, hidden_size))
                continue
            self.cells.append(ResLSTMCell(hidden_size, hidden_size))

        self.cells = nn.ModuleList(self.cells)

    def forward(self, inputs, hidden=None):
        # inputs shape = (seq_length, batch, input_dim)
        for idx, layer in enumerate(self.cells):
            inputs = inputs.unbind(0)  # split by seq_length
            outputs = []
            hidden = None
            for i in range(len(inputs)):
                out, hidden = layer(inputs[i], hidden)
                outputs += [out]
            inputs = torch.stack(outputs, dim=0)  # reshape, (seq_length, batch, input_dim)

        return inputs, hidden


