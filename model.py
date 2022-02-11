import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


from typing import Union


from dilated_rnn import DRNN
from residual_lstm import ResLSTMLayer
from GRUAlpha import GRUModelAlpha
from GRUBeta import GRUModelBeta


class SeqMnist(pl.LightningModule):
    def __init__(self,
                 rnn_type: Union[nn.RNN, nn.GRU, DRNN, ResLSTMLayer, GRUModelAlpha, GRUModelBeta],
                 input_dim: int,
                 hidden_dim: int,
                 seq_length: int,
                 num_layers: int,
                 num_classes: int,
                 learning_rate: float,
                 gpus: int,
                 epsilon: float,
                 alpha: float = 1,
                 beta: float = 1
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.gpus = gpus
        self.device_ = 'cuda' if gpus > 0 else 'cpu'
        self.alpha = alpha
        self.beta = beta
        if rnn_type is GRUModelAlpha:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers, alpha)
        elif rnn_type is GRUModelBeta:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers, beta)
        else:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon

    def forward(self, x, h=None):
        if h is None:
            h = torch.randn(self.num_layers, x.size(1), self.hidden_dim).to(self.device_)
            out, _ = self.rnn(x, h)
        else:
            out, _ = self.rnn(x, h)
        out = out[-1, :, :]
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_shape = (batch, seq_length, hidden_dim) -> (seq_length, batch, hidden_dim)
        x = x.permute(1, 0, 2).contiguous()
        out = self(x)
        loss = self.criterion(out, y)
        self.log('train_step_loss', loss, prog_bar=True, on_step=True)

        # for name, param in self.named_parameters():
        #     if "alpha_2_1" in name:
        #         self.log('alpha_2_1', param, prog_bar=True, on_step=True)
        #
        #     elif 'beta_1_0' in name:
        #         self.log('beta_1_0', param, prog_bar=True, on_step=True)

        """norm = self.get_norm()
        self.log('training step loss', loss, prog_bar=True, on_step=True)
        for idx, val in norm.items():
            self.log(f"norm_{idx}", val, prog_bar=True, on_step=True)"""

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x_shape = (batch, seq_length, hidden_dim) -> (seq_length, batch, hidden_dim)
        x = x.permute(1, 0, 2).contiguous()
        output = self(x)
        prediction = (torch.argmax(output, dim=1) == y).float().mean().item()

        return prediction

    def validation_epoch_end(self, output_results):
        accuracy = 0
        for prediction in output_results:
            accuracy += prediction

        accuracy /= len(output_results)
        self.log('test_accuracy', accuracy, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(1, 0, 2).contiguous()
        output = self(x)
        prediction = (torch.argmax(output, dim=1) == y).float().mean().item()
        return prediction

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_norm(self):
        res = {}
        for idx, m in enumerate(self.parameters()):
            res[idx] = torch.norm(m)

        return res


class IMDB(pl.LightningModule):
    def __init__(self,
                 rnn_type: Union[nn.RNN, nn.GRU, DRNN, ResLSTMLayer, GRUModelAlpha, GRUModelBeta],
                 input_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 embed_dim: int,
                 seq_length: int,
                 num_layers: int,
                 num_classes: int,
                 learning_rate: float,
                 gpus: int,
                 epsilon: float,
                 alpha: float = 1,
                 beta: float = 1,
                 dropout_p: float = 0
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.gpus = gpus
        self.device_ = 'cuda' if gpus > 0 else 'cpu'
        self.alpha = alpha
        self.beta = beta
        if rnn_type is GRUModelAlpha:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers, alpha)
        elif rnn_type is GRUModelBeta:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers, beta)
        else:
            self.rnn = rnn_type(self.input_dim, self.hidden_dim, self.num_layers)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = out[-1, :, :]
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x_shape = (batch, seq_length, hidden_dim) -> (seq_length, batch, hidden_dim)
        x = x.permute(1, 0, 2).contiguous()
        out = self(x)
        loss = self.criterion(out, y)
        self.log('train_step_loss', loss, prog_bar=True, on_step=True)

        # for name, param in self.named_parameters():
        #     if "alpha_2_1" in name:
        #         self.log('alpha_2_1', param, prog_bar=True, on_step=True)
        #
        #     elif 'beta_1_0' in name:
        #         self.log('beta_1_0', param, prog_bar=True, on_step=True)

        """norm = self.get_norm()
        self.log('training step loss', loss, prog_bar=True, on_step=True)
        for idx, val in norm.items():
            self.log(f"norm_{idx}", val, prog_bar=True, on_step=True)"""

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x_shape = (batch, seq_length, hidden_dim) -> (seq_length, batch, hidden_dim)
        x = x.permute(1, 0, 2).contiguous()
        output = self(x)
        prediction = (torch.argmax(output, dim=1) == y).float().mean().item()

        return prediction

    def validation_epoch_end(self, output_results):
        accuracy = 0
        for prediction in output_results:
            accuracy += prediction

        accuracy /= len(output_results)
        self.log('test_accuracy', accuracy, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(1, 0, 2).contiguous()
        output = self(x)
        prediction = (torch.argmax(output, dim=1) == y).float().mean().item()
        return prediction

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_norm(self):
        res = {}
        for idx, m in enumerate(self.parameters()):
            res[idx] = torch.norm(m)

        return res




