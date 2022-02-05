import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import pytorch_lightning as pl

import os

input_size = 28
time_length = 28 * 28 // input_size
hidden_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(time_length, input_size))
])

train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

num_layers = 2
training_epochs = 2
lr = 0.01
criterion = nn.CrossEntropyLoss()
batch_size = 100


class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.lr = nn.Linear(self.hidden_size, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.lr(output[-1, :, :])
        return output

    def training_step(self, batch, batch_id):
        x, y = batch
        x = x.permute(1, 0, 2).contiguous()

        output = self(x)
        loss = self.criterion(output, y)
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        return optimizer

