import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, STL10
from torchvision.transforms import transforms

# from torchtext.legacy import data, datasets
import random

from hydra.utils import get_original_cwd, to_absolute_path
import pytorch_lightning as pl
from config import TrainConfig


def data_loader(cfg: TrainConfig):
    if cfg.data_type == 'mnist':
        return mnist_data_loader(cfg.batch_size, cfg.seq_length,
                                 cfg.input_dim, cfg.num_workers, cfg.is_permuted, cfg.seed)
    elif cfg.data_type == 'fashion_mnist':
        return fashion_mnist_data_loader(cfg.batch_size, cfg.seq_length,
                                         cfg.input_dim, cfg.num_workers, cfg.is_permuted, cfg.seed)
    elif cfg.data_type == 'stl':
        return stl_data_loader(cfg.batch_size, cfg.seq_length,
                               cfg.input_dim, cfg.num_workers, cfg.is_permuted, cfg.seed)


def mnist_data_loader(
        batch_size: int,
        seq_length: int,
        input_dim: int,
        num_workers: int,
        is_permuted: bool,
        seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    perm = torch.randperm(28 * 28) if is_permuted else torch.tensor(range(28 * 28))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(-1)[perm]),
         transforms.Lambda(lambda x: x.view(seq_length, input_dim))
         ]
    )
    train = MNIST(get_original_cwd(), download=True, train=True, transform=transform)
    test = MNIST(get_original_cwd(), download=True, train=False, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, drop_last=True, num_workers=num_workers)

    return train_loader, test_loader


def fashion_mnist_data_loader(
        batch_size: int,
        seq_length: int,
        input_dim: int,
        num_workers: int,
        is_permuted: bool,
        seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    perm = torch.randperm(28 * 28) if is_permuted else torch.tensor(range(28 * 28))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(-1)[perm]),
         transforms.Lambda(lambda x: x.view(seq_length, input_dim))
         ]
    )
    train = FashionMNIST(get_original_cwd(), download=True, train=True, transform=transform)
    test = FashionMNIST(get_original_cwd(), download=True, train=False, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, drop_last=True, num_workers=num_workers)

    return train_loader, test_loader


def stl_data_loader(
        batch_size: int,
        seq_length: int,
        input_dim: int,
        num_workers: int,
        is_permuted: bool,
        seed: int = 0):

    def stack(x):
        # x shape = (C, H, W)
        x1 = x[0, :, :]
        x2 = x[1, :, :]
        x3 = x[2, :, :]
        res = torch.cat([x1, x2, x3], dim=0)
        return res

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if input_dim != 96:
        raise ValueError('input_dim must be 32')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: stack(x))
         ]
    )
    train = STL10(get_original_cwd(), 'train', download=True, transform=transform)
    test = STL10(get_original_cwd(), 'test', download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, drop_last=True, num_workers=num_workers)

    return train_loader, test_loader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_type: str,
                 batch_size: int,
                 seq_length: int,
                 input_dim: int,
                 num_workers: int,
                 is_permuted: bool):
        super().__init__()
        self.data_type = data_type
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_workers = num_workers
        self.is_permuted = is_permuted

        self.perm = torch.randperm(28 * 28) if is_permuted else torch.tensor(range(28 * 28))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x.view(-1)[self.perm]),
             transforms.Lambda(lambda x: x.view(seq_length, input_dim))
             ]
        )

    def setup(self, stage=None):
        self.train_set = MNIST(get_original_cwd(), download=True, train=True, transform=self.transform)
        self.test_set = MNIST(get_original_cwd(), download=True, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)



