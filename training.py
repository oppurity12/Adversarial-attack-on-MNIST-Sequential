import os

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hydra.utils import to_absolute_path

from residual_lstm import ResLSTMLayer
from dilated_rnn import DRNN
from MidPointRungeGRU import MidPointRungeGRUModel
from SSPGRU import SSRGRUModel
from customGRU import GRUModel

from GRU_V1 import GRUModelV1
from GRU_V2 import GRUModelV2

from GRUAlpha import GRUModelAlpha

# from GRU_V1_1 import

from GRU_VX1_1 import GRUModelVX1_1
from GRU_VX2 import GRUModelVX2


from loader import data_loader
from loader import MNISTDataModule
from model import SeqMnist
from attack import fg_main
from attack import pgd_main


from config import TrainConfig
import pandas as pd


def train(cfg: TrainConfig):
    rnn_dict = {'rnn': nn.RNN,
                'gru': nn.GRU,
                'lstm': nn.LSTM,
                'dilated_rnn': DRNN,
                'residual_lstm': ResLSTMLayer,
                'midpointrunge_gru': MidPointRungeGRUModel,
                'ssp_gru': SSRGRUModel,
                'custom_gru': GRUModel,
                'gru_alpha': GRUModelAlpha}

    rnn_type = rnn_dict[cfg.rnn_type.lower()]

    model = SeqMnist(rnn_type,
                     cfg.input_dim,
                     cfg.hidden_dim,
                     cfg.seq_length,
                     cfg.num_layers,
                     cfg.num_classes,
                     cfg.learning_rate,
                     cfg.gpus,
                     cfg.epsilon,
                     cfg.alpha
                     )
    train_loader, test_loader = data_loader("MNIST", cfg.batch_size,
                                            cfg.seq_length, cfg.input_dim, cfg.num_workers, cfg.is_permuted)
    if rnn_type is GRUModelAlpha:
        filename = cfg.filename if cfg.filename else f"epochs{cfg.training_epochs}_input_dim_{cfg.input_dim}_{cfg.rnn_type}_{cfg.num_layers}_layers_alpha_{cfg.alpha}"
    else:
        filename = cfg.filename if cfg.filename else f"epochs{cfg.training_epochs}_input_dim_{cfg.input_dim}_{cfg.rnn_type}_{cfg.num_layers}_layers"

    dirpath = to_absolute_path(cfg.checkpoint_dir) if cfg.checkpoint_dir else os.getcwd()
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, monitor='test_accuracy', save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger(dirpath, name=filename)
    trainer = pl.Trainer(max_epochs=cfg.training_epochs, gpus=cfg.gpus, callbacks=[checkpoint_callback])
    trainer.logger = tb_logger

    # data_module = MNISTDataModule('mnist', cfg.batch_size, cfg.seq_length, cfg.input_dim,
    #                               cfg.num_workers, cfg.is_permuted)

    trainer.fit(model, train_loader, test_loader)
    test_model = SeqMnist.load_from_checkpoint(os.path.join(dirpath, filename + '.ckpt'))
    fgsm_accuracy, test_accuracy = fg_main(test_model, test_loader, epsilon=cfg.epsilon, gpus=cfg.gpus)
    # pgd_accuracy, test_accuracy = pgd_main(test_model, test_loader, epsilon=cfg.epsilon, gpus=cfg.gpus,
    #                                        attack_steps=cfg.attack_steps)

    data_frame = pd.DataFrame({'test_accuracy': test_accuracy,'fgsm_accuracy': fgsm_accuracy,'epsilon': cfg.epsilon},
                              index=[0])
    data_frame.to_csv(os.path.join(dirpath, 'adv_results', filename + f"_fgsm_{cfg.epsilon}" ".csv"))

    # data_frame = pd.DataFrame({'test_accuracy': test_accuracy,'pgd_accuracy': fgsm_accuracy,'epsilon': cfg.epsilon},
    #                           index=[0])
    # data_frame.to_csv(os.path.join(dirpath, 'adv_results', filename + f"_pgd_{cfg.epsilon}" ".csv"))

