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


from GRUAlpha import GRUModelAlpha
from GRUBeta import GRUModelBeta

from GRULearningAlpha import GRULearningAlphaModel
from GRUW import GRUWModel
from GRUW2 import GRUW2Model

from SSP2GRU import SSP2GRUModel
from SSP3GRU import SSP3GRUModel

from loader import data_loader
from loader import MNISTDataModule
from model import SeqMnist, IMDB
from attack import fg_main, fg_main2
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
                'ssp2_gru': SSP2GRUModel,
                'ssp3_gru': SSP3GRUModel,
                'custom_gru': GRUModel,
                'gru_alpha': GRUModelAlpha,
                'gru_beta': GRUModelBeta,
                'gru_learning_alpha': GRULearningAlphaModel,
                'gruw': GRUWModel,
                'gruw2': GRUW2Model}

    rnn_type = rnn_dict[cfg.rnn_type.lower()]
    train_loader, test_loader = data_loader(cfg=cfg)
    model = SeqMnist(rnn_type,
                     cfg.input_dim,
                     cfg.hidden_dim,
                     cfg.seq_length,
                     cfg.num_layers,
                     cfg.num_classes,
                     cfg.learning_rate,
                     cfg.gpus,
                     cfg.epsilon,
                     cfg.alpha,
                     cfg.beta
                     )

    if rnn_type is GRUModelAlpha:
        filename = cfg.filename if cfg.filename else f"epochs{cfg.training_epochs}_input_dim_{cfg.input_dim}_{cfg.rnn_type}_{cfg.num_layers}_layers_alpha_{cfg.alpha}"
    elif rnn_type is GRUModelBeta:
        filename = cfg.filename if cfg.filename else f"epochs{cfg.training_epochs}_input_dim_{cfg.input_dim}_{cfg.rnn_type}_{cfg.num_layers}_layers_beta_{cfg.beta}"
    else:
        filename = cfg.filename if cfg.filename else f"epochs{cfg.training_epochs}_input_dim_{cfg.input_dim}_{cfg.rnn_type}_{cfg.num_layers}_layers"

    dirpath = to_absolute_path(cfg.checkpoint_dir) if cfg.checkpoint_dir else os.getcwd()
    model_save_path = os.path.join(dirpath, f"{cfg.num_layers}_layers_model_save")
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path, filename=filename, monitor='test_accuracy', save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger(dirpath, name=filename)
    trainer = pl.Trainer(max_epochs=cfg.training_epochs, gpus=cfg.gpus, callbacks=[checkpoint_callback])
    trainer.logger = tb_logger

    # data_module = MNISTDataModule('mnist', cfg.batch_size, cfg.seq_length, cfg.input_dim,
    #                               cfg.num_workers, cfg.is_permuted)

    trainer.fit(model, train_loader, test_loader)
    test_model = SeqMnist.load_from_checkpoint(os.path.join(model_save_path, filename + '.ckpt'))
    fgsm_accuracy, test_accuracy = fg_main(test_model, test_loader, epsilon=cfg.epsilon, gpus=cfg.gpus)
    fgsm_accuracy2, test_accuracy2 = fg_main2(test_model, test_loader, epsilon=cfg.epsilon, gpus=cfg.gpus,
                                              num_layers=cfg.num_layers, batch_size=cfg.batch_size, hidden_dim=cfg.hidden_dim)
    # pgd_accuracy, test_accuracy = pgd_main(test_model, test_loader, epsilon=cfg.epsilon, gpus=cfg.gpus,
    #                                        attack_steps=cfg.attack_steps)
    result_path = os.path.join(dirpath, f"{cfg.num_layers}_layers_results_save")
    os.makedirs(result_path, exist_ok=True)
    data_frame = pd.DataFrame({"filename": f"{filename}",'test_accuracy': test_accuracy,'fgsm_accuracy': fgsm_accuracy,'epsilon': cfg.epsilon},
                              index=[0])
    data_frame.to_csv(os.path.join(result_path, filename + f"_fgsm_{cfg.epsilon}" ".csv"))

    # data_frame = pd.DataFrame({'test_accuracy': test_accuracy,'pgd_accuracy': fgsm_accuracy,'epsilon': cfg.epsilon},
    #                           index=[0])
    # data_frame.to_csv(os.path.join(dirpath, 'adv_results', filename + f"_pgd_{cfg.epsilon}" ".csv"))

