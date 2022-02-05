from dataclasses import dataclass
import os


@dataclass
class TrainConfig:
    rnn_type: str = 'gru'
    is_permuted: bool = True
    input_dim: int = 28
    hidden_dim: int = 128
    seq_length: int = 28
    num_layers: int = 1
    num_classes: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 128
    training_epochs: int = 3
    gpus: int = 1
    num_workers: int = 6
    epsilon: float = 0.05
    attack_steps: int = 3
    alpha: float = 1

    checkpoint_dir: str = "gru_test"
    filename: str = ""

