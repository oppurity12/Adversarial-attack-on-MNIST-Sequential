import hydra
from hydra.core.config_store import ConfigStore
from config import TrainConfig

from training import train

cs = ConfigStore.instance()
cs.store("config", node=TrainConfig)


@hydra.main(config_path=None, config_name="config")
def hydra_main(cfg: TrainConfig):
    train(cfg)


if __name__ == "__main__":
    hydra_main()

