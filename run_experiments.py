import hydra
import omegaconf
from experiments import translation


@hydra.main(config_name="configs/config")
def main(config: omegaconf.DictConfig):
    if config.experiment_type == "translation":
        translation.main(config=config.translation)


if __name__ == "__main__":
    main()
