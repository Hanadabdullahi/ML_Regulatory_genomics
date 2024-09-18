import os
import sys

import yaml

# Setup root environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# import logging
# from scripts.train_transformer import train
from scripts.train_model import train


os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "logs")

def update_cfg_with_wandb(cfg: DictConfig, wandb_config: dict):
    for key, value in wandb_config.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return cfg

@hydra.main(config_path="../configs", config_name="llm_bpnet", version_base="1.3")
def main(config):

    # load sweep config from sweep.yaml
    with open("./configs/sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)
    # sweep_id = wandb.sweep(sweep_config, project="regulate-me")
    sweep_id = "lj8tunid"
    print(f"Sweep ID: {sweep_id}")

    # Define the train function to accept the config parameter
    def wrapped_train():
        print(config)
        wandb.init(
            dir = os.getcwd() + "/logs",
        )
        print("Config")
        print(wandb.config)
        run_config = update_cfg_with_wandb(config, wandb.config)
        train(run_config)

    wandb.agent(sweep_id,project=config.wandb.project, entity=config.wandb.entity, function=wrapped_train, count = 100)

if __name__ == "__main__":
    main()
