import os
import sys
import logging

# Setup root environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.preprocess import (
    get_tokens,
    load_tif_seq_data,
    save_processed_data,
    match_with_genomic_data,
)
from src.utils.utils import set_seed, setup_logger, count_parameters
from src.training.data_module import YeastDataModule
from src.models.bpnet import BPNet
from src.models.transformer_lora import LoraBPNet
from src.models.lora import Lora
from src.training.loss import TotalBPNetLoss


def preprocess(config: DictConfig):
    set_seed(config.seed)
    preprocess_data_path = os.path.join(
        config.data.data_dir, config.data.preprocessed_file
    )
    if os.path.exists(preprocess_data_path):
        print(f"Data already preprocessed at {preprocess_data_path}")
        return
    
    # Load the genomic data
    counts = torch.load(os.path.join(config.data.data_dir, config.data.counts_file))
    
    # Load the TIF-seq data
    tif_seq_path = os.path.join(config.data.data_dir, config.data.tif_seq_file)
    tif_seq_data = load_tif_seq_data(tif_seq_path)
    
    # Raw data
    raw_data_path = os.path.join(config.data.data_dir, config.data.dataset_file)
    dataset, counts = match_with_genomic_data(tif_seq_data, raw_data_path)
    
    counts_file = os.path.join(config.data.data_dir, config.data.counts_preprocessed_file)
    preprocessed_file = os.path.join(
        config.data.data_dir, config.data.preprocessed_file
    )
    save_processed_data(counts, dataset, counts_file, preprocessed_file)


@hydra.main(config_path="../configs", config_name="llm_bpnet", version_base="1.3")
def main(config):
    preprocess(config)


if __name__ == "__main__":
    main()
