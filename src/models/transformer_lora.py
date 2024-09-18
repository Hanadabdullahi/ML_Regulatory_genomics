import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from src.utils.metrics import compute_metrics
from src.utils.preprocess import (
    load_tif_seq_data,
    match_with_genomic_data,
    save_processed_data,
    preprocess_data,
    get_tokens,
)
from src.training.data_module import YeastDataModule
from src.models.bpnet import BPNet
from src.utils.dataset import one_hot_encode_sequence as one_hot, split_by_chrom
from src.training.loss import TotalBPNetLoss
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from src.models.base_model import BaseModel


class LoraBPNet(BaseModel):

    def __init__(
        self,
        bpnet_params,
        learning_rate=1e-3,
        loss_fn=TotalBPNetLoss(),
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        use_lora=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load the pre-trained transformer model
        self.embedding_model = AutoModelForMaskedLM.from_pretrained(
            "gagneurlab/SpeciesLM", revision="downstream_species_lm"
        )

        self.use_lora = use_lora
        if self.use_lora:
            # Initialize PEFT
            self.peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="all",
            )
            self.embedding_model = get_peft_model(
                self.embedding_model, self.peft_config
            )
        else:
            self.peft_config = None
            # Freeze the transformer model
            for param in self.embedding_model.parameters():
                param.requires_grad = False

        # Initialize BPNet with provided parameters
        self.n_channels = 768
        bpnet_params["n_channels"] = self.n_channels
        self.bpnet = BPNet(**bpnet_params)
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        # Initialize lists to store outputs and labels
        self.validation_outputs = []
        self.validation_labels = []

    def forward(self, input_ids):
        # Get the embeddings from the transformer model
        outputs = self.embedding_model(input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        # Get rid of the tokens
        embeddings = embeddings[:, 2:-1, :]
        embeddings = embeddings.transpose(
            1, 2
        )  # Shape: (batch_size, hidden_dim, seq_len)

        # Pass the embeddings to BPNet
        profile, count = self.bpnet(embeddings)
        return profile, count

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
