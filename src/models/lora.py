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
import torch.nn as nn
from src.models.base_model import BaseModel
from torch.nn import functional as F


class Lora(BaseModel):

    def __init__(
        self,
        learning_rate=1e-3,
        loss_fn=TotalBPNetLoss(),
        use_lora=True,
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        unfreeze_layer=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Load the pre-trained transformer model
        self.embedding_model = AutoModelForMaskedLM.from_pretrained(
            "gagneurlab/SpeciesLM", revision="downstream_species_lm"
        )
        self.n_tracks = 2
        
        self.unfreeze_layer = unfreeze_layer
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

        if self.unfreeze_layer:
            parms = self.embedding_model.bert.encoder.layer[-1].parameters()
            for parm in parms:
                parm.requires_grad = True
        
        # Initialize convolutional layers for prediction head
        self.conv1 = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, self.n_tracks, kernel_size=3, padding=1)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.count_linear = nn.Linear(128, self.n_tracks)

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

    def forward(self, input_ids):
        # Get the embeddings from the transformer model
        outputs = self.embedding_model(input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        embeddings = embeddings[:, 2:-1, :]  # Shape: (batch_size, seq_len, hidden_dim)
        embeddings = embeddings.transpose(
            1, 2
        )  # Shape: (batch_size, hidden_dim, seq_len)

        # Pass through convolutional layers
        x = self.norm1(F.relu(self.conv1(embeddings)))
        x = self.norm2(F.relu(self.conv2(x)))
        x = self.norm3(F.relu(self.conv3(x)))
        profile = self.conv4(x)

        # Count head
        count = self.global_avgpool(x).squeeze()  # Count head
        count = self.count_linear(count)

        return profile.float(), count.float()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
