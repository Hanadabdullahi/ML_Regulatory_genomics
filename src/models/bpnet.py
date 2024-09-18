import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import wandb
from ..training.loss import TotalBPNetLoss
from ..utils.metrics import compute_metrics
from .base_model import BaseModel


class BPNet(BaseModel):

    def __init__(
        self,
        n_layers,
        n_tracks,
        n_filters,
        first_layer_kernel_size,
        final_kernel_size,
        dilation,
        kernel_size,
        use_batchnorm=False,
        learning_rate=1e-3,
        loss_fn=TotalBPNetLoss(),
        n_channels=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.use_batchnorm = use_batchnorm
        self.n_layers = n_layers
        self.loss_fn = loss_fn

        self.iconv = nn.Conv1d(
            n_channels, n_filters, kernel_size=first_layer_kernel_size, padding="same"
        )
        self.irelu = nn.ReLU()
        self.ibatchnorm = nn.BatchNorm1d(n_filters) if self.use_batchnorm else None

        self.rconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    dilation=2**i if dilation else 1,
                    padding="same",
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.rrelus = nn.ModuleList([nn.ReLU() for _ in range(1, self.n_layers + 1)])
        self.rbatchnorms = (
            nn.ModuleList(
                [nn.BatchNorm1d(n_filters) for _ in range(1, self.n_layers + 1)]
            )
            if self.use_batchnorm
            else None
        )

        self.out_conv = nn.Conv1d(
            n_filters, n_tracks, kernel_size=final_kernel_size, padding="same"
        )
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(300)
        self.count_linear = nn.Linear(n_filters, n_tracks)

        # Initialize lists to store outputs and labels
        self.validation_outputs = []
        self.validation_labels = []

    def forward(self, X):
        X = X.float()
        X = self.irelu(self.iconv(X))  # Initial convolution
        if self.use_batchnorm:
            X = self.ibatchnorm(X)
        for i in range(self.n_layers):
            X_conv = self.rrelus[i](self.rconvs[i](X))
            if self.use_batchnorm:
                X_conv = self.rbatchnorms[i](X_conv)
            X = X + X_conv  # Residual connection
        bottleneck = X
        profile = self.out_conv(bottleneck)  # Profile head
        # profile = self.adaptive_pool(profile)
        count = self.global_avgpool(bottleneck).squeeze()  # Count head
        count = self.count_linear(count)

        return profile.float(), count.float()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
