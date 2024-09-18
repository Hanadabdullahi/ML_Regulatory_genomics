import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..training.loss import TotalBPNetLoss
from src.training.loss import TotalBPNetLoss
from src.utils.metrics import compute_metrics


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.validation_outputs = []
        self.validation_labels = []

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Save the outputs and labels for metric computation
        self.validation_outputs.append(outputs)
        self.validation_labels.append(labels)

        return {"loss": loss, "outputs": outputs, "labels": labels}

    def on_validation_epoch_end(self):
        # Compute metrics using the collected outputs and labels
        count_preds = torch.cat([x[1] for x in self.validation_outputs], axis=0)
        count_true = torch.log(
            torch.cat(self.validation_labels, axis=0).sum(axis=-1) + 1
        )
        profile_preds = torch.cat(
            [x[0].float() for x in self.validation_outputs], axis=0
        )
        profile_true = torch.cat(self.validation_labels, axis=0)
        val_metrics = compute_metrics(
            count_preds.cpu(),
            count_true.cpu(),
            profile_preds.cpu(),
            profile_true.cpu(),
            preds_in_logit_scale=True,
        )

        self.log("val_count_r2", np.mean(val_metrics["count_r2"]), prog_bar=True)
        self.log(
            "val_profile_pearson_median",
            np.mean(val_metrics["profile_pearson_median"]),
            prog_bar=True,
        )
        self.log(
            "val_profile_pearson_mean",
            np.mean(val_metrics["profile_pearson_mean"]),
            prog_bar=True,
        )
        self.log(
            "val_profile_auprc", np.mean(val_metrics["profile_auprc"]), prog_bar=True
        )

        # Clear the lists for the next epoch
        self.validation_outputs.clear()
        self.validation_labels.clear()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return {"loss": loss, "outputs": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Save the outputs and labels for metric computation
        self.validation_outputs.append(outputs)
        self.validation_labels.append(labels)

        return {"loss": loss, "outputs": outputs, "labels": labels}
