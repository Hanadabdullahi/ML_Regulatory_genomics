import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch


class YeastDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_idx, val_idx, test_idx, data, counts):
        super().__init__()
        self.batch_size = batch_size
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.one_hots = data
        self.counts = counts

    def setup(self, stage=None):
        self.train_data = TensorDataset(
            self.one_hots[self.train_idx], self.counts[self.train_idx]
        )
        self.val_data = TensorDataset(
            self.one_hots[self.val_idx], self.counts[self.val_idx]
        )
        self.test_data = TensorDataset(
            self.one_hots[self.test_idx], self.counts[self.test_idx]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
