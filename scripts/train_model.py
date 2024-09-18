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

from src.utils.preprocess import preprocess_data, get_tokens
from src.utils.utils import set_seed, setup_logger, count_parameters
from src.training.data_module import YeastDataModule
from src.models.bpnet import BPNet
from src.models.transformer_lora import LoraBPNet
from src.models.lora import Lora
from src.training.loss import TotalBPNetLoss


def train(config: DictConfig):
    set_seed(config.seed)
    torch.set_float32_matmul_precision("medium")

    run_name = wandb.run.name
    logger = setup_logger(run_name)
    logger.info(OmegaConf.to_yaml(config))

    # See if GPUs are available
    logger.info("Is CUDA available: %s", torch.cuda.is_available())

    # Load data
    counts = torch.load(os.path.join(config.data.data_dir, config.data.counts_file))
    dataset = pd.read_parquet(
        os.path.join(config.data.data_dir, config.data.preprocessed_file)
    )
    logger.info(f"Data loaded from {config.data.data_dir}")
    logger.info(f"Counts shape: {counts.shape}")
    logger.info(f"Dataset shape: {dataset.shape}")

    # Preprocess data
    train_idx, val_idx, test_idx, one_hots, counts, dataset = preprocess_data(
        dataset,
        counts,
        config.data.restrict_seq_len,
        config.data.seq_col,
        set(config.data.val_chroms),
        set(config.data.test_chroms),
    )
    logger.info("Data preprocessed")

    if config.model_name != "bpnet":
        # Calculate the sequence length
        output_seq_len = 300 - config.tokenizer.kmer_size + 1
        counts = counts[:, :, :output_seq_len]
        data = get_tokens(
            dataset,
            config.tokenizer.stride,
            config.data.seq_col,
            config.tokenizer.kmer_size,
        )
    else:
        data = one_hots

    # Create DataLoader
    data_module = YeastDataModule(
        batch_size=config.training.batch_size,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        data=data,
        counts=counts,
    )
    logger.info("Data module initialized")

    # Initialize loss function
    loss_fn = TotalBPNetLoss(
        alpha=config.loss.alpha,
        beta=config.loss.beta,
        profile_loss_type=config.loss.profile_loss_type,
        eps=config.loss.eps,
    )
    logger.info("Loss function initialized")

    # Initialize model
    if config.model_name == "bpnet":
        model = BPNet(**config.bpnet_model)
    elif config.model_name == "llm_bpnet":
        model = LoraBPNet(bpnet_params=config.bpnet_model, **config.model)
    elif config.model_name == "lora":
        model = Lora(**config.model)
    else:
        raise ValueError(f"Model {config.model_name} not recognized")

    # Add wandb run ID and name to model hparams
    model.hparams["wandb_run_id"] = wandb.run.id
    model.hparams["wandb_run_name"] = wandb.run.name

    logger.info("Model initialized")
    logger.info(f"Model name: {model.__class__.__name__}")
    logger.info(f"Model config: {model.hparams}")
    logger.info(model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        default_root_dir=os.getcwd() + "/logs",
    )

    use_lora = config.get("use_lora", False)
    use_lora = "lora" if use_lora else "no_lora"
    model_identifer = f"{config.model_name}_{use_lora}"
    # log identifier
    wandb_logger.experiment.log({"model_identifier": model_identifer})

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "logs", "checkpoints"),
        filename="best-checkpoint",
        save_top_k=1,  # Only keep the best model
        monitor="val_loss",
        mode="min",  # 'min' means it will keep the checkpoint with the lowest val_loss
    )

    num_params = count_parameters(model)
    wandb.log({"Number of trainable parameters": num_params})

    # Trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=config.training.patience),
        ],
        accelerator=config.training.accelerator,
        default_root_dir=os.getcwd() + "/logs",
    )

    # Log the GPU device information
    if torch.cuda.is_available():
        device_index = trainer.strategy.root_device.index
        device_name = torch.cuda.get_device_name(device_index)
        logger.info(f"Using GPU: {device_name} (Device Index: {device_index})")
    else:
        logger.info("No GPU available. Using CPU.")

    # Train model
    trainer.fit(model, data_module)
    logger.info("Model trained")

    # Save model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved to {best_model_path}")

    model_save_path = os.path.join(
        config.data.model_dir, f"{config.model_name}_{run_name}.ckpt"
    )
    os.makedirs(config.data.model_dir, exist_ok=True)
    trainer.save_checkpoint(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    wandb.finish()
    return trainer, model


@hydra.main(config_path="../configs", config_name="llm_bpnet", version_base="1.3")
def main(config):
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True),
        dir= os.path.join(os.getcwd(), "logs"),
        settings=wandb.Settings(start_method="fork"),
    )
    train(config)


if __name__ == "__main__":
    main()
