import logging
import os
import sys
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import random
import pytorch_lightning as pl


def predict_for_eval(
    model,
    val_loader,
    apply_softmax=True,
    device="cpu",
):
    # Collect data
    model.eval()
    collected_labels = []
    collected_outputs = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            collected_labels.append(labels.detach().cpu())
            collected_outputs.append([x.detach().cpu() for x in outputs])

    count_preds = torch.concat([x[1] for x in collected_outputs], axis=0)
    profile_preds = torch.concat([x[0].float() for x in collected_outputs], axis=0)
    if apply_softmax:
        profile_preds = torch.softmax(profile_preds, axis=-1)

    profile_true = torch.concat(collected_labels, axis=0)
    return count_preds, profile_preds, profile_true


def set_seed(seed):
    # Python standard library
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch Lightning
    pl.seed_everything(seed)

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(run_name):
    log_dir = os.path.join(os.getcwd(), "logs/logger")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    # Logger
    logger = logging.getLogger(run_name)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
