# Project description.

This repository contains code for building, training, and evaluating a neural network model for predicting gene expression profiles from DNA sequences using PyTorch Lightning and Hydra for configuration management.

## Installation

To install the required dependencies, use the provided `environment.yml` file and run:

```sh
conda env create -f environment.yml
conda activate regulate-me
```
<!-- Uncomment  pytorch-cuda=12.1 if used  -->
**Note:** If you are using a GPU, you will need to install the appropriate version of PyTorch for your CUDA version.

### Copy the datafolder

Move `saccharomyces_cerevisiae` to a new folder `data` in the root directory of the project.

### Preprocess the data

Needs to be done only once. Run the following command to preprocess the data:

```sh
python scripts/preprocess_data.py
```

## Configuration

You will need to login to wandb to use the sweep functionality. You can do this by running the following command and following the instructions:

```sh
wandb login
```

## Usage

To run training using the config file `config.yaml` in the `configs` folder, use the following command:

```sh
python scripts/train_model.py --config-name config
```

There are 3 differnet configuration files for 3 different models:

- `config.yaml` for the baseline BPNet model
- `lora.yaml` for the SpeciesLM model with a 1D convolutional prediction head
- `llm_bpnet.yaml` for the SpeciesLM model with BPNet prediction head

## Lora

Lora is used by defualt in both the `lora.yaml` and `llm_bpnet.yaml` configuration files. To toggle Lora on or off, set the `use_lora` parameter to `True` or `False` respectively. This illustreates also how other parameters can be changed from the command line.

```sh
python scripts/train_model.py --config-name lora model.use_lora=False
```

## Hyperparameter Optimization

To run hyperparameter optimization using `sweep.yaml` in the `configs` folder, use the following command. Needs to be defnied per model in the `sweep.yaml` file.
```sh
python scripts/run_sweep.py --config-name llm_bpnet
```

## Notebooks

The `notebooks` folder contains a Jupyter notebook for interpreation of the model results and also for final model evaluation as well as a notebook for interactive training of the model.

- `interpretation_bpnet.ipynb` for interpretation of the BPNet model results
- `interpretation_lora.ipynb` for interpretation of the SpeciesLM model results
- `final_evaluation.ipynb` for final model evaluation on the test set
