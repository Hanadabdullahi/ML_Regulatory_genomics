import os
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import tqdm

from src.training.data_module import YeastDataModule
from src.utils.dataset import one_hot_encode_sequence, split_by_chrom
from transformers import (
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
)

from datasets import Dataset as PDataset

from src.utils.metrics import compute_metrics


def load_tif_seq_data(file_path):
    # Read the file into a list of lines
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Split the lines based on whitespace, keeping the last column as a single string
    data = []
    for line in lines:
        parts = line.strip().split(maxsplit=6)  # Split into at most 7 parts
        data.append(parts)

    # Create the DataFrame from the data
    df = pd.DataFrame(
        data[1:], columns=["chr", "strand", "t5", "t3", "ypd", "gal", "type_name"]
    )

    df["t5"] = df["t5"].astype(int)
    df["t3"] = df["t3"].astype(int)
    df["ypd"] = df["ypd"].astype(int)
    df["gal"] = df["gal"].astype(int)

    arabic_to_roman = {
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "6": "VI",
        "7": "VII",
        "8": "VIII",
        "9": "IX",
        "10": "X",
        "11": "XI",
        "12": "XII",
        "13": "XIII",
        "14": "XIV",
        "15": "XV",
        "16": "XVI",
    }

    df["pos"] = df["t3"] - 1
    df["Chromosome"] = df["chr"].apply(lambda x: arabic_to_roman[x])
    df["Strand"] = df["strand"]
    return df


def match_with_genomic_data(tif_df, dataset_path):
    dataset = pd.read_parquet(dataset_path)
    seq_col = "three_prime_seq"
    dataset = dataset.loc[
        dataset[seq_col].str.len() == 1003
    ]  # Filter sequences with length 1003
    regions = dataset[["Chromosome", "three_prime_start", "three_prime_end", "Strand"]]

    count_arrs = []
    for _, region in tqdm.tqdm(regions.iterrows(), total=regions.shape[0]):
        chrom = region["Chromosome"]
        start = region["three_prime_start"]
        end = region["three_prime_end"]
        strand = region["Strand"]
        region_arr = torch.zeros((2, 1003))
        region_tif = tif_df.query(
            "Chromosome == @chrom and Strand == @strand and pos > @start and pos < @end"
        )
        for _, row in region_tif.iterrows():
            relpos = (
                row["pos"] - start if strand == "+" else 1002 - (row["pos"] - start)
            )
            region_arr[0, relpos] += float(row["ypd"])
            region_arr[1, relpos] += float(row["gal"])
        count_arrs.append(region_arr)

    counts = torch.stack(count_arrs)
    return dataset, counts


def save_processed_data(counts, dataset, counts_file, preprocessed_file):
    torch.save(counts, counts_file)
    dataset.to_parquet(preprocessed_file)


def preprocess_data(
    dataset, counts, restrict_seq_len, seq_col, val_chroms, test_chroms
):
    dataset = dataset.loc[dataset[seq_col].str.len() == 1003]
    if restrict_seq_len:
        dataset[seq_col] = dataset[seq_col].str[:300]
        counts = counts[:, :, :300]

    dataset = dataset.reset_index(drop=True)
    count_sums = counts.sum(axis=-1)  # Sum over the last dimension

    count_min = count_sums.min(
        dim=-1
    ).values  # Find the minimum over the second to last dimension

    count_mask = count_min > 0  # Create mask where the minimum value is greater than 0

    train_idx, val_idx, test_idx = split_by_chrom(
        dataset, test_chroms, val_chroms, count_mask=count_mask
    )
    one_hots = torch.stack(
        [one_hot_encode_sequence(x) for x in dataset["three_prime_seq"]], axis=0
    ).permute(0, 2, 1)

    return train_idx, val_idx, test_idx, one_hots, counts, dataset


def get_tokens(dataset, stride=1, seq_col="three_prime_seq", kmer_size=6):
    # Initialize the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(
        "gagneurlab/SpeciesLM", revision="downstream_species_lm"
    )

    # Function to generate k-mers from the sequence
    def get_kmers(seq, k=6, stride=1):
        return [seq[i : i + k] for i in range(0, len(seq) - k + 1, stride)]

    # Function to tokenize the k-mers of the sequence
    def tok_func(x):
        kmers = get_kmers(x[seq_col], k=kmer_size, stride=stride)
        tokenized_output = tokenizer(
            "yeast " + " ".join(kmers), add_special_tokens=True
        )
        return tokenized_output

    # Convert the pandas DataFrame to a Dataset
    ds = PDataset.from_pandas(dataset[[seq_col]])

    # Apply the tokenization function to the dataset
    tok_ds = ds.map(tok_func, batched=False, num_proc=4)
    print("test")
    # Remove the original sequence column

    embeddings = tok_ds["input_ids"]
    # its a list of lists
    embeddings = [torch.tensor(x) for x in embeddings]
    embeddings = torch.stack(embeddings)

    return embeddings


def validate_test_data(model, test_loader, loss_fn):
    model.eval()

    all_count_preds, all_count_true = [], []
    all_profile_preds, all_profile_true = [], []
    all_losses = []

    for batch in test_loader:
        inputs, labels = batch
        with torch.no_grad():
            output = model(inputs)
            loss = loss_fn(output, labels)

        all_count_preds.append(output[1].cpu())
        counts = torch.log(labels.sum(dim=-1) + 1)
        all_count_true.append(counts.cpu())
        all_profile_preds.append(output[0].cpu())
        all_profile_true.append(labels.cpu())
        all_losses.append(loss.cpu().item())

    count_preds = torch.cat(all_count_preds)
    count_true = torch.cat(all_count_true)
    profile_preds = torch.cat(all_profile_preds)
    profile_true = torch.cat(all_profile_true)

    # Compute metrics
    metrics = compute_metrics(
        count_preds,
        count_true,
        profile_preds,
        profile_true,
        preds_in_logit_scale=True,
        min_total_count_cutoff=5,
        pos_cutoff=5,
        fast_pearson=True,
        smoothing_kernel_sizes=[1, 2, 5, 10],
    )

    avg_loss = np.mean(all_losses)
    return metrics, avg_loss
