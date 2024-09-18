import torch
import pandas as pd


def split_by_chrom(
    df,
    test_chroms,
    val_chroms,
    excl_chroms=set(["Mito"]),
    chr_col="Chromosome",
    count_mask=None,
):
    train_idx = torch.tensor(
        list(df.loc[~df[chr_col].isin(test_chroms | val_chroms | excl_chroms)].index)
    )
    val_idx = torch.tensor(list(df.loc[df[chr_col].isin(val_chroms)].index))
    test_idx = torch.tensor(list(df.loc[df[chr_col].isin(test_chroms)].index))
    if count_mask is not None:
        count_mask_train = count_mask[train_idx]
        train_idx = train_idx[count_mask_train]
        count_mask_val = count_mask[val_idx]
        val_idx = val_idx[count_mask_val]
        count_mask_test = count_mask[test_idx]
        test_idx = test_idx[count_mask_test]
    return train_idx, val_idx, test_idx


def one_hot_encode_sequence(seq):
    nuc_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_mask = []
    nuc_idxs = []
    for nuc in seq:
        nuc_idx = nuc_to_int.get(nuc, -1)
        if nuc_idx >= 0:
            nuc_idxs.append(nuc_idx)
            n_mask.append(1)
        else:
            nuc_idxs.append(0)
            n_mask.append(0)
    seq_enc = torch.tensor(nuc_idxs)
    return torch.nn.functional.one_hot(seq_enc, num_classes=4) * torch.tensor(
        n_mask
    ).unsqueeze(-1)
