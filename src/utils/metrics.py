import sklearn.metrics
import torch
import numpy as np
import scipy
import sklearn
import collections
import functorch


def pearson_correlation_rowwise(matrix_a_row, matrix_b_row):
    mean_a = torch.mean(matrix_a_row)
    mean_b = torch.mean(matrix_b_row)
    centered_a = matrix_a_row - mean_a
    centered_b = matrix_b_row - mean_b
    covariance = torch.sum(centered_a * centered_b)
    std_a = torch.sqrt(torch.sum(centered_a**2))
    std_b = torch.sqrt(torch.sum(centered_b**2))
    correlation = covariance / (std_a * std_b)
    return correlation


def pearson_correlation_with_pooling(
    profile_true_track, profile_pred_track, kernel_size
):
    # this function computes the pearson correlation row-wise (i.e. per profile)
    # it accounts for overlaps
    # it also allows smoothing over the profile to reduce the resolution (to allow for small errors)
    corrs = []
    for i in range(profile_true_track.shape[0]):
        # subset
        true = profile_true_track[i]
        pred = profile_pred_track[i]
        # apply pooling
        if kernel_size > 1:
            # apply average pooling to reduce resolution
            true = (
                torch.nn.functional.avg_pool1d(
                    true.unsqueeze(0), kernel_size, padding=(kernel_size - 1) // 2
                )
                * kernel_size
            ).squeeze(0)
            pred = (
                torch.nn.functional.avg_pool1d(
                    pred.unsqueeze(0), kernel_size, padding=(kernel_size - 1) // 2
                )
                * kernel_size
            ).squeeze(0)
        corrs.append(
            scipy.stats.pearsonr(true.numpy(force=True), pred.numpy(force=True))[0]
        )
    return torch.tensor(corrs)


def classification_metrics_with_binning(
    replicated_pos_track, replicated_neg_track, profile_pred_prbs_track, kernel_size
):
    if kernel_size > 1:
        # apply max pooling to reduce resolution for positive labels (if one position is positive, the whole bin is positive)
        pos = (
            (
                torch.nn.functional.max_pool1d(
                    replicated_pos_track.unsqueeze(0).float(),
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
                * kernel_size
            )
            .squeeze(0)
            .bool()
        )
        # for negatives, we demand that the entire bin is negative
        neg = (
            torch.nn.functional.avg_pool1d(
                replicated_neg_track.unsqueeze(0).float(),
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            * kernel_size
        ).squeeze(0) == kernel_size
        # apply max pooling to predictions
        pred_prbs = (
            torch.nn.functional.max_pool1d(
                profile_pred_prbs_track.unsqueeze(0),
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            * kernel_size
        ).squeeze(0)
    else:
        pos = replicated_pos_track
        neg = replicated_neg_track
        pred_prbs = profile_pred_prbs_track
    nonambig = (
        pos | neg
    )  # exclude positions which are neither clear positives nor clear negatives
    pos_nonambig = pos[nonambig]
    class_balance = pos_nonambig.sum() / pos_nonambig.shape[0]
    pred_prbs_nonambig = pred_prbs[nonambig]
    profile_auprc = sklearn.metrics.average_precision_score(
        pos_nonambig.numpy(), pred_prbs_nonambig.numpy()
    )

    profile_auroc = sklearn.metrics.roc_auc_score(
        pos_nonambig.numpy(), pred_prbs_nonambig.numpy()
    )
    return profile_auprc, profile_auroc, class_balance.item()


def compute_metrics(
    count_preds,
    count_true,
    profile_preds,
    profile_true,
    preds_in_logit_scale=True,
    min_total_count_cutoff=5,
    pos_cutoff=5,
    fast_pearson=True,
    smoothing_kernel_sizes=[1, 2, 5, 10],
):
    metrics = collections.defaultdict(list)
    metrics["count_r2"] = [
        scipy.stats.pearsonr(count_preds[:, x], count_true[:, x])[0] ** 2
        for x in range(count_preds.shape[1])
    ]

    replicated_neg = profile_true.amax(axis=1) == 0
    replicated_pos = profile_true.amin(axis=1) > 0
    n_tracks = profile_preds.shape[1]

    if preds_in_logit_scale:
        profile_pred_prbs = torch.softmax(profile_preds, axis=-1)
    else:
        profile_pred_prbs = profile_preds / profile_preds.sum(axis=-1, keepdim=True)

    for x in range(n_tracks):
        profile_pred_track = profile_preds[:, x, :]
        profile_pred_prbs_track = profile_pred_prbs[:, x, :]
        profile_true_track = profile_true[:, x, :]
        min_count = profile_true_track.sum(axis=-1) > min_total_count_cutoff
        profile_pred_track = profile_pred_track[min_count]
        profile_pred_prbs_track = profile_pred_prbs_track[min_count]
        profile_true_track = profile_true_track[min_count]
        replicated_neg_track = replicated_neg[min_count]
        replicated_pos_track = replicated_pos[min_count]
        profile_true_track = profile_true_track / profile_true_track.sum(
            axis=-1, keepdim=True
        )

        if fast_pearson:
            profile_pearson = torch.nan_to_num(
                torch.vmap(pearson_correlation_rowwise)(
                    profile_pred_prbs_track, profile_true_track
                )
            )
            metrics["profile_pearson_median"].append(
                profile_pearson.median().cpu().item()
            )
            metrics["profile_pearson_mean"].append(profile_pearson.mean().cpu().item())
        else:
            for kernel_size in smoothing_kernel_sizes:
                profile_pearson = pearson_correlation_with_pooling(
                    profile_true_track, profile_pred_prbs_track, kernel_size=kernel_size
                )
                profile_pearson = torch.nan_to_num(profile_pearson)
                metrics["profile_pearson_median_{}bp".format(kernel_size)].append(
                    profile_pearson.median().cpu().item()
                )
                metrics["profile_pearson_mean_{}bp".format(kernel_size)].append(
                    profile_pearson.mean().cpu().item()
                )

        if fast_pearson:
            profile_auprc, profile_auroc, class_balance = (
                classification_metrics_with_binning(
                    replicated_pos_track,
                    replicated_neg_track,
                    profile_pred_prbs_track,
                    kernel_size=1,
                )
            )
            metrics["profile_auprc"].append(profile_auprc)
            metrics["profile_auroc"].append(profile_auroc)
        else:
            for kernel_size in smoothing_kernel_sizes:
                profile_auprc, profile_auroc, class_balance = (
                    classification_metrics_with_binning(
                        replicated_pos_track,
                        replicated_neg_track,
                        profile_pred_prbs_track,
                        kernel_size=kernel_size,
                    )
                )
                metrics["profile_auprc_{}bp".format(kernel_size)].append(profile_auprc)
                metrics["profile_auroc_{}bp".format(kernel_size)].append(profile_auroc)
                metrics["class_balance_{}bp".format(kernel_size)].append(class_balance)

    return metrics


