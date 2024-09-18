import torch


def multinomial_nll(profile_pred, profile_true, eps=1e-3):
    profile_true = profile_true + eps
    log_fact_sum = torch.lgamma(torch.sum(profile_true, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(profile_true + 1), dim=-1)
    logps = torch.nn.functional.log_softmax(profile_pred, dim=-1)
    log_prod_exp = torch.sum(profile_true * logps, dim=-1)
    loss = (-log_fact_sum + log_prod_fact - log_prod_exp).mean()
    return loss


def profile_ce_loss(profile_pred, profile_true, eps=1e-3):
    profile_true = profile_true + eps
    profile_true = profile_true / profile_true.sum(axis=-1, keepdim=True)
    shape_tuple = (profile_true.shape[0] * profile_true.shape[1], profile_true.shape[2])
    loss = torch.nn.functional.cross_entropy(
        profile_pred.reshape(shape_tuple), profile_true.reshape(shape_tuple)
    )
    return loss


def count_mse_loss(count_pred, profile_true):
    log_true = torch.log(profile_true.sum(axis=-1) + 1)
    loss = torch.nn.functional.mse_loss(count_pred, log_true, reduction="mean")
    return loss


class TotalBPNetLoss:
    def __init__(self, alpha=0.2, beta=0.8, profile_loss_type="multinomial", eps=1e-3):
        self.alpha = alpha
        self.beta = beta
        self.profile_loss_type = profile_loss_type
        self.eps = eps

    def __call__(self, preds, labels):
        profile_pred, count_pred = preds
        profile_true = labels

        count_loss = count_mse_loss(count_pred, profile_true)
        if self.profile_loss_type == "multinomial":
            profile_loss = multinomial_nll(profile_pred, profile_true, eps=self.eps)
        elif self.profile_loss_type == "ce":
            profile_loss = profile_ce_loss(profile_pred, profile_true, eps=self.eps)
        else:
            raise ValueError("Invalid profile_loss_type specified")
        loss = (self.beta * profile_loss) + (self.alpha * count_loss)
        return loss
