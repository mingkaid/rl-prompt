import torch
import numpy as np
import torch.nn.functional as F
from functools import partial

from typing import Tuple, Dict, Any, Optional

from rlprompt.losses import loss_utils
from rlprompt.utils import utils


def sql_loss_with_sparse_rewards(
        implementation: str,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        sampled_actions: Optional[torch.LongTensor],
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Soft Q Learning Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        logits_pi:       [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
        sequence_length: [batch_size]
    """
    if implementation not in ["v0", "v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r"]:
        raise ValueError

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    if implementation == "v0":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_0

    if implementation == "v1":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_1

    if implementation == "v2":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_2

    if implementation == "v3":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_3

    if implementation == "v2_v2r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed,
            coefficient=coefficient,
            margin_constant=margin_constant,
            margin_coefficient=margin_coefficient)

    if implementation == "v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_3_3_reversed,
            coefficient=coefficient)

    if implementation == "v2_v2r_v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed,
            coefficient=coefficient)

    if logits.shape != logits_.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_.shape` = {logits_.shape}")

    raw_losses, quantities_to_log = _sql_loss_func(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length
    )

    loss = loss_utils.mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": loss_utils.mask_and_reduce(
            sequence=raw_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
    }

    for key, value in quantities_to_log.items():
        masked_mean, masked_min, masked_max = \
            loss_utils.get_masked_mean_min_max(value,
                                               lengths=sequence_length)
        loss_log[f"{key}/min"] = masked_min
        loss_log[f"{key}/max"] = masked_max
        loss_log[f"{key}/mean"] = masked_mean

    return loss, loss_log


def soft_q_loss_with_sparse_rewards_1(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    # use `V` from the target if available
    V_ = logits_.logsumexp(dim=-1)

    # Build the target `= V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    Q_[:, :-1] = V_[:, 1:]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards

    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": logits.logsumexp(dim=-1),
        "Q_": Q_,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        _recover_mle: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V
    # print(logits.shape)
    # print(V)
    print(Q)

    # Target outputs
    Q_ = torch.zeros_like(Q)
    A_ = torch.zeros_like(Q)
    V_ = logits_.logsumexp(dim=-1)
    Q_[:, :-1] = V_[:, 1:]
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # Terminal V-target is the last V-target before
    # the episode ends, thus depends on `sequence_length`
    terminal_V_ = V_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1]
#     print(Q_)
#     print(sequence_length)
#     print(torch.arange(sequence_length.shape[0]))
#     print(rewards)
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards
    A_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards - terminal_V_

    if _recover_mle is True:
        utils.colorful_warning("Recover-MLE Mode", bg="red")
        A_ = A.detach() + 1

    raw_losses = F.mse_loss(A, A_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_,
        "H": loss_utils._get_entropy(logits),
        "H_": loss_utils._get_entropy(logits_),
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        freeze_future_steps: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    V_ = logits_.logsumexp(dim=-1)

    A2 = loss_utils.masked_reverse_cumsum(
        A,
        lengths=sequence_length,
        dim=-1)

    if freeze_future_steps is True:
        # This line of code essentially
        # decompose `A` (with gradient)
        # and cumsum of future `A`
        # (without gradient)
        A2 = (A2 - A).detach() + A

    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="0/")

    if coefficient is not None:
        raw_losses_2_r, quantities_to_log_2_r = soft_q_loss_with_sparse_rewards_2(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_2 +
            (1 - coefficient) * raw_losses_2_r)

        utils.add_prefix_to_dict_keys_inplace(
            quantities_to_log_2_r, prefix="1/")

        quantities_to_log = utils.unionize_dicts([
            quantities_to_log_2,
            quantities_to_log_2_r,
        ])

    else:
        raw_losses = raw_losses_2
        quantities_to_log = quantities_to_log_2

    if margin_constant is not None and margin_coefficient is not None:
        raw_losses_margin, quantities_to_log_margin = large_margin_classification_loss(
            logits=logits,
            expert_actions=actions,
            margin_constant=margin_constant)

        raw_losses = raw_losses + margin_coefficient * raw_losses_margin
        utils.add_prefix_to_dict_keys_inplace(
            quantities_to_log_margin, prefix="margin/")
        quantities_to_log = utils.unionize_dicts([
            quantities_to_log,
            quantities_to_log_margin,
        ])

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="0/")

    if coefficient is not None:
        raw_losses_3_r, quantities_to_log_3_r = soft_q_loss_with_sparse_rewards_3(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_3 +
            (1 - coefficient) * raw_losses_3_r)

        utils.add_prefix_to_dict_keys_inplace(
            quantities_to_log_3_r, prefix="1/")

        quantities_to_log = utils.unionize_dicts([
            quantities_to_log_3,
            quantities_to_log_3_r,
        ])
    else:
        raw_losses = raw_losses_3
        quantities_to_log = quantities_to_log_3

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses = (raw_losses_2 + raw_losses_3) / 2
    # print(rewards)
    # print(raw_losses)

    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="v2/")
    utils.add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="v3/")
    quantities_to_log = utils.unionize_dicts([
        quantities_to_log_2,
        quantities_to_log_3,
    ])
    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_0(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.Tensor,
        sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    V = logits.logsumexp(dim=-1)
    V_ = logits_.logsumexp(dim=-1)
    raw_losses = F.mse_loss(V, V_, reduction="none")
    quantities_to_log = {
        "V": V,
        "V_": V_,
    }

    return raw_losses, quantities_to_log
