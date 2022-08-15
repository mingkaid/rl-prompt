import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, TypeVar, Union, Tuple

T = TypeVar('T')
MaybeList = Union[T, List[T]]

def gather_2d_on_last_dim(
    tensor: torch.Tensor,
    index: torch.LongTensor,
    shape: torch.Size
) -> torch.Tensor:
    """Simplified version of `tf.gather_nd` in PyTorch"""
    flattened_tensor = tensor.view(-1, tensor.shape[-1])
    flattened_index = index.view(-1)
    flattened_gathered_tensor = flattened_tensor[
        torch.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.view(shape)


def get_masked_mean_min_max(
    X: torch.Tensor,
    lengths: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X.ndim != 2 and lengths.ndim != 1:
        raise ValueError

    if X.shape[0] != lengths.shape[0]:
        raise ValueError

    mask = get_lengths_mask(
        X=X,
        lengths=lengths)

    masked_min = X.masked_fill(~mask, np.inf).min(dim=1)
    masked_max = X.masked_fill(~mask, -np.inf).max(dim=1)
    masked_mean = mask_and_reduce(
        sequence=X,
        sequence_length=lengths,
        average_across_timesteps=True,
        sum_over_timesteps=False)

    return (masked_mean,
            masked_min.values.mean(),
            masked_max.values.mean())


def masked_reverse_cumsum(
    X: torch.Tensor,
    lengths: torch.LongTensor,
    dim: int
) -> torch.Tensor:
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])

    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def get_lengths_mask(
    X: torch.Tensor,
    lengths: torch.LongTensor
) -> torch.Tensor:

    if any([X.ndim != 2,
            lengths.ndim != 1,
            X.shape[0] != lengths.shape[0]]):
        raise ValueError

    # [batch_size, 1]
    expanded_lengths = lengths.unsqueeze(dim=1)
    return (torch.arange(X.shape[1], device=X.device)
            .expand(X.shape)
            < expanded_lengths)


# Below from Texar-PyTorch: https://github.com/asyml/texar-pytorch/blob/master/texar/torch/losses/losses_utils.py

def mask_and_reduce(
    sequence: torch.Tensor,
    sequence_length: Optional[torch.LongTensor],
    rank: int = 2,
    average_across_batch: bool = True,
    average_across_timesteps: bool = False,
    average_across_remaining: bool = False,
    sum_over_batch: bool = False,
    sum_over_timesteps: bool = True,
    sum_over_remaining: bool = True,
    dtype: Optional[torch.dtype] = None,
    time_major: bool = False
) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths, and reduces (average or sum) away dimensions.
    This is a combination of :func:`~texar.torch.utils.shapes.mask_sequences`
    and :func:`~texar.torch.losses.losses_utils.reduce_batch_time`.
    Args:
        sequence: A tensor of sequence values.
            If `time_major=False` (default), this must be a tensor of shape
            `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
            the tensor is specified with :attr:`rank`.
            The batch and time dimensions are exchanged if `time_major` is True.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        rank (int): The rank of :attr:`sequence`. Must be >= 2. Default is 2,
            i.e., `sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the time
            dimension. Must not set `average_across_timesteps` and
            `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the batch
            dimension. Must not set `average_across_batch` and `sum_over_batch`
            at the same time.
        sum_over_remaining (bool): If set, sum the sequence across the remaining
            dimension. Must not set `average_across_remaining` and
            `sum_over_remaining` at the same time.
        dtype (torch.dtype): The dtype of the returned mask.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), `sequence` must have
            shape `[batch_size, max_time, ...]`.
    Returns:
        A tensor containing the masked and reduced sequence.
    """
    if rank < 2:
        raise ValueError('`rank` must be >= 2.')

    if time_major:
        sequence = transpose_batch_time(sequence)

    if sequence_length is not None:
        sequence = mask_sequences(sequence,
                                  sequence_length,
                                  dtype=dtype,
                                  time_major=False)
    if rank > 2:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and "
                             "`sum_over_remaining` can be set.")
        if average_across_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.mean(sequence, dim=axis)
        elif sum_over_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.sum(sequence, dim=axis)

    sequence = reduce_batch_time(sequence,
                                 sequence_length,
                                 average_across_batch,
                                 average_across_timesteps,
                                 sum_over_batch,
                                 sum_over_timesteps)

    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch
    if not reduce_time and not reduce_batch and time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def reduce_batch_time(
    sequence: torch.Tensor,
    sequence_length: Optional[torch.LongTensor],
    average_across_batch: bool = True,
    average_across_timesteps: bool = False,
    sum_over_batch: bool = False,
    sum_over_timesteps: bool = True
) -> torch.Tensor:
    r"""Average or sum over the respective dimensions of :attr:`sequence`, which
    is of shape `[batch_size, max_time, ...]`.
    Assumes :attr:`sequence` has been properly masked according to
    :attr:`sequence_length`.
    Args:
        sequence: A tensor to reduce.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
    Returns:
        A tensor with dimension reduction.
    """
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        if sequence_length is None:
            sequence = torch.mean(sequence, dim=1)
        else:
            sequence = (torch.sum(sequence, dim=1).float() /
                        sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def reduce_dimensions(
    tensor: torch.Tensor,
    average_axes: Optional[MaybeList[int]] = None,
    sum_axes: Optional[MaybeList[int]] = None,
    keepdims: Optional[bool] = None
) -> torch.Tensor:
    r"""Average or sum over dimensions of :attr:`tensor`.
    :attr:`average_axes` and :attr:`sum_axes` must be mutually exclusive. That
    is, elements in `average_axes` must not be contained in
    `sum_axes`, and vice versa.
    Args:
        tensor: A tensor to reduce.
        average_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking average.
        sum_axes (optional): A (list of) `int` that indicates the
            dimensions to reduce by taking sum.
        keepdims (optional): If `True`, retains reduced dimensions with
            length 1.
    Returns:
        A tensor with dimension reduction.
    """
    reduced_axes = set()
    if average_axes is not None:
        if not isinstance(average_axes, (list, tuple)):
            average_axes = [average_axes]
        if len(average_axes) > 0:
            for average_axis in average_axes:
                tensor = torch.mean(tensor, dim=average_axis, keepdim=True)
            reduced_axes.update(average_axes)

    if sum_axes is not None:
        if not isinstance(sum_axes, (list, tuple)):
            sum_axes = [sum_axes]
        if len(sum_axes) > 0:
            for sum_axis in sum_axes:
                tensor = torch.sum(tensor, dim=sum_axis, keepdim=True)
            reduced_axes.update(sum_axes)

            if average_axes is not None:
                if len(reduced_axes) != len(average_axes) + len(sum_axes):
                    raise ValueError('`average_axes` and `sum_axes` must not '
                                     'have overlapped elements.')
    if not keepdims:
        for axis in sorted(list(reduced_axes), reverse=True):
            tensor = torch.squeeze(tensor, dim=axis)
    return tensor


# Below from Texar-PyTorch:
# https://github.com/asyml/texar-pytorch/blob/master/texar/torch/utils/utils.py

def sequence_mask(
    lengths: Union[torch.LongTensor, List[int]],
    max_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.ByteTensor:
    r"""Return a mask tensor representing the first N positions of each cell.
    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with
    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```
    Examples:
    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]
    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```
    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


# Below from 
# https://github.com/asyml/texar-pytorch/blob/master/texar/torch/losses/entropy.py
def _get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.
    Args:
        logits: Unscaled log probabilities.
    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy


# Below from
# https://github.com/asyml/texar-pytorch/blob/master/texar/torch/utils/shapes.py

def transpose_batch_time(inputs: torch.Tensor) -> torch.Tensor:
    r"""Transposes inputs between time-major and batch-major.
    Args:
        inputs: A Tensor of shape ``[batch_size, max_time, ...]`` (batch-major)
            or ``[max_time, batch_size, ...]`` (time-major), or a (possibly
            nested) tuple of such elements.
    Returns:
        A (possibly nested tuple of) Tensor with transposed batch and
        time dimensions of inputs.
    """
    return inputs.transpose(0, 1)


def mask_sequences(sequence: Union[torch.Tensor, List[int]],
                   sequence_length: Union[torch.LongTensor, List[int]],
                   dtype: Optional[torch.dtype] = None,
                   time_major: bool = False) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.
    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are Python arrays (or None), the
    return will be a Python array as well.
    Args:
        sequence: A Tensor or Python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.
    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).
        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)

    return sequence