from typing import Any
PrecisionLike = Any

import builtins
from enum import IntEnum
import functools
import itertools
import operator
from typing import (Any, Callable, List, NamedTuple, Optional, Sequence,\
                    Union, Tuple)
import warnings

import numpy as np

import jax
from jax import core
from jax import dtypes
from jax import config
# from jax.abstract_arrays import array_types
from jax.tree_util import tree_map

_max = builtins.max
_min = builtins.min
_reduce = functools.reduce

Array = Any
DType = Any
Shape = Sequence[int]


class ConvDimensionNumbers(NamedTuple):
  """Describes batch, spatial, and feature dimensions of a convolution.

  Args:
    lhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
    rhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(out feature dimension, in feature dimension, spatial dimensions...)`.
    out_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
  """
  lhs_spec: Sequence[int]
  rhs_spec: Sequence[int]
  out_spec: Sequence[int]

ConvGeneralDilatedDimensionNumbers = Union[
  None, ConvDimensionNumbers, Tuple[str, str, str]]

def _flip_axes(x, axes):
  """Flip ndarray 'x' along each axis specified in axes tuple."""
  for axis in axes:
    x = np.flip(x, axis)
  return x


def _deconv_output_length(input_length,
                          filter_size,
                          padding,
                          output_padding=None,
                          stride=0,
                          dilation=1):
  """Determines the output length of a transposed convolution given the input length.

  Function modified from Keras.

  Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"SAME"`, `"VALID"`, or a 2-integer tuple.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'VALID':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'SAME':
      length = input_length * stride
    else:
      length = ((input_length - 1) * stride + filter_size
                - padding[0] - padding[1])

  else:
    if padding == 'SAME':
      pad = filter_size // 2
      total_pad = pad * 2
    elif padding == 'VALID':
      total_pad = 0
    else:
      total_pad = padding[0] + padding[1]

    length = ((input_length - 1) * stride + filter_size - total_pad +
              output_padding)

  return length


def _compute_adjusted_padding(
    input_size: int,
    output_size: int,
    kernel_size: int,
    stride: int,
    padding: Union[str, Tuple[int, int]],
    dilation: int = 1,
) -> Tuple[int, int]:
  """Computes adjusted padding for desired ConvTranspose `output_size`.

  Ported from DeepMind Haiku.
  """
  kernel_size = (kernel_size - 1) * dilation + 1
  if padding == "VALID":
    expected_input_size = (output_size - kernel_size + stride) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_before = 0
  elif padding == "SAME":
    expected_input_size = (output_size + stride - 1) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_needed = max(0,
                         (input_size - 1) * stride + kernel_size - output_size)
    padding_before = padding_needed // 2
  else:
    padding_before = padding[0]  # type: ignore[assignment]

  expanded_input_size = (input_size - 1) * stride + 1
  padded_out_size = output_size + kernel_size - 1
  pad_before = kernel_size - 1 - padding_before
  pad_after = padded_out_size - expanded_input_size - pad_before
  return (pad_before, pad_after)


def gradient_based_conv_transpose(lhs: Array, rhs: Array, strides: Sequence[int],
                                  padding: Union[str, Sequence[Tuple[int, int]]],
                                  output_padding: Optional[Sequence[int]] = None,
                                  output_shape: Optional[Sequence[int]] = None,
                                  dilation: Optional[Sequence[int]] = None,
                                  dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
                                  transpose_kernel: bool = True,
                                  precision: PrecisionLike = None) -> Array:
  """Convenience wrapper for calculating the N-d transposed convolution.

  Much like `conv_transpose`, this function calculates transposed convolutions
  via fractionally strided convolution rather than calculating the gradient
  (transpose) of a forward convolution. However, the latter is more common
  among deep learning frameworks, such as TensorFlow, PyTorch, and Keras.
  This function provides the same set of APIs to help reproduce results in these frameworks.

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    strides: sequence of `n` integers, amounts to strides of the corresponding forward convolution.
    padding: `"SAME"`, `"VALID"`, or a sequence of `n` integer 2-tuples that controls
      the before-and-after padding for each `n` spatial dimension of
      the corresponding forward convolution.
    output_padding: A sequence of integers specifying the amount of padding along
      each spacial dimension of the output tensor, used to disambiguate the output shape of
      transposed convolutions when the stride is larger than 1.
      (see a detailed description at
      1https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
      The amount of output padding along a given dimension must
      be lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
      If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
    output_shape: Output shape of the spatial dimensions of a transpose
      convolution. Can be `None` or an iterable of `n` integers. If a `None` value is given (default),
      the shape is automatically calculated.
      Similar to `output_padding`, `output_shape` is also for disambiguating the output shape
      when stride > 1 (see also
      https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose)
      If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
    dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. Dilated convolution
      is also known as atrous convolution.
    dimension_numbers: tuple of dimension descriptors as in
      lax.conv_general_dilated. Defaults to tensorflow convention.
    transpose_kernel: if `True` flips spatial axes and swaps the input/output
      channel axes of the kernel. This makes the output of this function identical
      to the gradient-derived functions like keras.layers.Conv2DTranspose and
      torch.nn.ConvTranspose2d applied to the same kernel.
      Although for typical use in neural nets this is unnecessary
      and makes input/output channel specification confusing, you need to set this to `True`
      in order to match the behavior in many deep learning frameworks, such as TensorFlow, Keras, and PyTorch.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.

  Returns:
    Transposed N-d convolution.
  """
  assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
  ndims = len(lhs.shape)
  one = (1,) * (ndims - 2)
  # Set dimensional layout defaults if not specified.
  if dimension_numbers is None:
    if ndims == 2:
      dimension_numbers = ('NC', 'IO', 'NC')
    elif ndims == 3:
      dimension_numbers = ('NHC', 'HIO', 'NHC')
    elif ndims == 4:
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    elif ndims == 5:
      dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
    else:
      raise ValueError('No 4+ dimensional dimension_number defaults.')
  dn = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  k_shape = np.take(rhs.shape, dn.rhs_spec)
  k_sdims = k_shape[2:]  # type: ignore[index]
  i_shape = np.take(lhs.shape, dn.lhs_spec)
  i_sdims = i_shape[2:]  # type: ignore[index]

  # Calculate correct output shape given padding and strides.
  if dilation is None:
    dilation = (1,) * (rhs.ndim - 2)

  if output_padding is None:
    output_padding = [None] * (rhs.ndim - 2)  # type: ignore[list-item]

  if isinstance(padding, str):
    if padding in {'SAME', 'VALID'}:
      padding = [padding] * (rhs.ndim - 2)  # type: ignore[list-item]
    else:
      raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

  inferred_output_shape = tuple(map(_deconv_output_length, i_sdims, k_sdims,
                              padding, output_padding, strides, dilation))
  if output_shape is None:
    output_shape = inferred_output_shape  # type: ignore[assignment]
  else:
    if not output_shape == inferred_output_shape:
      raise ValueError(f"`output_padding` and `output_shape` are not compatible."
                       f"Inferred output shape from `output_padding`: {inferred_output_shape}, "
                       f"but got `output_shape` {output_shape}")

  pads = tuple(map(_compute_adjusted_padding, i_sdims, output_shape,
                   k_sdims, strides, padding, dilation))

  if transpose_kernel:
    # flip spatial dims and swap input / output channel axes
    rhs = _flip_axes(rhs, np.array(dn.rhs_spec)[2:])
    rhs = np.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
  return jax.lax.conv_general_dilated(lhs, rhs, one, pads, strides, dilation, dn,
                              precision=precision)