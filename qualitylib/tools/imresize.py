# MIT License

# Copyright (c) 2020 Alex

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
from typing import Callable, Tuple, List, Union, Optional
import numpy as np
from math import ceil


def _derive_size_from_scale(img_shape: Tuple[int], scale: Tuple[float]) -> Tuple[int]:
    '''
    Calculate output size from input size and scale factor.

    Args:
        img_shape: Shape of input image
        scale: Resize scale factors along each dimension

    Returns:
        Shape of resized image
    '''
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def _derive_scale_from_size(img_shape_in: Tuple[int], img_shape_out: Tuple[int]) -> Tuple[float]:
    '''
    Calculate output size from input size and scale factor.

    Args:
        img_shape_in: Shape of input image
        imag_shape_out: Shape of resized image

    Returns:
        Effective resize scale factors along each dimension
    '''
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return tuple(scale)


def _triangle(x: Union[np.ndarray, List[float]]) -> np.ndarray:
    '''
    Triangle (linear) interpolation weights

    Args:
        x: Input values to be interpolated

    Returns:
        np.ndarray: Interpolated values
    '''
    x = np.array(x).astype(np.float64)
    negative = np.logical_and((x>=-1), x<0)
    non_negative = np.logical_and((x<=1), x>=0)
    f = np.multiply((x+1), negative) + np.multiply((1-x), non_negative)
    return f


def _cubic(x: Union[np.ndarray, List[float]]) -> np.ndarray:
    '''
    Cubic interpolation weights

    Args:
        x: Input values to be interpolated

    Returns:
        np.ndarray: Interpolated values
    '''
    x = np.array(x).astype(np.float64)
    abs_x = np.absolute(x)
    abs_x2 = np.multiply(abs_x, abs_x)
    abs_x3 = np.multiply(abs_x2, abs_x)
    f = np.multiply(1.5*abs_x3 - 2.5*abs_x2 + 1, abs_x <= 1) + np.multiply(-0.5*abs_x3 + 2.5*abs_x2 - 4*abs_x + 2, (1 < abs_x) & (abs_x <= 2))
    return f


def _contributions(in_length: int, out_length: int, scale: float, kernel: Callable, k_width: int) -> Tuple[np.ndarray]:
    '''
    Interpolation contributions of each pixel

    Args:
        in_length: Input length
        out_length: Output length
        scale: Scale factor
        kernel: Interpolation kernel
        k_width: Interpolation width

    Returns:
        Weights and indices of interpolating pixels
    '''
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def _imresize_mex(inimg: np.ndarray, weights: np.ndarray, indices: np.ndarray, dim: int) -> np.ndarray:
    '''
    Iterative method for image resizing

    Args:
        inimg: Input image
        weights: Interpolating weights
        indices: Indices of interpolating pixels
        dim: Dimension along which pixels are interpolated

    Returns:
        Interpolated image
    '''
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    out_img = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                out_img[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                out_img[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        out_img = np.clip(out_img, 0, 255)
        return np.around(out_img).astype(np.uint8)
    else:
        return out_img


def _imresize_vec(inimg: np.ndarray, weights: np.ndarray, indices: np.ndarray, dim: int) -> np.ndarray:
    '''
    Vectorized method for image resizing

    Args:
        inimg: Input image
        weights: Interpolating weights
        indices: Indices of interpolating pixels
        dim: Dimension along which image is interpolated

    Returns:
        Interpolated image
    '''
    w_shape = weights.shape
    if dim == 0:
        weights = weights.reshape((w_shape[0], w_shape[2], 1, 1))
        out_img =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, w_shape[0], w_shape[2], 1))
        out_img =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        out_img = np.clip(out_img, 0, 255)
        return np.around(out_img).astype(np.uint8)
    else:
        return out_img


def _resize_along_dim(A: np.ndarray, dim: int, weights: np.ndarray, indices: np.ndarray, mode: str = 'vec') -> np.ndarray:
    '''
    Wrapper around :obj:`qualitylib.tools.imresize._imresize_mex` and :obj:`qualitylib.tools.imresize._imresize_vec`

    Args:
        A: Input image
        dim: Dimension along which image is interpolated
        weights: Interpolating weights
        indices: Indices of interpolating pixels
        mode: Resize mode - 'iter' (iterative) or 'vec' (vectorized). Defaults to 'vec'.

    Returns:
        np.ndarray: _description_
    '''
    if mode == 'iter':
        out = _imresize_mex(A, weights, indices, dim)
    else:
        out = _imresize_vec(A, weights, indices, dim)
    return out


def imresize(I: np.ndarray, scalar_scale: Optional[float] = None, method:str = 'bicubic', output_shape: Optional[Tuple[int]] = None, mode: str = 'vec') -> np.ndarray:
    '''
    Resize image given scale factor or output shape

    Args:
        I: Input image
        scalar_scale: Scalar scale factor (optional). Defaults to None - resized to `output_shape`.
        method: Interpolation method. Defaults to 'bicubic'.
        output_shape: Output shape (optional). Defaults to None - inferred from `scalar_scale`.
        mode: Resize mode - 'iter' (iterative) or 'vec' (vectorized). Defaults to 'vec'.

    Raises:
        ValueError: If kernel `method` is neither 'bicubic' nor 'bilinear'.
        ValueError: If both `scalar_scale` and `output_shape` are `None`.
        ValueError: If both `scalar_scale` and `output_shape` are provided.

    Returns:
        Resized image
    '''
    if method == 'bicubic':
        kernel = _cubic
    elif method == 'bilinear':
        kernel = _triangle
    else:
        raise ValueError('unidentified kernel method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = _derive_size_from_scale(I.shape, scale)
    elif output_shape is not None:
        scale = _derive_scale_from_size(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = _contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag_2d = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag_2d = True
    for k in range(2):
        dim = order[k]
        B = _resize_along_dim(B, dim, weights[dim], indices[dim], mode)
    if flag_2d:
        B = np.squeeze(B, axis=2)
    return B
