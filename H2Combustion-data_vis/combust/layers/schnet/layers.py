import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_

from functools import partial

zeros_initializer = partial(constant_, val=0.0)


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)


def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """
    def __init__(self,
                 start=0.0,
                 stop=5.0,
                 n_gaussians=50,
                 centered=False,
                 trainable=False):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor(
            (offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(distances,
                                 self.offsets,
                                 self.width,
                                 centered=self.centered)


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        # initialize linear layer y = xW^T + b
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y
