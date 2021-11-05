from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_


class Conv3D(nn.Conv3d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 bias=True,
                 activation=nn.ReLU(),
                 weight_init=xavier_uniform_,
                 bias_init=zeros_,
                 dropout=None,
                 norm=True):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Conv3D, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     bias=bias)
        self.activation = activation
        # initialize linear layer y = xW^T + b

        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.norm = norm
        if norm:
            self.norm = nn.BatchNorm3d(
                num_features=out_channels
            )  #, momentum=0.99, eps=0.001) # momentum and eps are based on Keras default values

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x):
        # compute linear layer y = xW^T + b
        x = super(Conv3D, self).forward(x)

        # batch normalization
        if self.norm:
            x = self.norm(x)

        # add activation function
        if self.activation:
            x = self.activation(x)

        # dropout
        if self.dropout:
            x = self.dropout(x)

        return x
