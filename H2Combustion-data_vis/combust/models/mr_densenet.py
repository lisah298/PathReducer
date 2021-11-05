import torch
from torch import nn
from combust.layers import Conv3D, Crop3D, Dense

# class MRDenseNetBlock(nn.Module):


class MRDenseNet(nn.Module):
    """
    The multiresolution 3D densenet model as it was developed in Shuai Liu's paper.
    Reference: https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01570

    Parameters
    ----------
    in_channels: int
        number of input channels

    dropout: float
        The dropout probability in the fully-connected layers.

    """
    def __init__(self, in_channels, dropout):
        super(MRDenseNet, self).__init__()

        self.trans1 = Conv3D(in_channels, 64, kernel_size=3, padding=1)

        # block1
        self.b1_u1_c1 = Conv3D(64, 256, kernel_size=1, padding=0)
        self.b1_u1_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b1_u1_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b1_u2_c1 = Conv3D(128, 256, kernel_size=1, padding=0)
        self.b1_u2_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b1_u2_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b1_u3_c1 = Conv3D(192, 256, kernel_size=1, padding=0)
        self.b1_u3_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b1_u3_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b1_u4_c1 = Conv3D(256, 256, kernel_size=1, padding=0)
        self.b1_u4_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b1_u4_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.avg_pool_b1 = nn.AvgPool3d(kernel_size=2)
        self.crop_b1 = Crop3D(4)

        self.b1_trans1 = Conv3D(128, 256, kernel_size=1, padding=0)
        self.b1_trans2 = Conv3D(256, 64, kernel_size=1, padding=0)

        self.trans2 = Conv3D(64, 64, kernel_size=1, padding=0)

        # block2
        self.b2_u1_c1 = Conv3D(64, 256, kernel_size=1, padding=0)
        self.b2_u1_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b2_u1_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b2_u2_c1 = Conv3D(128, 256, kernel_size=1, padding=0)
        self.b2_u2_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b2_u2_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b2_u3_c1 = Conv3D(192, 256, kernel_size=1, padding=0)
        self.b2_u3_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b2_u3_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.b2_u4_c1 = Conv3D(256, 256, kernel_size=1, padding=0)
        self.b2_u4_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        self.b2_u4_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.avg_pool_b2 = nn.AvgPool3d(kernel_size=2)
        self.crop_b2 = Crop3D(2)

        self.b2_trans1 = Conv3D(128, 256, kernel_size=1, padding=0)
        self.b2_trans2 = Conv3D(256, 64, kernel_size=1, padding=0)

        # dense block
        self.dense1 = Dense(64 * 4 * 4 * 4,
                            32,
                            activation=nn.ReLU(),
                            dropout=dropout,
                            norm=True)
        self.dense2 = Dense(32,
                            16,
                            activation=nn.ReLU(),
                            dropout=dropout,
                            norm=True)
        self.dense3 = Dense(16, 1, activation=None, dropout=None, norm=None)

    def forward(self, x):
        # first transition
        x = self.trans1(x)

        # block 1
        x1 = self.b1_u1_c1(x)
        x1 = self.b1_u1_c2(x1)
        x1 = self.b1_u1_c3(x1)

        x2 = torch.cat((x, x1), dim=1)  # along channels
        x2 = self.b1_u2_c1(x2)
        x2 = self.b1_u2_c2(x2)
        x2 = self.b1_u2_c3(x2)

        x3 = torch.cat((x, x1, x2), dim=1)  # along channels
        x3 = self.b1_u3_c1(x3)
        x3 = self.b1_u3_c2(x3)
        x3 = self.b1_u3_c3(x3)

        x4 = torch.cat((x, x1, x2, x3), dim=1)  # along channels
        x4 = self.b1_u4_c1(x4)
        x4 = self.b1_u4_c2(x4)
        x4 = self.b1_u4_c3(x4)

        x1 = self.avg_pool_b1(x4)
        x2 = self.crop_b1(x4)
        x = torch.cat((x1, x2), dim=1)  # along channels

        x = self.b1_trans1(x)
        x = self.b1_trans2(x)

        # second transition
        x = self.trans2(x)

        # block 2
        x1 = self.b2_u1_c1(x)
        x1 = self.b2_u1_c2(x1)
        x1 = self.b2_u1_c3(x1)

        x2 = torch.cat((x, x1), dim=1)  # along channels
        x2 = self.b2_u2_c1(x2)
        x2 = self.b2_u2_c2(x2)
        x2 = self.b2_u2_c3(x2)

        x3 = torch.cat((x, x1, x2), dim=1)  # along channels
        x3 = self.b2_u3_c1(x3)
        x3 = self.b2_u3_c2(x3)
        x3 = self.b2_u3_c3(x3)

        x4 = torch.cat((x, x1, x2, x3), dim=1)  # along channels
        x4 = self.b2_u4_c1(x4)
        x4 = self.b2_u4_c2(x4)
        x4 = self.b2_u4_c3(x4)

        x1 = self.avg_pool_b2(x4)
        x2 = self.crop_b2(x4)
        x = torch.cat((x1, x2), dim=1)  # along channels

        x = self.b2_trans1(x)
        x = self.b2_trans2(x)

        # flatten layer
        x = torch.flatten(x, start_dim=1)

        # dense block
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)

        return y


class Testnet(nn.Module):
    """
    The multiresolution 3D densenet model as it was developed in Shuai Liu's paper.
    Reference: https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01570

    Parameters
    ----------
    in_channels: int
        number of input channels

    dropout: float
        The dropout probability in the fully-connected layers.

    """
    def __init__(self, in_channels, dropout):
        super(Testnet, self).__init__()

        self.trans1 = Conv3D(in_channels, 16, kernel_size=3, padding=1)

        # block1
        self.b1_u1_c1 = Conv3D(16, 32, kernel_size=1, padding=0)
        self.b1_u1_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b1_u1_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        self.b1_u2_c1 = Conv3D(32, 32, kernel_size=1, padding=0)
        self.b1_u2_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b1_u2_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        self.b1_u3_c1 = Conv3D(48, 32, kernel_size=1, padding=0)
        self.b1_u3_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b1_u3_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        # self.b1_u4_c1 = Conv3D(256, 256, kernel_size=1, padding=0)
        # self.b1_u4_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        # self.b1_u4_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.avg_pool_b1 = nn.AvgPool3d(kernel_size=2)
        self.crop_b1 = Crop3D(4)

        self.b1_trans1 = Conv3D(32, 32, kernel_size=1, padding=0)
        self.b1_trans2 = Conv3D(32, 16, kernel_size=1, padding=0)

        self.trans2 = Conv3D(16, 16, kernel_size=1, padding=0)

        # block2
        self.b2_u1_c1 = Conv3D(16, 32, kernel_size=1, padding=0)
        self.b2_u1_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b2_u1_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        self.b2_u2_c1 = Conv3D(32, 32, kernel_size=1, padding=0)
        self.b2_u2_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b2_u2_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        self.b2_u3_c1 = Conv3D(48, 32, kernel_size=1, padding=0)
        self.b2_u3_c2 = Conv3D(32, 16, kernel_size=1, padding=0)
        self.b2_u3_c3 = Conv3D(16, 16, kernel_size=3, padding=1)

        # self.b2_u4_c1 = Conv3D(256, 256, kernel_size=1, padding=0)
        # self.b2_u4_c2 = Conv3D(256, 64, kernel_size=1, padding=0)
        # self.b2_u4_c3 = Conv3D(64, 64, kernel_size=3, padding=1)

        self.avg_pool_b2 = nn.AvgPool3d(kernel_size=2)
        self.crop_b2 = Crop3D(2)

        self.b2_trans1 = Conv3D(32, 32, kernel_size=1, padding=0)
        self.b2_trans2 = Conv3D(32, 16, kernel_size=1, padding=0)

        # dense block
        self.dense1 = Dense(16 * 4 * 4 * 4,
                            128,
                            activation=nn.ReLU(),
                            dropout=dropout,
                            norm=True)
        self.dense2 = Dense(128,
                            64,
                            activation=nn.ReLU(),
                            dropout=dropout,
                            norm=True)
        self.dense3 = Dense(64, 1, activation=None, dropout=None, norm=None)

    def forward(self, x):
        # first transition
        x = self.trans1(x)

        # block 1
        x1 = self.b1_u1_c1(x)
        x1 = self.b1_u1_c2(x1)
        x1 = self.b1_u1_c3(x1)

        x2 = torch.cat((x, x1), dim=1)  # along channels
        x2 = self.b1_u2_c1(x2)
        x2 = self.b1_u2_c2(x2)
        x2 = self.b1_u2_c3(x2)

        x3 = torch.cat((x, x1, x2), dim=1)  # along channels
        x3 = self.b1_u3_c1(x3)
        x3 = self.b1_u3_c2(x3)
        x3 = self.b1_u3_c3(x3)

        # x4 = torch.cat((x, x1, x2, x3), dim=1)  # along channels
        # x4 = self.b1_u4_c1(x4)
        # x4 = self.b1_u4_c2(x4)
        # x4 = self.b1_u4_c3(x4)
        #
        x1 = self.avg_pool_b1(x3)
        x2 = self.crop_b1(x3)
        x = torch.cat((x1, x2), dim=1)  # along channels

        x = self.b1_trans1(x)
        x = self.b1_trans2(x)

        # second transition
        x = self.trans2(x)

        # block 2
        x1 = self.b2_u1_c1(x)
        x1 = self.b2_u1_c2(x1)
        x1 = self.b2_u1_c3(x1)

        x2 = torch.cat((x, x1), dim=1)  # along channels
        x2 = self.b2_u2_c1(x2)
        x2 = self.b2_u2_c2(x2)
        x2 = self.b2_u2_c3(x2)

        x3 = torch.cat((x, x1, x2), dim=1)  # along channels
        x3 = self.b2_u3_c1(x3)
        x3 = self.b2_u3_c2(x3)
        x3 = self.b2_u3_c3(x3)

        # x4 = torch.cat((x, x1, x2, x3), dim=1)  # along channels
        # x4 = self.b2_u4_c1(x4)
        # x4 = self.b2_u4_c2(x4)
        # x4 = self.b2_u4_c3(x4)
        #
        x1 = self.avg_pool_b2(x3)
        x2 = self.crop_b2(x3)
        x = torch.cat((x1, x2), dim=1)  # along channels

        x = self.b2_trans1(x)
        x = self.b2_trans2(x)

        # flatten layer
        x = torch.flatten(x, start_dim=1)

        # dense block
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)

        return y
