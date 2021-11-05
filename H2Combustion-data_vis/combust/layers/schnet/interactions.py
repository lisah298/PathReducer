import torch
from torch import nn

from .layers import GaussianSmearing, shifted_softplus, Dense


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """
    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # compute sum of input along axis
        y = torch.sum(input, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y


class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """
    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, f_ij, neighbours, pairwise_mask=None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(
            f_ij)  # Tensor (n_batch,max_atoms,max_neighbours,n_filters)

        # apply cutoff
        # if self.cutoff_network is not None:
        #     C = self.cutoff_network(r_ij)
        #     W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)  # Tensors (n_batch, max_atoms, n_filters)

        # pad y with zero values on dim 1 for the index -1 elements of neighbours
        # y = F.pad(y, (0,0,0,1), mode="constant", value=0)  # Tensors (n_batch, max_atoms+1, n_filters)

        # convert neighbours shape to same shape as y except for dim 1
        nbh_size = neighbours.size()  # (n_batch, max_atoms, max_neighbours)
        nbh = neighbours.view(-1, nbh_size[1] * nbh_size[2],
                              1)  # (n_batch, max_atoms*max_neighbours, 1)
        nbh = nbh.expand(
            -1, -1,
            y.size(2))  # (n_batch, max_atoms*max_neighbours, n_filters)
        y = torch.gather(y, 1,
                         nbh)  # (n_batch, max_atoms*max_neighbours, n_filters)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2],
                   -1)  # (n_batch, max_atoms, max_neighbours, n_filters)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y


class Interaction(nn.Module):
    """
    """
    def __init__(self,
                 n_atom_features=128,
                 n_atom_features_out=None,
                 n_pair_features=50,
                 n_filters=128,
                 normalize_filter=True):

        super(Interaction, self).__init__()

        # filter block
        self.filter_network = nn.Sequential(
            Dense(n_pair_features, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )

        if n_atom_features_out is None:
            n_atom_features_out = n_atom_features

        # continuous filter
        self.cfconv = CFConv(
            n_atom_features,
            n_filters,
            n_atom_features_out,
            self.filter_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_features_out,
                           n_atom_features_out,
                           bias=True,
                           activation=None)

    def forward(self, atom_features, distance_features, distances, neighbours,
                masks):
        v = self.cfconv(x=atom_features,
                        r_ij=distances,
                        f_ij=distance_features,
                        neighbours=neighbours,
                        pairwise_mask=masks)
        v = self.dense(v)
        return v


class AtomicConvolution(nn.Module):
    def __init__(self,
                 n_atom_features=128,
                 n_pair_features=50,
                 n_interactions=3,
                 n_filters=128,
                 max_z=10,
                 cutoff=7.0,
                 trainable_gaussians=False,
                 normalize_filter=True,
                 arch='schnet'):
        super(AtomicConvolution, self).__init__()

        # densenet arch
        self.arch = arch

        # embedding for atomic numbers
        self.embedding = nn.Embedding(max_z, n_atom_features, padding_idx=0)

        # how to expand distances
        self.distance_expansion = GaussianSmearing(
            0.0, cutoff, n_pair_features, trainable=trainable_gaussians)

        # how to compute interactions
        if self.arch != 'densenet_concat':
            self.interactions = nn.ModuleList([
                Interaction(n_atom_features=n_atom_features,
                            n_pair_features=n_pair_features,
                            n_filters=n_filters,
                            normalize_filter=normalize_filter)
                for _ in range(n_interactions)
            ])
        elif self.arch == 'densenet_concat':
            self.interactions = nn.ModuleList([
                Interaction(n_atom_features=n_atom_features * (it_inter + 1),
                            n_atom_features_out=n_atom_features,
                            n_pair_features=n_pair_features,
                            n_filters=n_filters,
                            normalize_filter=normalize_filter)
                for it_inter in range(n_interactions)
            ])

    def forward(self, atoms, distances, neighbours, masks):

        # get atomic features from embedding
        atom_features = self.embedding(
            atoms)  # Tensors (n_batch, max_atoms, n_atom_features)
        # print(atom_features.size())

        # expand distances
        distance_features = self.distance_expansion(
            distances)  # Tensor (n_batch,max_atoms,max_neighbours,n_gaussians)
        # print(distance_features.size())

        # compute interactions
        preceding_atom_features = [atom_features]
        for interaction in self.interactions:
            v = interaction(atom_features, distance_features, distances,
                            neighbours, masks)
            preceding_atom_features.append(v)
            if self.arch == 'densenet_sum':
                atom_features = torch.stack(preceding_atom_features,
                                            dim=-1).sum(dim=-1)
            elif self.arch == 'densenet_concat':
                atom_features = torch.cat(preceding_atom_features, dim=-1)
            elif self.arch == 'schnet':
                atom_features = atom_features + v
            elif self.arch == 'gcn':
                atom_features = v

        if self.arch == 'gcn':
            atom_features = torch.cat(preceding_atom_features, dim=-1)
        # print(atom_features.size())

        return atom_features


class AtomicDense(nn.Module):
    """"""
    def __init__(self,
                 n_in,
                 n_out=1,
                 n_hidden=[64, 32],
                 dropout=None,
                 mean=None,
                 std=None):
        super(AtomicDense, self).__init__()

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        std = torch.FloatTensor([1.0]) if std is None else std

        # normalization layer
        # self.normalize = ScaleShift(mean, std)

        # mlp layers
        neurons = zip([n_in] + n_hidden[:-1], n_hidden)
        if dropout is None:
            layers = [
                Dense(nl[0], nl[1], activation=shifted_softplus)
                for nl in neurons
            ]
        else:
            layers = []
            for nl in neurons:
                dense_layer = Dense(nl[0], nl[1], activation=shifted_softplus)
                dropout_layer = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [dense_layer, dropout_layer]
        layers.append(Dense(n_hidden[-1], n_out, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, atom_features, property_mask=None):
        """"""
        predicted_yi = self.mlp(atom_features)
        # predicted_yi = self.normalize(predicted_yi)

        # mask properties to set padded values to absolute zero
        if property_mask is not None:
            predicted_yi = predicted_yi * property_mask[..., None]

        return predicted_yi
