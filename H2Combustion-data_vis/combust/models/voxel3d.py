import torch
from torch import nn
from torch.autograd import grad

from combust.layers import ScaleShift


class Voxel3D(nn.Module):
    """
    Parameters
    ----------
    shell: ucbshift3d.layers.ShellProvider

    mb_voxel: ucbshift3d.layers.ManyBodyVoxel

    mr_densenet: ucbshift3d.models.MRDenseNet

    normalizer: tuple, optional (default: (0.0, 1.0))
        A tuple of mean and standard deviation of normalizer.
        This will be used as the last leyer to inverse normalize the predictions.

    """
    def __init__(self,
                 shell,
                 mb_voxel,
                 mr_densenet,
                 mode='channelize',
                 normalizer=(0.0, 1.0),
                 device=None,
                 derivative=False,
                 create_graph=False):
        super(Voxel3D, self).__init__()
        self.shell = shell
        self.mb_voxel = mb_voxel
        self.mr_densenet = mr_densenet
        self.mode = mode

        self.derivative = derivative
        self.create_graph = create_graph

        self.inverse_normalize = ScaleShift(
            mean=torch.tensor(normalizer[0],
                              dtype=torch.float32,
                              device=device),
            stddev=torch.tensor(normalizer[1],
                                dtype=torch.float32,
                                device=device))

    def forward(self, data):
        """

        Parameters
        ----------
        data: ucbshift3d.data.BatchDataset

        Returns
        -------

        """
        # require grad
        if self.derivative:
            data.atoms.requires_grad_()

        # compute distance vectors
        distance_vector = self.shell(data.atoms, data.neighbors)

        # compute voxel representation
        boxes = self.mb_voxel(distance_vector,
                              data.atomic_numbers)  # B, A, C, G, G, G

        # MR DenseNet model
        if self.mode == 'channelize':
            # reshape boxes to 5 dimensional
            bs = boxes.size()
            boxes = boxes.view(bs[0] * bs[1], bs[2], bs[3], bs[4], bs[5])

            output = self.mr_densenet(boxes)

            # reshape output and inverse normalize
            Ei = output.view(bs[0], bs[1])
            E = torch.sum(Ei, -1)
            E = E.view(bs[0], 1)
            E = self.inverse_normalize(E)

        elif self.mode == 'embed_elemental':
            # reshape boxes to 5 dimensional
            bs = boxes.size()
            boxes = boxes.view(bs[0] * bs[1], bs[2], bs[3], bs[4], bs[5])

            output = self.mr_densenet(boxes)

            # reshape output and inverse normalize
            Ei = output.view(bs[0], bs[1])
            E = torch.sum(Ei, -1)
            E = E.view(bs[0], 1)
            E = self.inverse_normalize(E)

        elif self.mode in ['split_elemental', 'shared_elemental']:
            for ib, atomic_box in enumerate(boxes):
                # reshape boxes to 5 dimensional
                bs = atomic_box.size()
                atomic_box = atomic_box.view(bs[0] * bs[1], bs[2], bs[3], bs[4], bs[5])

                output = self.mr_densenet[ib](atomic_box)
                # energy of target atoms based on certain atom types
                if ib == 0:
                    Eia = output.view(bs[0], bs[1])
                else:
                    Eia += output.view(bs[0], bs[1])

            E = torch.sum(Eia, -1)
            E = E.view(bs[0], 1)
            E = self.inverse_normalize(E)

        # derivative
        if self.derivative:
            dE = grad(
                E,
                data.atoms,
                grad_outputs=torch.ones_like(E),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            dE = -1.0 * dE

            return E, dE

        return E, data.forces
