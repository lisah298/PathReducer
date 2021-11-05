import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch import nn
from ase.io import iread

from combust.utils import rotate_molecule, parse_reax4, parse_irc4


class BatchDataset(object):
    """
    Parameters
    ----------
    input: dict
        The dictionary of batch data in ndarray format.

    """
    def __init__(self, input, device):

        self.atoms = torch.tensor(input['atoms'], device=device, dtype=torch.float32)

        self.atomic_numbers = torch.tensor(input['atomic_numbers'],
                                           dtype=torch.long,
                                           device=device)
        self.neighbors = torch.tensor(input['neighbors'],
                                      dtype=torch.long,
                                      device=device)
        self.energy = torch.tensor(input['energy'],
                                   dtype=torch.float32,
                                   device=device)
        self.forces = torch.tensor(input['forces'],
                                   dtype=torch.float32,
                                   device=device)

    def __getitem__(self, index):

        output = dict()
        output['atoms'] = self.atoms

        output['atomic_numbers'] = self.atomic_numbers[index]
        output['neighbours'] = self.neighbors[index]
        output['energy'] = self.energy[index]
        output['forces'] = self.forces[index]

        return output

    def __len__(self):
        return self.chem_shift.size()[0]


def extensive_train_loader(dir_path,
                           env_provider,
                           data_size,
                           batch_size=32,
                           n_rotations=0,
                           freeze_rotations=False,
                           device=None,
                           shuffle=True,
                           drop_last=False):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    dir_path: str
        The path to all available xyz trj files

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.

    freeze_rotations: bool, optional (default: False)
        If True rotation angles will be determined and fixed during generation.

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    carts, atomic_numbers, energy, forces = parse_reax4(dir_path)

    # clean data
    def cleaning(inds, carts, energy, forces, task, rep=1):
        if task == 'add':
            for r in range(rep):
                carts = np.concatenate([carts, carts[inds]], axis=0)
                energy = np.concatenate([energy, energy[inds]], axis=0)
                forces = np.concatenate([forces, forces[inds]], axis=0)
        elif task == 'remove':
            carts = np.delete(carts, inds, axis=0)
            energy = np.delete(energy, inds, axis=0)
            forces = np.delete(forces, inds, axis=0)

        return carts, energy, forces

    labels = pd.read_csv('local/data/labels.csv', header=None)
    # labels: r, ts, p, ts_, o
    over_inds = labels[labels[0].map(lambda x: x in ['ts'])].index.tolist()
    carts, energy, forces = cleaning(over_inds,
                                     carts,
                                     energy,
                                     forces,
                                     'add',
                                     rep=1)
    # remove outliers
    # remove RL
    rem_inds = labels[labels[0].map(lambda x: x in ['o'])].index.tolist()
    carts, energy, forces = cleaning(rem_inds, carts, energy, forces, 'remove')

    # shuffle and select %90
    from sklearn.utils import shuffle
    inds = shuffle(list(range(carts.shape[0])), random_state=90)
    if data_size>0:
        carts = carts[inds][:data_size]
        energy = energy[inds][:data_size]
        forces = forces[inds][:data_size]
    else:
        carts = carts[inds][-data_size:]
        energy = energy[inds][-data_size:]
        forces = forces[inds][-data_size:]

    n_data = carts.shape[0]  # D
    n_atoms = carts.shape[1]  # A

    print("number of data w/o augmentation: ", n_data)

    # get neighbors
    carts, atomic_numbers, neighbors = env_provider.get_environment(
        carts, atomic_numbers)

    # # shuffle
    # if shuffle:
    #     shuffle_idx = np.arange(n_data)
    #     carts = carts[shuffle_idx]
    #     atomic_numbers = atomic_numbers[shuffle_idx]
    #     energy = energy[shuffle_idx]
    #     forces = forces[shuffle_idx]

    if freeze_rotations:
        thetas = [None
                  ]  # index 0 is reserved for the original data (no rotation)
        thetas += [
            np.random.uniform(-np.pi / 2., np.pi / 2., size=3)
            for _ in range(n_rotations)
        ]

    # iterate over data snapshots
    seen_all_data = 0
    while True:

        # iterate over rotations
        for r in range(n_rotations + 1):

            # split by batch size and yield
            data_atom_indices = list(range(n_data))

            if shuffle:
                np.random.shuffle(data_atom_indices)

            split = 0
            while (split + 1) * batch_size <= n_data:
                # Output a batch
                data_batch_idx = data_atom_indices[split *
                                                   batch_size:(split + 1) *
                                                   batch_size]

                atoms = carts[data_batch_idx]  # B, A, 3

                # rotation
                if r == 0:
                    atoms = rotate_molecule(atoms,
                                            theta=np.array([0.0, 0.0, 0.0]))
                else:
                    if freeze_rotations:
                        theta = thetas[r]
                    else:
                        theta = None
                    atoms = rotate_molecule(atoms, theta=theta)

                batch_dataset = {
                    'atoms': atoms,
                    'atomic_numbers': atomic_numbers,
                    'neighbors': neighbors,
                    'energy': energy[data_batch_idx],
                    'forces': forces[data_batch_idx]
                }
                batch_dataset = BatchDataset(batch_dataset, device=device)
                yield batch_dataset
                split += 1

            # Deal with the part smaller than a batch_size
            left_len = n_data % batch_size
            if left_len != 0 and drop_last:
                continue

            elif left_len != 0 and not drop_last:
                left_idx = data_atom_indices[split * batch_size:]

                atoms = carts[left_idx]  # B, A, 3

                # rotation
                if r == 0:
                    atoms = rotate_molecule(atoms,
                                            theta=np.array([0.0, 0.0, 0.0]))
                else:
                    atoms = rotate_molecule(atoms)

                batch_dataset = {
                    'atoms': atoms,
                    'atomic_numbers': atomic_numbers,
                    'neighbors': neighbors,
                    'energy': energy[left_idx],
                    'forces': forces[left_idx]
                }

                batch_dataset = BatchDataset(batch_dataset, device)

                yield batch_dataset

            seen_all_data += 1
            # print('\n# trained on entire data: %i (# rotation: %i)\n'%(seen_all_data, (n_rotations+1)))


def extensive_irc_loader(dir_path,
                         env_provider,
                         batch_size=32,
                         n_rotations=0,
                         freeze_rotations=False,
                         device=None,
                         shuffle=True,
                         drop_last=False):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    dir_path: str
        The path to all available xyz trj files

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.

    freeze_rotations: bool, optional (default: False)
        If True rotation angles will be determined and fixed during generation.

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """

    carts, atomic_numbers, energy, forces = parse_irc4(dir_path)

    n_data = carts.shape[0]  # D
    n_atoms = carts.shape[1]  # A

    # get neighbors
    carts, atomic_numbers, neighbors = env_provider.get_environment(
        carts, atomic_numbers)

    # # shuffle
    # if shuffle:
    #     shuffle_idx = np.arange(n_data)
    #     carts = carts[shuffle_idx]
    #     atomic_numbers = atomic_numbers[shuffle_idx]
    #     energy = energy[shuffle_idx]
    #     forces = forces[shuffle_idx]

    if freeze_rotations:
        thetas = [None
                  ]  # index 0 is reserved for the original data (no rotation)
        thetas += [
            np.random.uniform(-np.pi / 2., np.pi / 2., size=3)
            for _ in range(n_rotations)
        ]

    # iterate over data snapshots
    while True:

        # iterate over rotations
        for r in range(n_rotations + 1):

            # split by batch size and yield
            data_atom_indices = list(range(n_data))

            if shuffle:
                np.random.shuffle(data_atom_indices)

            split = 0
            while (split + 1) * batch_size <= n_data:
                # Output a batch
                data_batch_idx = data_atom_indices[split *
                                                   batch_size:(split + 1) *
                                                   batch_size]

                atoms = carts[data_batch_idx]  # B, A, 3

                # rotation
                if r == 0:
                    atoms = rotate_molecule(atoms,
                                            theta=np.array([0.0, 0.0, 0.0]))
                else:
                    if freeze_rotations:
                        theta = thetas[r]
                    else:
                        theta = None
                    atoms = rotate_molecule(atoms, theta=theta)

                batch_dataset = {
                    'atoms': atoms,
                    'atomic_numbers': atomic_numbers,
                    'neighbors': neighbors,
                    'energy': energy[data_batch_idx],
                    'forces': forces[data_batch_idx]
                }
                batch_dataset = BatchDataset(batch_dataset, device)
                yield batch_dataset
                split += 1

            # Deal with the part smaller than a batch_size
            left_len = n_data % batch_size
            if left_len != 0 and drop_last:
                continue

            elif left_len != 0 and not drop_last:
                left_idx = data_atom_indices[split * batch_size:]

                atoms = carts[left_idx]  # B, A, 3

                # rotation
                if r == 0:
                    atoms = rotate_molecule(atoms,
                                            theta=np.array([0.0, 0.0, 0.0]))
                else:
                    atoms = rotate_molecule(atoms)

                batch_dataset = {
                    'atoms': atoms,
                    'atomic_numbers': atomic_numbers,
                    'neighbors': neighbors,
                    'energy': energy[left_idx],
                    'forces': forces[left_idx]
                }

                batch_dataset = BatchDataset(batch_dataset, device)

                yield batch_dataset
