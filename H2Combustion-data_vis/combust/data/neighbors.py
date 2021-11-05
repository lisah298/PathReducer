"""
A compilation of modules that help to find closest neighbours of each atom in a molecule.
Each Molecule is represented as a dictionary with following keys:
    - atoms: atomic positions with shape (n_atom, 3)
    - z: atomic numbers with shape (n_atoms, 1)
    - cell: unit cell with shape (3,3)
    - atom_prop: atomic property with shape (n_atoms, n_atomic_prop)
    - mol_prop: molecular property with shape (1, n_mol_prop)

"""
import numpy as np
from ase.neighborlist import neighbor_list

from combust.utils import padaxis


class Molecule(object):
    """
    The Molecule class
    """
    def __init__(self, atoms, z, cell, atom_prop, mol_prop):
        self.atoms = atoms
        self.z = z
        self.cell = cell
        self.atom_prop = atom_prop
        self.mol_prop = mol_prop


class ExtensiveEnvironment(object):
    """
    Provide atomic environment of an array of atoms and their atomic numbers.
    No cutoff, No periodic boundary condition

    Parameters
    ----------
    max_n_neighbors: int, optional (default: None)
        maximum number of neighbors to pad arrays if they have less elements
        if None, it will be ignored (e.g., in case all atoms have same length)

    """
    def __init__(self, max_n_neighbors=None):
        if max_n_neighbors is None:
            max_n_neighbors = 0
        self.max_n_neighbors = max_n_neighbors

    def get_environment(self, carts, atomic_numbers):
        """

        Parameters
        ----------
        carts: ndarray
            A 3D array of atomic positions in XYZ coordinates with shape (D, A, 3), where
            D is number of snapshots and A is number of atoms per data point

        atomic_numbers: ndarray
            A 2D array of atomic numbers with shape (A, 1)

        Returns
        -------
        ndarray: 2D array of atomic positions with shape (D, A, 3)
        ndarray: 2D array of atomic numbers with shape (A-1, N), where N is max_n_neighbors, N>=(A-1) (padded with zeros)
        ndarray: 2D array of neighbors with shape (A-1, N), (padded with -1)

        """
        n_data = carts.shape[0]  # D
        n_atoms = carts.shape[1]  # A
        if n_atoms != atomic_numbers.shape[0]:
            raise ValueError(
                "@SimpleEnvironment: atoms and atomic_numbers must have same length."
            )

        # 2d array of all indices for all atoms in a single data point
        N = np.tile(np.arange(n_atoms), (n_atoms, 1))  # (A, A)

        # remove the diagonal self indices
        neighbors = N[~np.eye(n_atoms, dtype=bool)].reshape(n_atoms,
                                                            -1)  # (A, A-1)
        # neighbors = np.tile(neighbors, (n_data, 1))     # (D*A, A-1)
        # neighbors = neighbors.reshape(n_data, n_atoms, n_atoms-1)  # (D, A, A-1)

        # atomic numbers
        atomic_numbers = atomic_numbers[neighbors]  # (A, A-1, 1)
        atomic_numbers = atomic_numbers.reshape(n_atoms,
                                                n_atoms - 1)  # (A, A-1)

        if n_atoms < self.max_n_neighbors:
            neighbors = padaxis(neighbors,
                                self.max_n_neighbors,
                                axis=-1,
                                pad_value=-1)  # (D, A, N)
            atomic_numbers = padaxis(atomic_numbers,
                                     self.max_n_neighbors,
                                     axis=-1,
                                     pad_value=0)  # (D, A, N)

        return carts, atomic_numbers, neighbors
