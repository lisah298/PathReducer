"""
Miscellaneous utility functions
"""

import os
import pandas as pd
import numpy as np
from ase import Atoms
from ase.io import iread
from ase.io.xyz import read_xyz

def read_qchem_files(path, skiprows=(0, 0)):
    """
    reads QChem files: tested for Energy, NucCarts, and NucForces

    Parameters
    ----------
    path: str
        path to the file

    skiprows: tuple
        This must be a tuple of two integers.
        The first element indicates number of lines to skip from top,
        and the second element is the number of lines at bottom of file to skip.

    Returns
    -------
    arraylike: 2D array of corresponding data

    """

    # read energy file to dataframe
    df = pd.read_csv(path,
                     skiprows=skiprows[0],
                     skipfooter=skiprows[1],
                     delim_whitespace=True,
                     header=None,
                     engine='python')

    return df.values


def parse_reax4(dir_path):
    """
    The main parser function only good for reax4
    Todo: need to become generalized

    Parameters
    ----------
    dir_path

    Returns
    -------

    """
    n_atoms = 4

    energy = []
    forces = []
    carts = []

    for length in ['short', 'long', 'very_short', 'short/more']:
        for temp in [500, 1000, 2000, 3000]:
            if length == 'very_short':
                n_files = 20
            else:
                n_files = 10
            for it in range(1, n_files + 1):
                path = os.path.join(dir_path, str(temp), length, str(it),
                                    'AIMD')

                # read files, skip first row: header line
                tmp_energy = read_qchem_files(os.path.join(path, 'TandV'),
                                              (1, 0))
                tmp_forces = read_qchem_files(os.path.join(path, 'NucForces'),
                                              (1, 0))
                tmp_carts = read_qchem_files(os.path.join(path, 'NucCarts'),
                                             (1, 0))

                # sanity check
                if length in ['short', 'short/more']:
                    n_data = 50
                elif length == 'very_short':
                    n_data = 25
                else:
                    n_data = 100

                assert tmp_energy.shape[0] == tmp_forces.shape[0] == n_data
                assert tmp_energy.shape[1] == 5
                assert tmp_forces.shape[1] == n_atoms * 3 + 1

                assert tmp_carts.shape[0] == tmp_energy.shape[
                    0] + 1  # cartesian coordinates has one extra time step
                assert tmp_carts.shape[1] == n_atoms * 3 + 1
                tmp_carts = tmp_carts[:-1, :]

                # check time steps
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_forces[:, 0],
                                               decimal=4)
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_carts[:, 0],
                                               decimal=4)

                # update data collection
                energy.append(tmp_energy)
                forces.append(tmp_forces)
                carts.append(tmp_carts)

    # concatenate numpy arrays
    energy = np.concatenate(energy, axis=0)
    forces = np.concatenate(forces, axis=0)
    carts = np.concatenate(carts, axis=0)

    # atomic energies (au)
    H = -0.5004966690 * 627.5
    O = -75.0637742413 * 627.5

    # units ( au to kcal/mol)
    energy = energy * 627.5 - 2 * H - 2 * O  # relative energy
    forces = forces * 1182.683

    energy = energy[:, 1].reshape(-1, 1)
    forces = forces[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    carts = carts[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    atomic_numbers = np.array([1, 8, 1, 8]).reshape(-1, 1)

    return carts, atomic_numbers, energy, forces


def parse_irc(dir_path):
    """
    The main parser function

    Parameters
    ----------
    full_path

    Returns
    -------

    """
    #get information on atoms from random file
    for f in os.listdir(dir_path):
       if 'energy' in f:
           energy_file = f 
       if 'rpath' in f:
           structure_file = f
       if 'gradient' in f:
           gradient_file = f


    atom_reader = iread(os.path.join(dir_path,structure_file))
    atoms = next(atom_reader)
    n_atoms = atoms.get_atomic_numbers().size

    # energy
    ef = pd.read_csv(os.path.join(dir_path,energy_file), header=None)

    # atomic energies (au)
    H = -0.5004966690 * 627.509
    O = -75.0637742413 * 627.509

    nr_H = atoms.get_chemical_symbols().count('H')
    nr_O = atoms.get_chemical_symbols().count('O')

    # units ( au to kcal/mol)
    irc_energy = ef[0].values * 627.509 - nr_H * H - nr_O * O  # relative energy
    irc_energy = irc_energy.reshape(-1, 1)
    assert irc_energy.ndim == 2
    assert irc_energy.shape[-1] == 1

    # carts
    irc_carts = iread(os.path.join(dir_path,structure_file))
    irc_carts = np.array([i.positions for i in irc_carts])
    assert irc_carts.shape[-1] == 3
    assert irc_carts.shape[-2] == n_atoms

    # forces
    irc_forces = pd.read_csv(os.path.join(dir_path,gradient_file), header=None).values
    irc_forces = irc_forces.reshape(irc_forces.shape[0], n_atoms, 3)

    atomic_numbers = atoms.get_atomic_numbers().reshape(-1, 1)

    return irc_carts, atomic_numbers, irc_energy, irc_forces



def parse_reax(dir_path):
    """
    The main parser function
    This will work for all reactions with the directory structure as provided
    by our submission script.
    ToDo: similar function for reaction with 2 spin states

    Parameters
    ----------
    dir_path

    Returns
    -------

    """
    #get information on atoms from random file
    init_path = os.path.join(dir_path,'500','short','1','AIMD','View.xyz')
    testfile = open(init_path,'r')
    n_atoms = int(testfile.readline())
    testfile.readline()
    atoms = []
    nr_H = 0
    nr_O = 0
    for i in range(n_atoms):
       atom = testfile.readline().split()[0]
       if atom == 'H':
          nr_H += 1
          atoms.append(1)
       elif atom == 'O':
          nr_O += 1
          atoms.append(8)
       else:
          print('Element not relevant for hydrogen combustion.')
    testfile.close()

    energy = []
    forces = []
    carts = []

    for length in ['short', 'long', 'very_short']:
        for temp in [500, 1000, 2000, 3000]:
            if length == 'long':
                n_files = 10
            else:
                n_files = 20
            for it in range(1, n_files + 1):
                path = os.path.join(dir_path, str(temp), length, str(it),
                                    'AIMD')

                # read files, skip first row: header line
                tmp_energy = read_qchem_files(os.path.join(path, 'TandV'),
                                              (1, 0))
                tmp_forces = read_qchem_files(os.path.join(path, 'NucForces'),
                                              (1, 0))
                tmp_carts = read_qchem_files(os.path.join(path, 'NucCarts'),
                                             (1, 0))

                # sanity check
                if length == 'short':
                    n_data = 50
                elif length == 'very_short':
                    n_data = 25
                else:
                    n_data = 100

                assert tmp_energy.shape[0] == tmp_forces.shape[0] == n_data
                assert tmp_energy.shape[1] == 5
                assert tmp_forces.shape[1] == n_atoms * 3 + 1

                assert tmp_carts.shape[0] == tmp_energy.shape[
                    0] + 1  # cartesian coordinates has one extra time step
                assert tmp_carts.shape[1] == n_atoms * 3 + 1
                tmp_carts = tmp_carts[:-1, :]

                # check time steps
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_forces[:, 0],
                                               decimal=4)
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_carts[:, 0],
                                               decimal=4)

                # update data collection
                energy.append(tmp_energy)
                forces.append(tmp_forces)
                carts.append(tmp_carts)

    # concatenate numpy arrays
    energy = np.concatenate(energy, axis=0)
    forces = np.concatenate(forces, axis=0)
    carts = np.concatenate(carts, axis=0)

    # atomic energies (au)
    H = -0.5004966690 * 627.509
    O = -75.0637742413 * 627.509

    # units ( au to kcal/mol)
    energy = energy * 627.509 - nr_H * H - nr_O * O  # relative energy
    forces = forces * 1182.683

    energy = energy[:, 1].reshape(-1, 1)
    forces = forces[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    carts = carts[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    atomic_numbers = np.array(atoms).reshape(-1, 1)

    return carts, atomic_numbers, energy, forces

def parse_nn_disp(dir_path):
#start only with second structure (first is just IRC structure)
   count = 0
   e_count = 0
   first = True
   structure_count = 0
   atoms = []
   energies = []

   for folder in os.listdir(dir_path):
      path = os.path.join(dir_path,folder)
      if os.path.isdir(path):
         for f in os.listdir(path):
            if first:
               testfile = open(os.path.join(path,f),'r')
               n_atoms = int(testfile.readline())
               testfile.readline()
               nr_H = 0
               nr_O = 0
               for i in range(n_atoms):
                  atom = testfile.readline().split()[0]
                  if atom == 'H':
                     print("H")
                     nr_H += 1
                  elif atom == 'O':
                     print("O")
                     nr_O += 1
                  else:
                     print('Element not relevant for hydrogen combustion.')
               testfile.close()
               first=False 
               
            f_ob = open(os.path.join(path,f))
            e_count = 0
            for line in f_ob:
               if 'E=' in line:
                  e_count += 1
                  if e_count > 1: 
                     energies.append(float(line.split()[1][2:]))
            f_ob.close()
            count = count + 1
            atom_reader = iread(os.path.join(path,f))  
            next(atom_reader)
            while True:
               try: 
                  atoms.append(next(atom_reader))
                  structure_count = structure_count + 1
               except StopIteration:
                  break

   print("There are "+str(count)+" files and "+str(structure_count)+" structures.")


   # atomic energies (au)
   H = -0.5004966690 * 627.509
   O = -75.0637742413 * 627.509

   # units ( au to kcal/mol)
   energies = [i * 627.509 - nr_H * H - nr_O * O for i in energies]  # relative energy
   energies = np.array(energies)

   atoms_carts = np.array([i.positions for i in atoms])
   assert atoms_carts.shape[-1] == 3
   assert atoms_carts.shape[-2] == atoms[0].get_atomic_numbers().size

   atomic_numbers = atoms[0].get_atomic_numbers().reshape(-1, 1)

   return atoms_carts, atomic_numbers, energies


