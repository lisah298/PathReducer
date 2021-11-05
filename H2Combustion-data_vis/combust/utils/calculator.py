import os
import numpy as np
import time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm

def fermi_dirac(x, mu, sigma):
    """
    The Fermi-Dirac distribution with simplified hyperparameters.

    Parameters
    ----------
    x: ndarray
        In this case the array of distances (e.g., OH distances)

    mu: flaot
        equivalent to the total chemical potential

    sigma: float
        equivalent to the 1/kT
        CJS: the equilibrium bond length between two atoms

    Returns
    -------
    ndarray: The F-D distribution
    CJS: I multiplied it by 2 to stretch the distribution and have the inflection point at 1
    """
    return 2.0/(1+np.exp(sigma * (x-mu)))


def coord_num(arrays, mu, sigma):
    """
    This function compute the Fermi Dirac distributions of
    each input array and returns sum of all values.

    Parameters
    ----------
    arrays: list
        A list of 1D arrays with same shape.

    mu: float
        The Fermi-Dirac parameter equivalent to the total chemical potential

    sigma: float
        The Fermi-Dirac parameterequivalent to the 1/kT

    Returns
    -------
    ndarray: The coordination number in same shape as each of the input arrays.

    Examples
    --------
    # example based on reaction #4
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.spatial import distance_matrix
    >>> from combust.utils import parse_reax4
    >>> from combust.utils import coord_num, cn_visualize2D

    >>> data_path = "local_path_to/AIMD/04/combined/"
    >>> carts, atomic_numbers, energy, forces = parse_reax4(data_path)
    >>> carts = carts.reshape(carts.shape[0], 12)

    >>> def get_distances(atoms, n_atoms):
    >>>     atoms = atoms.reshape(n_atoms, 3)
    >>>     dist = distance_matrix(atoms, atoms, p=2)
    >>>     return dist[np.triu_indices(n_atoms, k=1)]

    >>> dist = np.apply_along_axis(get_distances, 1, carts, n_atoms=4)
    >>> dist = pd.DataFrame(dist, columns=['H1_O1', 'H1_H2', 'H1_O2', 'O1_H2', 'O1_O2', 'H2_O2'])
    >>> cn1 = coord_num([dist.H1_O1.values, dist.O1_H2.values], mu=1.0, sigma=3.0)
    >>> cn2 = coord_num([dist.H1_O2.values, dist.H2_O2.values], mu=1.0, sigma=3.0)

    >>> cn_visualize2D(cn1, cn2, "cn_figs")

    """
    cn = 0
    for array in arrays:
        cn += fermi_dirac(array, mu, sigma)

    return cn


def clean_data(cn1_aimd, cn2_aimd, energies_aimd, cn1_nn_disp, cn2_nn_disp, energies_nn_disp, ts_energy):

   tic = time.perf_counter()
   #FIRST STEP: remove data that is too high in energy
   removed_aimd = 0
   removed_nn_disp = 0

   it = len(energies_aimd)-1
   while it >= 0:
      if energies_aimd[it] > ts_energy+10:
         energies_aimd = np.delete(energies_aimd, it)
         cn1_aimd = np.delete(cn1_aimd,it)
         cn2_aimd = np.delete(cn2_aimd,it)
         removed_aimd += 1
      it -= 1

   it = len(energies_nn_disp)-1
   while it >= 0:
      if energies_nn_disp[it] > ts_energy+10:
         energies_nn_disp = np.delete(energies_nn_disp, it)
         cn1_nn_disp = np.delete(cn1_nn_disp,it)
         cn2_nn_disp = np.delete(cn2_nn_disp,it)
         removed_nn_disp += 1
      it -= 1

   #SECOND STEP: rank data according to maximum distance in CN space
   energies = np.concatenate([energies_aimd,energies_nn_disp])
   cn1 = np.concatenate([cn1_aimd,cn1_nn_disp])
   cn2 = np.concatenate([cn2_aimd,cn2_nn_disp])
   

   #calculate pair potentials
   pp = np.zeros((len(cn1),len(cn1)))
   count = 0
   for i in range(len(cn1)):
      count += 1
      if count == 500:
         print("500 rows")
         count = 0
      for j in range(i,len(cn1)):
         pp[i,j] = dist(cn1[i],cn2[i],cn1[j],cn2[j])
         pp[j,i] = pp[i,j]

   print("Finished calculating pair potentials")

   toc = time.perf_counter()

   # welcher datenpunkt is am weitesten von allen bereits beruecksichtigten Datenpunkten entfernt?
   selected_data = []

   #try zipped lists to keep information about indices
   indices = range(len(cn1))
   zipped_cn1 = zip(cn1,indices)
   zipped_cn2 = zip(cn2,indices)
   zipped_energies = zip(energies,indices)
  
   #backconversion to list to allow for indexing
   cn1 = list(zipped_cn1)
   cn2 = list(zipped_cn2)
   energies = list(zipped_energies)

   # Start von beliebigem Punkt
   first_index = 0
   selected_data.append([cn1[first_index][0],cn2[first_index][0],energies[first_index][0],first_index])

   #entferne diesen Punkt aus alten arrays
   del cn1[first_index]
   del cn2[first_index]
   del energies[first_index]

   # Finde Datenpunkt der maximal entfernt ist
   min_repulsion = []
   count = 0

   min_dist_sel = []
   for i in range(len(cn1)):
      min_dist_sel.append(pp[cn1[i][1],selected_data[0][3]])

   min_dist_sel = np.asarray(min_dist_sel)


   while len(cn1) > 0:
      print(str(len(cn1))+' elements in unused data')
      max_index = min_dist_sel.argmax()
      min_repulsion.append(min_dist_sel[max_index])
      new_data = [cn1[max_index][0],cn2[max_index][0],energies[max_index][0],cn1[max_index][1]]
      selected_data.append(new_data)
      
      del cn1[max_index]
      del cn2[max_index]
      del energies[max_index]
      min_dist_sel = np.delete(min_dist_sel,max_index)

      for i in range(len(cn1)):
         if pp[cn1[i][1],new_data[3]] < min_dist_sel[i]:
            min_dist_sel[i] = pp[cn1[i][1],new_data[3]]
            
      count += 1
      if count == 12000:
         count = 0
         plot_data_distribution(min_repulsion)
         cn_vis_fragment(selected_data,ts_energy)
      
   print(f"Removal of high energy data and calculation of pair potentials took {toc - tic:0.4f} seconds")
   toc2 = time.perf_counter()
   print(f"The full furthest point sort algorithm took {toc2 - tic:0.4f} seconds")

   return cn1_aimd, cn2_aimd, energies_aimd, cn1_nn_disp, cn2_nn_disp, energies_nn_disp

def harmonic(first_cn1,first_cn2,second_cn1,second_cn2):
   #alternativ:harmonic potential
   dist = np.sqrt((first_cn1-second_cn1)**2+(first_cn2-second_cn2)**2)
   return (dist-0.8)**2


def dist(first_cn1,first_cn2,second_cn1,second_cn2):
   return np.sqrt((first_cn1-second_cn1)**2+(first_cn2-second_cn2)**2)


def plot_data_distribution(max_distance):
    #general settings plot
   plt.style.use('seaborn-whitegrid')
   plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
   plt.rcParams['font.family'] = "sans-serif"
   plt.rcParams['font.size'] = 18

   # visualization
   fig, ax = plt.subplots()
   size = fig.get_size_inches()
   fig.set_size_inches(size*2)

   x_data = range(len(max_distance))

   ax.scatter(x_data, max_distance)
   plt.xlabel("# data point")
   plt.ylabel("average harmonic repulsion")
   plt.title("Data diversity analysis")

   plt.tight_layout()
   plt.show()

def cn_vis_fragment(selected_data, ts_energy):
   cn1 = []
   cn2 = []
   energies = []   
   for el in selected_data:
      cn1.append(el[0])
      cn2.append(el[1])
      energies.append(el[2])

    #general settings plot
   plt.style.use('seaborn-whitegrid')
   plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
   plt.rcParams['font.family'] = "sans-serif"
   plt.rcParams['font.size'] = 18

   # visualization
   fig, ax = plt.subplots()
   size = fig.get_size_inches()
   fig.set_size_inches(size*2)
   viridis = cm.get_cmap('viridis')

   comb = ax.scatter(cn2, cn1, s=15, c=energies, marker='.', cmap=viridis, vmin=min(energies),vmax=ts_energy+10.0)
   #plt.xlabel(xtitle)
   #plt.ylabel(ytitle)
   #plt.title(title)


   cbar = fig.colorbar(comb, ax=ax)
   cbar.set_label(r'$\Delta$ E / kcal mol$^{-1}$', rotation=270, labelpad=20)
   #ax.legend(loc='upper right',fontsize='large', frameon=True)
   plt.tight_layout()
   
   plt.show()



def cn_visualize2D(cn1, cn2, energies, cn1_ref, cn2_ref, energies_ref, output_path, title, xtitle, ytitle, cn1_nn_disp, cn2_nn_disp, energies_nn_disp):
    """ 
    A quick 2D scatter plot with point density, mostly
    hard-coded for coordination numbers (cn).

    Parameters
    ----------
    cn1: ndarray
        1D array of the first coordination numbers (reactant).

    cn2: ndarray
        1D array of the second coordination numbers (product).
 
    energies: ndarray
        1D array of the corresponding potential energies.

    cn1_ref: ndarray
        1D array of the first coordination numbers for IRC (reactant).
    
    cn2_ref: ndarray
        1D array of the second coordination numbers for IRC (product).

    energies: ndarray
        1D array of the corresponding potential energies for IRC.

    output_path: str
        The full path to the output directory.
        will be created if it doesn't exist.

    title: str
        Title for the plots. Should be reaction number + Lewis equation for this reaction

    xtitle: str
        Label for xaxis. Should specify the second coordination number (product).

    ytitle: str
        Label for yaxis. Should specify the first coordination number (reactant).

    """
    #general settings
    plt.style.use('seaborn-whitegrid')
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = 18

    #FIRST PLOT: color coded density

    # Calculate the point density
    combined = np.vstack([cn1, cn2])
    z = gaussian_kde(combined)(combined)

    # visualization
    fig, ax = plt.subplots()
    size = fig.get_size_inches()
    fig.set_size_inches(size*2)
    viridis = cm.get_cmap('viridis')

    ax.scatter(cn2, cn1, s=20, c=z, marker='.',cmap=viridis)
    irc_lines = ax.plot(cn2_ref, cn1_ref, color='black', marker='o',linestyle='dashed', markersize=4, fillstyle='full',label='IRC')

    R = ax.scatter(cn2_ref[0], cn1_ref[0],c='red',s=65,marker='^',label='R')
    ts = ax.scatter(cn2_ref[energies_ref.argmax()], cn1_ref[energies_ref.argmax()],c='red',s=65,marker='*',label='TS')
    P = ax.scatter(cn2_ref[-1], cn1_ref[-1],c='red',s=65,marker='v',label='P')
    
    #make sure these are displayed on the top layer
    R.set_zorder(19)
    ts.set_zorder(20)
    P.set_zorder(18)

    #normal mode displacement
    ax.scatter(cn2_nn_disp,cn1_nn_disp, marker='o',color='blue',label="normal mode")
    
    ax.legend(loc='upper right',fontsize='large', frameon=True)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)

    plt.tight_layout()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig.savefig(os.path.join(output_path, "cn_density.eps"), close=True, verbose=True)
    fig.savefig(os.path.join(output_path, "cn_density.png"), close=True, verbose=True)
    plt.close(fig)

    #SECOND PLOT: color coded energy
    fig2, ax2 = plt.subplots()
    size = fig2.get_size_inches()
    fig2.set_size_inches(size*2)
    viridis = cm.get_cmap('viridis')
    viridis.set_over('red')

    aimd = ax2.scatter(cn2, cn1, s=5, c=energies, marker='.', cmap=viridis, vmin=min(energies),vmax=max(energies_ref)+10.0)
    irc_lines = ax2.plot(cn2_ref, cn1_ref, color='black', marker='o',linestyle='dashed', markersize=4, fillstyle='full',label='IRC')

    R = ax2.scatter(cn2_ref[0], cn1_ref[0],c='red',s=65,marker='^',label='R')
    ts = ax2.scatter(cn2_ref[energies_ref.argmax()], cn1_ref[energies_ref.argmax()],c='red',s=65,marker='*',label='TS')
    P = ax2.scatter(cn2_ref[-1], cn1_ref[-1],c='red',s=65,marker='v',label='P')

    #make sure these are displayed on the top layer
    R.set_zorder(19)
    ts.set_zorder(20)
    P.set_zorder(18)

    #normal mode displacement
    nn_disp = ax2.scatter(cn2_nn_disp, cn1_nn_disp, c=energies_nn_disp, marker='o',cmap=viridis, vmin=min(energies),vmax=max(energies_ref)+10.0,label="normal mode")

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)


    cbar = fig2.colorbar(aimd, ax=ax2)
    cbar.set_label(r'$\Delta$ E / kcal mol$^{-1}$', rotation=270, labelpad=20)
    ax2.legend(loc='upper right',fontsize='large', frameon=True)
    plt.tight_layout()
    
    plt.show()
    #fig2.savefig(os.path.join(output_path, "cn_energy.eps"), close=True, verbose=True)
    #fig2.savefig(os.path.join(output_path, "cn_energy.png"), close=True, verbose=True)
    plt.close(fig2)



