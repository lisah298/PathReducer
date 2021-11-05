import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from parsers import parse_reax, parse_irc, parse_nn_disp
from calculator import coord_num, cn_visualize2D, clean_data

#title for plot, axis and directory for picture
title = r'R2: O +H$_2 \rightarrow$ OH + H (triplet)'
xtitle = 'CN2 [O1-(H1,H2)]'
ytitle = 'CN1 [H1-H2]'
directory ='cn_figs_R2' 


def get_distances(atoms, n_atoms):
    atoms = atoms.reshape(n_atoms, 3)
    dist = distance_matrix(atoms, atoms, p=2)
    return dist[np.triu_indices(n_atoms, k=1)]

#get IRC data  !!PATH NEEDS TO BE ADAPTED!!
irc_path = "/home/christopher/projects/luke_combustion/all_runs/IRC/02_O+H2_OH+H.t"
carts_irc, atomic_numbers_irc, energy_irc, forces_irc = parse_irc(irc_path)
carts_irc = carts_irc.reshape(carts_irc.shape[0], len(atomic_numbers_irc)*3)

#!!DEFINE APPROPRIATE DISTANCES FOR COORDINATION NUMBER CALCULATION AND ADD THEM THERE!!
dist_irc = np.apply_along_axis(get_distances, 1, carts_irc, n_atoms=len(atomic_numbers_irc))
dist_irc = pd.DataFrame(dist_irc, columns=['H1_O1','H1_H2','O1_H2'])  
cn1_irc = coord_num([dist_irc.H1_H2.values], mu=0.741, sigma=3.0) #sigma =0.741  is exp eq. bond length
cn2_irc = coord_num([dist_irc.H1_O1.values, dist_irc.O1_H2.values], mu=0.97, sigma=3.0) #sigma =0.97  is exp eq. bond length

#get AIMD data  !!PATH NEEDS TO BE ADAPTED!!
aimd_path = "/home/christopher/projects/luke_combustion/all_runs/R_2/"
carts, atomic_numbers, energy, forces = parse_reax(aimd_path)
carts = carts.reshape(carts.shape[0], len(atomic_numbers)*3)

#!!DEFINE APPROPRIATE DISTANCES FOR COORDINATION NUMBER CALCULATION AND ADD THEM THERE!!
dist_aimd = np.apply_along_axis(get_distances, 1, carts, n_atoms=len(atomic_numbers))
dist_aimd = pd.DataFrame(dist_aimd, columns=['H1_O1','H1_H2','O1_H2'])
cn1_aimd = coord_num([dist_aimd.H1_H2.values], mu=0.741, sigma=3.0) #sigma =0.741  is exp eq. bond length
cn2_aimd = coord_num([dist_aimd.H1_O1.values, dist_aimd.O1_H2.values], mu=0.97, sigma=3.0) #sigma =0.97  is exp eq. bond length

#get normal mode displacement data
nn_path = "/home/christopher/projects/luke_combustion/all_runs/normal_mode/reaction2_disp_new"
carts_nn_disp, atomic_numbers_nn_disp, energies_nn_disp = parse_nn_disp(nn_path)
carts_nn_disp = carts_nn_disp.reshape(carts_nn_disp.shape[0], len(atomic_numbers_nn_disp)*3)

#!!DEFINE APPROPRIATE DISTANCES FOR COORDINATION NUMBER CALCULATION AND ADD THEM THERE!!
dist_nn_disp = np.apply_along_axis(get_distances, 1, carts_nn_disp, n_atoms=len(atomic_numbers_nn_disp))
dist_nn_disp = pd.DataFrame(dist_nn_disp, columns=['H1_O1','H1_H2','O1_H2'])
cn1_nn_disp = coord_num([dist_nn_disp.H1_H2.values], mu=0.741, sigma=3.0) #sigma =0.741  is exp eq. bond length
cn2_nn_disp = coord_num([dist_nn_disp.H1_O1.values, dist_nn_disp.O1_H2.values], mu=0.97, sigma=3.0) #sigma =0.97  is exp eq. bond length


#INTERMEDIATE STEP TO CLEAN DATA
cn1_aimd, cn2_aimd, energy, cn1_nn_disp, cn2_nn_disp, energies_nn_disp = clean_data(cn1_aimd, cn2_aimd, energy, cn1_nn_disp, cn2_nn_disp, energies_nn_disp, max(energy_irc))


cn_visualize2D(cn1_aimd, cn2_aimd, energy, cn1_irc, cn2_irc, energy_irc, directory, title, xtitle, ytitle, cn1_nn_disp, cn2_nn_disp, energies_nn_disp)

