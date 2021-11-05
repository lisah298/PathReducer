import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from parsers import parse_reax, parse_irc
from calculator import coord_num, cn_visualize2D

#title for plot, axis and directory for picture
title = r'R19: H$_2$O$_2$ + O(3)H $\rightarrow$ HO$_2$ + H$_2$O(3) (doublet)'
xtitle = 'CN2 [O3-(H1,H2,H3)]'
ytitle = 'CN1 [O2-(H1,H2,H3)]'
directory ='cn_figs_R19' 


def get_distances(atoms, n_atoms):
    atoms = atoms.reshape(n_atoms, 3)
    dist = distance_matrix(atoms, atoms, p=2)
    return dist[np.triu_indices(n_atoms, k=1)]

#get IRC data  !!PATH NEEDS TO BE ADAPTED!!
irc_path = "/home/christopher/projects/luke_combustion/all_runs/IRC/19_H2O2+OH_H2O+HO2.d"
carts_irc, atomic_numbers_irc, energy_irc, forces_irc = parse_irc(irc_path)
carts_irc = carts_irc.reshape(carts_irc.shape[0], len(atomic_numbers_irc)*3)

#!!DEFINE APPROPRIATE DISTANCES FOR COORDINATION NUMBER CALCULATION AND ADD THEM THERE!!
dist_irc = np.apply_along_axis(get_distances, 1, carts_irc, n_atoms=len(atomic_numbers_irc))
dist_irc = pd.DataFrame(dist_irc, columns=['O1_O2','O1_H1','O1_H2','O1_O3','O1_H3','O2_H1','O2_H2','O2_O3','O2_H3','H1_H2','H1_O3','H1_H3','H2_O3','H2_H3','O3_H3'])  
cn1_irc = coord_num([dist_irc.O1_H1.values, dist_irc.O1_H2.values, dist_irc.O1_H3.values], mu=0.97, sigma=3.0) #sigma =0.741  is exp eq. bond length
cn2_irc = coord_num([dist_irc.H1_O3.values, dist_irc.H2_O3.values, dist_irc.O3_H3.values], mu=0.97, sigma=3.0) #sigma =0.97  is exp eq. bond length

#get AIMD data  !!PATH NEEDS TO BE ADAPTED!!
aimd_path = "/home/christopher/projects/luke_combustion/all_runs/R_19/"
carts, atomic_numbers, energy, forces = parse_reax(aimd_path)
carts = carts.reshape(carts.shape[0], len(atomic_numbers)*3)

#!!DEFINE APPROPRIATE DISTANCES FOR COORDINATION NUMBER CALCULATION AND ADD THEM THERE!!
dist_aimd = np.apply_along_axis(get_distances, 1, carts, n_atoms=len(atomic_numbers))
dist_aimd = pd.DataFrame(dist_aimd, columns=['O1_O2','O1_H1','O1_H2','O1_O3','O1_H3','O2_H1','O2_H2','O2_O3','O2_H3','H1_H2','H1_O3','H1_H3','H2_O3','H2_H3','O3_H3'])
cn1_aimd = coord_num([dist_aimd.O1_H1.values, dist_aimd.O1_H2.values, dist_aimd.O1_H3.values], mu=0.97, sigma=3.0) #sigma =0.741  is exp eq. bond length
cn2_aimd = coord_num([dist_aimd.H1_O3.values, dist_aimd.H2_O3.values, dist_aimd.O3_H3.values], mu=0.97, sigma=3.0) #sigma =0.97  is exp eq. bond length


cn_visualize2D(cn1_aimd, cn2_aimd, energy, cn1_irc, cn2_irc, energy_irc, directory, title, xtitle, ytitle)

