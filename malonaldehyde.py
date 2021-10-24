import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
input_path = './examples/malonaldehyde_irc2.xyz'
stereo_atoms_malon = [4, 7, 1, 8]
points_to_circle = [0, 12, 24]

# DISTANCES INPUT, NOT MASS-WEIGHTED. To mass-weight coordinates, add "MW=True" to function call.
# system_name, output_directory, pca, pca_fit, pca_components, mean, values, aligned_original_coords = \
#dim_red.pathreducer(input_path, ndim, input_type="Cartesians")
pca = dim_red.pathreducer(input_path, ndim, input_type="Cartesians")[2]
pca_df = pd.DataFrame(pca)
dim_red.plotting_functions.colored_line_and_scatter_plot(
    pca_df[0], pca_df[1], pca_df[2])

'''
# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], output_directory=output_directory_D,
          imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)




dim_red.plotting_functions.plot_irc('/Users/ec18006/Documents/CHAMPS/Dimensionality_reduction/xyz_pdb_files/examples/malondialdehyde/*ener1.txt')'''
