import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot, colored_line_and_scatter_plot, colored_line_plot_projected_data

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
input_path = './examples/12a_irc.xyz'

#stereo_atoms_malon = [4, 7, 1, 8]
#points_to_circle = [0, 12, 24]
output_directory = './output'


# DISTANCES INPUT, NOT MASS-WEIGHTED. To mass-weight coordinates, add "MW=True" to function call.
system_name, output_directory, pca, pca_fit, pca_components, mean, values, lengths, aligned_coords = \
    dim_red.pathreducer(input_path, ndim, input_type="Cartesians")

pca_df = pd.DataFrame(pca)


# Original Data
colored_line_and_scatter_plot(pca_df[0], pca_df[1], pca_df[2], output_directory=output_directory, imgname=(
    system_name + "hydrogen"))


'''
# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], output_directory=output_directory_D,
          imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)
'''
