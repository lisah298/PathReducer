import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot, colored_line_and_scatter_plot, colored_line_plot_projected_data

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
input_path = './testdata/malonaldehyde_IRC.xyz'
newdata = './examples/malonaldehyde_irc2.xyz'
newdata2 = './examples/malonaldehyde_manipulated.xyz'
stereo_atoms_malon = [4, 7, 1, 8]
points_to_circle = [0, 12, 24]
output_directory = './output'
output_directory_new = './output'


# DISTANCES INPUT, NOT MASS-WEIGHTED. To mass-weight coordinates, add "MW=True" to function call.
system_name, output_directory, pca, pca_fit, pca_components, mean, values, lengths, aligned_coords = \
    dim_red.pathreducer(input_path, ndim, input_type="Cartesians")


new_data_df = dim_red.transform_new_data(newdata2, './newdata', 2, pca_fit,
                                         pca_components, mean, aligned_coords, stereo_atoms=stereo_atoms_malon,  input_type='Cartesians')[1]


pca_df = pd.DataFrame(pca)


# Original Data
# colored_line_and_scatter_plot(pca_df[0], y=pca_df[1], output_directory=output_directory,
# imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)

# New Data
colored_line_plot_projected_data(x=pca_df[0], y=pca_df[1], z=pca_df[2], new_data_x=new_data_df[0], new_data_y=new_data_df[1], new_data_z=new_data_df[2],
                                 points_to_circle=points_to_circle)


'''
# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], output_directory=output_directory_D,
          imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)
'''
