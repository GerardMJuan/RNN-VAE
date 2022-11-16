"""
Use the results from missingdata.py to plot the results.
Line plots per task, with the number of missing data points on the x-axis and
the MAE on the y-axis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ast import literal_eval as make_tuple

#output dir to save the figures
out_dir = "/home/gerard/DATA/RNNVAE/"

# Load the results
results = pd.read_csv(f'{out_dir}results_full.csv')

# Plot the results

# Set the style
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)

# Set the figure size
# We need to do a 3x4 subplot, with the top 3x3 matrix being the plots of the channel to channel recon tasks
# and the bottom row being the plots of the all channels to one channel recon tasks
# Each subplot must be square
# We want to have the size of the subplots be 2.5x2.5 inches
# So the total figure size is 7.5x7.5 inches
fig, axes = plt.subplots(4, 3, figsize=(15.5, 15.5))
# fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.15, wspace=0.15)

# Each subplot has various linepoints, one for each method (KNN, GFA, MCVAE)
# Note that subplots in the diagonal do not have GFA
# We want to plot the results for each method in a different color
# We want to plot the results for each task in a different line style
# We want to plot the results for each number of missing data points in a different marker
# Shaded error bands are plotted with fill_between

# Set the colors
colors = sns.color_palette('colorblind', 3)

# Set the line styles
line_styles = ['-', '--', '-.']
markers = ['o', 's', 'D']
missing_data_points = list(range(54))# s[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
channels_namess = ['Vol.','Cort.', 'Cog.']
channels = ['_mri_vol','_mri_cort', '_cog']

## ITERATE TO CREATE THE PLOTS
for i, ch in enumerate(channels):
    for j, ch2 in enumerate(channels):
        curr_ax = axes[i][j] # current ax
        
        # KNN
        mean_knn, std_knn = zip(*[x for x in map(make_tuple, results[f"KNN_{channels[i]}_to_{channels[j]}"])])
        mean_knn = list(mean_knn)
        std_knn = list(std_knn)
        sns.lineplot(x=missing_data_points, y=mean_knn, ax=curr_ax, color=colors[0], marker=markers[0], linestyle=line_styles[0], label='KNN')
        curr_ax.errorbar(missing_data_points, mean_knn, std_knn, color=colors[0], linestyle=line_styles[0], marker=markers[0])
        # Set corresponding x, y axis labels and title

        # GFA (only if i!= j)
        if i != j:
            mean_gfa, std_gfa = zip(*[x for x in map(make_tuple, results[f"GFA_{channels[i]}_to_{channels[j]}"])])
            mean_gfa = list(mean_gfa)
            std_gfa = list(std_gfa)
            print(mean_gfa)
            sns.lineplot(x=missing_data_points, y=mean_gfa, ax=curr_ax, color=colors[1], marker=markers[1], linestyle=line_styles[1], label='GFA')
            curr_ax.errorbar(missing_data_points, mean_gfa, std_gfa, color=colors[1], linestyle=line_styles[1], marker=markers[1])
        
        # MC-RVAE
        mean_mc, std_mc = zip(*[x for x in map(make_tuple, results[f"MCRVAE_{channels[i]}_to_{channels[j]}"])])
        mean_mc = list(mean_mc)
        std_mc = list(std_mc)
        sns.lineplot(x=missing_data_points, y=mean_mc, ax=curr_ax, color=colors[2], marker=markers[2], linestyle=line_styles[2], label='MC-RVAE')
        curr_ax.errorbar(missing_data_points, mean_mc, std_mc, color=colors[2], linestyle=line_styles[2], marker=markers[2])

        curr_ax.set_xlabel('Missing data percentage')
        curr_ax.set_ylabel('MAE')
        curr_ax.set_title(f'{channels_namess[i]} to {channels_namess[j]}')


# Plot the last three, in the last row of the axes
# the name in df is just method_channel[i]
for i, ch in enumerate(channels):
    curr_ax = axes[3][i]
    # KNN
    mean_knn, std_knn = zip(*[x for x in map(make_tuple, results[f"KNN_{channels[i]}"])])
    mean_knn = list(mean_knn)
    std_knn = list(std_knn)
    sns.lineplot(x=missing_data_points, y=mean_knn, ax=curr_ax, color=colors[0], marker=markers[0], linestyle=line_styles[0], label='KNN')
    curr_ax.errorbar(missing_data_points, mean_knn, std_knn, color=colors[0], linestyle=line_styles[0], marker=markers[0])

    # GFA
    mean_gfa, std_gfa = zip(*[x for x in map(make_tuple, results[f"GFA_{channels[i]}"])])
    mean_gfa = list(mean_gfa)
    std_gfa = list(std_gfa)
    sns.lineplot(x=missing_data_points, y=mean_gfa, ax=curr_ax, color=colors[1], marker=markers[1], linestyle=line_styles[1], label='GFA')
    curr_ax.errorbar(missing_data_points, mean_gfa, std_gfa, color=colors[1], linestyle=line_styles[1], marker=markers[1])

    # MC-RVAE
    mean_mc, std_mc = zip(*[x for x in map(make_tuple, results[f"MCRVAE_{channels[i]}"])])
    mean_mc = list(mean_mc)
    std_mc = list(std_mc)
    sns.lineplot(x=missing_data_points, y=mean_mc, ax=curr_ax, color=colors[2], marker=markers[2], linestyle=line_styles[2], label='MC-RVAE')
    curr_ax.errorbar(missing_data_points, mean_mc, std_mc, color=colors[2], linestyle=line_styles[2], marker=markers[2])

    curr_ax.set_xlabel('Missing data percentage')
    curr_ax.set_ylabel('MAE')
    curr_ax.set_title(f'{channels_namess[i]} recon.')

# Remove the legends
[[c.get_legend().remove() for c in r] for r in axes]

handles, labels = curr_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()

# Save the figure
plt.savefig(f'{out_dir}missingdata.png', dpi=300)

#Save the figure as pdf too
plt.savefig(f'{out_dir}missingdata.pdf', dpi=300)

# LATER
# ax.fill_between(missing_data_points, y1=mean_knn - std_knn, y2=mean_knn + std_knn, alpha=.5, color=colors[0])
