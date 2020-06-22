import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
RED_BLUE = plt.get_cmap('RdBu')


def plot_corr_matrix(data, annotate=False, color_map=RED_BLUE):
    sns.set_style('white')
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask,
                cmap=color_map, cbar_kws={"shrink": .5},
                vmax=.3, center=0,
                square=True, linewidths=.5,
                annot=annotate
                )
