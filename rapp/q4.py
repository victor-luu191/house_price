import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('binary_classification_ds.csv')

# plot histogram of features, by batch of 10
b_size = 10
cols = data.columns
n_batch = len(cols)//b_size
remain = len(cols) % b_size


def plot_hist_of_batch(b, cols, data):

    start = b * b_size
    end = start + b_size
    to_plot = cols[start:end]
    fig = plt.figure(figsize=(10, 10))
    _ = data[to_plot].hist(grid=False)
    return fig


for b in range(n_batch):
    fig = plot_hist_of_batch(b, cols, data)

plt.show()


