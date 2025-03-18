import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

TRAIN_DATA_PATH = '/Users/akshaylakkur/PycharmProjects/RNAComp/RNA-Data-Sorted/'
SEQUENCES_PATH = '/Users/akshaylakkur/PycharmProjects/RNAComp/stanford-rna-3d-folding/train_sequences.csv'
SEQUENCES = pd.read_csv(SEQUENCES_PATH)['sequence'].unique()
print(SEQUENCES, len(SEQUENCES))

# scatter = go.Scatter3d(
#     x=xs, y=ys, z=zs,
#     mode='lines',
#     marker=dict(
#         size=5,
#         color=zs,  # Color by z-value
#         colorscale='Viridis',  # Choose a colorscale
#         opacity=0.8
#     )
# )
#
# # Create the layout
# layout = go.Layout(
#     title='3D Scatter Plot',
#     scene=dict(
#         xaxis=dict(title='X-axis'),
#         yaxis=dict(title='Y-axis'),
#         zaxis=dict(title='Z-axis')
#     )
# )
#
# # Create the figure
# fig = go.Figure(data=[scatter], layout=layout)
#
# # Show the plot
# fig.show()