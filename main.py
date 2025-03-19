import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import os

TRAIN_DATA_PATH = '/Users/akshaylakkur/PycharmProjects/RNAComp/RNA-Data-Sorted/'
SEQUENCES_PATH = '/Users/akshaylakkur/PycharmProjects/RNAComp/stanford-rna-3d-folding/train_sequences.csv'
def apply_func_ids(string):
    string = string.split('_')[0]
    return string
def apply_func_seqs(string):
    string = string.split('-')[0]
    return string

def sort_sorted():
    x_lis, y_lis, z_lis = [], [], []
    for i in IDS:
        path = TRAIN_DATA_PATH + i + '.csv'
        csv = pd.read_csv(path)
        x_1, y_1, z_1 = torch.tensor(csv['x_1']), torch.tensor(csv['y_1']), torch.tensor(csv['z_1'])
        x_lis.append(x_1), y_lis.append(y_1), z_lis.append(z_1)
    padded_x, padded_y, padded_z = pad_sequence(x_lis, batch_first=True, padding_value=0), pad_sequence(y_lis, batch_first=True, padding_value=0), pad_sequence(z_lis, batch_first=True, padding_value=0)
    return padded_x, padded_y, padded_z

IDS = list(pd.read_csv(SEQUENCES_PATH)['target_id'].apply(apply_func_ids))
SEQUENCES = list(pd.read_csv(SEQUENCES_PATH)['sequence'].apply(apply_func_seqs).unique())
print(IDS)
x_1,y_1,z_1 = sort_sorted()
print(x_1)
'''
model input -> Sequence, individual nucleotides
'''
def one_hot(sequence):
    maps = {'A':1, 'C':2, 'G':3, 'U':4, 'X':0}
    seqs = []
    for i in sequence:
        seqs.append(maps[i])
    indices = torch.tensor(seqs)
    one_hot_sqn = F.one_hot(indices, num_classes=5)
    return one_hot_sqn, indices

def create_one_hot_plus_pad():
    SEQUENCES_LIST = []
    TRUTH = []
    for i in SEQUENCES:
        onehot, seqs = one_hot(i)
        SEQUENCES_LIST.append(onehot)
        TRUTH.append(seqs)
    return SEQUENCES_LIST, TRUTH

SEQUENCES_LIST, TRUTH = create_one_hot_plus_pad()
x = pad_sequence(SEQUENCES_LIST, batch_first=True, padding_value=0)
mask = x.dim() != 0
class RNAModule(nn.Module):
    def __init__(self):
        super(RNAModule, self).__init__()

