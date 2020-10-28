from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import re
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

n_features = 100
smiles = ('[C:5]1([C@H:4]([N:40]([C:41]2[CH:50]=[CH:49][C:44]([C:45]([O:47][CH3:48])=[O:46])=[CH:43][CH:42]=2)[CH3:39])[CH2:8][N:11]2[CH2:15][CH2:14][C@H:13]([O:16][CH2:17][O:18][CH3:19])[CH2:12]2)[CH2:21][ClH:1][CH:2]=[CH:7][CH:6]=1')

mol1 = Chem.MolFromSmiles(smiles)
print(mol1.GetNumAtoms())