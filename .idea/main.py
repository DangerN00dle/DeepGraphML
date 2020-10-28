from __future__ import division
from __future__ import unicode_literals
import multiprocessing
import logging
import matplotlib.pyplot as plt
import numpy as np
import re

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
from torch_geometric.data import Data

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns

#################################
##      PREREQUISITS           ##
#################################

training_size = 15
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
max_nb = 10

##################################
##      DEFINED FUNCTIONS       ##
##################################

# function, which creates an allowable_set with all atom_features
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# function which gets the atom features based on the atom as input
#   atom.GetSymbol: Returns the atomic symbol
#   atom.GetDegree: Returns the degree of the atom in the molecule
#   atom.GetExplicitValence: Returns the explicit valence of the atom
#   atom.GetImplicitValence: Returns the implicit valence of the atom
#   atom.GetIsAromatic: Return boolean value, if it is aromatic
#   atom.IsInRing: Returns whether or not the atom is in a ring
#   atom.GetAtomicNum: Returns the atomic number of the element, ordered by ; is != GetAtomicNumber
#   atom.GetDoubleProp: returns the value of a particular property
def atom_features(atom):
    attributes = onek_encoding_unk(atom.GetSymbol(), elem_list) \
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) \
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6]) \
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5]) \
            + [atom.GetIsAromatic()] \
            + [atom.GetIsAromatic() == False and any([neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()])] \
            + [atom.IsInRing()] \
            + [atom.GetAtomicNum() in [9, 17, 35, 53, 85, 117]] \
            + [atom.GetAtomicNum() in [8, 16, 34, 52, 84, 116]] \
            + [atom.GetAtomicNum() in [7, 15, 33, 51, 83]] \
            + [atom.GetAtomicNum() in [3, 11, 19, 37, 55, 87]] \
            + [atom.GetAtomicNum() in [4, 12, 20, 38, 56, 88]] \
            + [atom.GetAtomicNum() in [13, 22, 24, 25, 26, 27, 28, 29, 30, 33, 42, 44, 45, 46, 47, 48, 49, 50, 78, 80, 82]] \
            + [atom.GetDoubleProp('crippen_logp'), atom.GetDoubleProp('crippen_mr'), atom.GetDoubleProp('tpsa'),
               atom.GetDoubleProp('asa'), atom.GetDoubleProp('estate'),
               atom.GetDoubleProp('_GasteigerCharge'), atom.GetDoubleProp('_GasteigerHCharge')]
    attributes = np.array(attributes, dtype=np.float32)
    attributes[np.isnan(attributes)] = 0.0 # filter if the output is not a number
    attributes[np.isinf(attributes)] = 0.0 # filter if the output is Infinity
    return attributes

# function which returns the bond features
#   atom.GetBondType: Returns the type of the bond as a BondType
#   atom.GetIsConjugated: Returns whether or not the bond is considered to be conjugated.
#   atom.IsInRing: Returns whether or not the bond is in a ring of any size
def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

# function that converts a smile into a graph
#   MolFromSmiles: returns the molecular structure of a smiles
def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    n_atoms = mol.GetNumAtoms()                             # Number of atoms                         
    n_bonds = max(mol.GetNumBonds(), 1)                     # Number of Bonds
    fatoms = np.zeros((n_atoms, atom_fdim))                 # Creates feature atom zero matrix(number of atoms x number of features)
    fbonds = np.zeros((n_bonds, bond_fdim))                 # Creates feature bond zero matrix(number of bonds x number of features)
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)   # Creates zero matrix atom_number_of_bonds 
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)   # Creates zero matrix bond_number_of_bonds
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)          # Creates a zero matrix (nubmer of atoms x number of atoms)

    assignProperties(mol)                                   # Assingnes further properties

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)                                 # Returns the atom’s index of the atom
        if idx >= n_atoms:
            raise Exception(smiles)
        fatoms[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())                   # Reutrns the bonds between the two atoms a1 and a 2 
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()                                 # GetIdx: Returns the atom’s index (ordering in the molecule)
        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(smiles)
        atom_nb[a1,num_nbs[a1]] = a2                        # Returns which atom is bonded to which atom
        atom_nb[a2,num_nbs[a2]] = a1
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
    
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    data = Data(
            x=torch.tensor(adj_matrix, dtype=torch.long, device=device),
            edge_index=torch.tensor(bond_nb,dtype=torch.long, device=device),
            edge_attr=torch.tensor(fbonds,dtype=torch.float)
            )
    print(data)
    print('Adj: ', data.x)
    print('Bonds: ', data.edge_index)
    # data = Data(
    #         x=torch.tensor(adj_matrix, dtype=torch.float, device=device),
    #         num_nbs=torch.tensor(num_nbs,dtype=torch.float, device=device),
    #         fatoms=torch.tensor(fatoms, dtype=torch.float, device=device),
    #         fbonds=torch.tensor(fbonds,dtype=torch.float, device=device),
    #         atom_nb=torch.tensor(atom_nb,dtype=torch.float, device=device),
    #         bond_nb=torch.tensor(bond_nb,dtype=torch.float, device=device)
    #         )
    return data


# function, which assignes enhanced atom properties
#   rdMolDescriptors: Module containing functions to compute molecular descriptors
#   GetAtomWithIdx: Returns a particular Atom "i" from the mol "x"
#   SetDoubleProp: Sets a double valued molecular property
#   rdPartialCharges: Compute Gasteiger partial charges for molecule
def assignProperties(mol):
    for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('crippen_logp',x[0])
        mol.GetAtomWithIdx(i).SetDoubleProp('crippen_mr', x[1])
    for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('tpsa', x)
    for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(mol)[0]):
        mol.GetAtomWithIdx(i).SetDoubleProp('asa', x)
    for (i, x) in enumerate(EState.EStateIndices(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('estate', x)
    rdPartialCharges.ComputeGasteigerCharges(mol) # '_GasteigerCharge', '_GasteigerHCharge'


##################################
##         START SCRIPT         ##
##################################


# reads the smiles file line by line, removes whitespace 
# characters and splits it into a reasonable training_size 
with open('.idea/data/100kSmiles.rsmi','r') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
training_mol = content[1:training_size]

# filter out patentnumber, year, etc.
filtered_training_tmp = []
filtered_training_mol = []

for element in training_mol:
    tmp = element.split(' ', 1)[0]
    filtered_training_tmp.append(tmp)

for element in filtered_training_tmp:
    tmp = element.split('\t', 1)[0]
    filtered_training_mol.append(tmp)

# splits the smile into reaction components, uncomment 
reaction_smiles = []

for element in filtered_training_mol:
    if ">>" in element:
        reaction_smiles.append(re.split('>>', element))       
    # else:
    #     reaction_smiles.append(re.split('>', element))

data = []
label = []

# todo export as def
for index in range(len(reaction_smiles)):
    smiles = reaction_smiles[index][0]
    mol = Chem.MolFromSmiles(smiles)
    assignProperties(mol)
    atom = mol.GetAtoms()[0]
    bond = mol.GetBonds()[0]
    atom_fdim = len(atom_features(atom))
    bond_fdim = len(bond_features(bond))
    data.append(smiles2graph(smiles))
    print(data)

for index in range(len(reaction_smiles)):
    smiles = reaction_smiles[index][1]
    mol = Chem.MolFromSmiles(smiles)
    assignProperties(mol)
    atom = mol.GetAtoms()[0]
    bond = mol.GetBonds()[0]
    atom_fdim = len(atom_features(atom))
    bond_fdim = len(bond_features(bond))
    label.append(smiles2graph(smiles))
    
train_mols = data[:2]
test_mols = data[2:]

train_X = train_mols
for i, data in enumerate(train_X):
    y = label[i]
    data.y = y
 
test_X = test_mols
for i, data in enumerate(test_X):
    y = label[i]
    data.y = y

train_loader = DataLoader(train_X, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(test_X, batch_size=1, shuffle=True, drop_last=True)

##=============================##
##          GCN-Model            ##
##=============================##

n_features = 1700

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, n_features)
         
    def forward(self, data):
        x, edge_index, edge_attr  = data.x, data.edge_index, data.edge_attr
        print('edges: ', edge_index.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, 1)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_X)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
hist = {"loss":[], "acc":[], "test_acc":[]}

for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    hist["loss"].append(train_loss)
    hist["acc"].append(train_acc)
    hist["test_acc"].append(test_acc)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')
ax = plt.subplot(1,1,1)
ax.plot([e for e in range(1,101)], hist["loss"], label="train_loss")
ax.plot([e for e in range(1,101)], hist["acc"], label="train_acc")
ax.plot([e for e in range(1,101)], hist["test_acc"], label="test_acc")
plt.xlabel("epoch")
ax.legend()