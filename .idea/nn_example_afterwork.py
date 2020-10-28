from __future__ import division
from __future__ import unicode_literals
import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import torch
from torch_geometric.data import Data
import re
from scipy.linalg import block_diag

training_size = 2000
train_test_threshold = 1500
n_features = 200
batch_size = 1
end_range = 31
device = 'cuda'
 
def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))
 
 
def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))
 
 
def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths"""
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
 
  return intervals
 
 
def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found"""
  try:
    return l.index(e)
  except:
    return len(l)
 
 
possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']
 
reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]
 
intervals = get_intervals(reference_lists)
 
 
def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  return features
 
 
def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals"""
  id = 0
  for k in range(len(intervals)):
    id += features[k] * intervals[k]
 
  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id
 
 
def id_to_features(id, intervals):
  features = 6 * [0]
 
  # Correct for null
  id -= 1
 
  for k in range(0, 6 - 1):
    # print(6-k-1, id)
    features[6 - k - 1] = id // intervals[6 - k - 1]
    id -= features[6 - k - 1] * intervals[6 - k - 1]
  # Correct for last one
  features[0] = id
  return features
 
 
def atom_to_id(atom):
  """Return a unique id corresponding to the atom type"""
  features = get_feature_list(atom)
  return features_to_id(features, intervals)
 
 
def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
     [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
 
    return np.array(results)
 
 
def bond_features(bond, use_chirality=False):
  from rdkit import Chem
  bt = bond.GetBondType()
  bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
  ]
  if use_chirality:
    bond_feats = bond_feats + one_of_k_encoding_unk(
        str(bond.GetStereo()),S
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
  return np.array(bond_feats)
 
#################
# pen added
#################
def get_bond_pair(mol):
  bonds = mol.GetBonds()
  res = [[],[]]
  for bond in bonds:
    res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
  return res
 
def mol2vec(smiles):
  mol = Chem.MolFromSmiles(smiles)
  
  atoms = mol.GetAtoms()
  node_f= [atom_features(atom) for atom in atoms]
  node_f_pading = np.zeros(((n_features - len(node_f)),(len(node_f[0]))))
  node_f = np.vstack((node_f, node_f_pading))
  node_f = np.pad(node_f, [(0, (n_features - len(node_f))), (0, (n_features - len(node_f[0])))], mode='constant', constant_values=0)
  bonds = mol.GetBonds()
  edge_index = get_bond_pair(mol)
  edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
  for bond in bonds:
    edge_attr.append(bond_features(bond))
  data = Data(x=torch.tensor(node_f, dtype=torch.float, device=device),
              edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
              edge_attr=torch.tensor(edge_attr,dtype=torch.float, device=device)
              )
  return data
  
def mol2vec_sol(smiles):
  mol = Chem.MolFromSmiles(smiles)
  atoms = mol.GetAtoms()
  node_f= [atom_features(atom) for atom in atoms]
  node_f_pading = np.zeros(((n_features - len(node_f)),(len(node_f[0]))))
  node_f = np.vstack((node_f, node_f_pading))
  node_f = np.pad(node_f, [(0, (n_features - len(node_f))), (0, (n_features - len(node_f[0])))], mode='constant', constant_values=0)
  data = torch.tensor(node_f, dtype=torch.float, device=device)
  return data


def mol2vec_dual(smiles):
  smiles_one = smiles[0]
  smiles_two = smiles[1]

  mol1 = Chem.MolFromSmiles(smiles_one)
  mol2 = Chem.MolFromSmiles(smiles_two)
  
  atoms1 = mol1.GetAtoms()
  atoms2 = mol2.GetAtoms()

  bonds1 = mol1.GetBonds()
  bonds2 = mol2.GetBonds()

  node_f1= [atom_features(atom) for atom in atoms1]
  node_f2= [atom_features(atom) for atom in atoms2]

  edge_index1 = get_bond_pair(mol1)
  edge_index2 = get_bond_pair(mol2)

  edge_attr1 = [bond_features(bond, use_chirality=False) for bond in bonds1]
  edge_attr2 = [bond_features(bond, use_chirality=False) for bond in bonds2]

  for bond in bonds1:
    edge_attr1.append(bond_features(bond))

  for bond in bonds2:
    edge_attr2.append(bond_features(bond))

  if (edge_index2 == [[], []]):
    edge_index2 = [[0],[0]]

  zero_attr = ([False, False, False, False, False, False])
  if (edge_attr2 == []):
    edge_attr2.append(zero_attr)

  node_f = np.vstack((node_f1, node_f2))
  node_f_pading = np.zeros(((n_features - len(node_f)),(node_f.shape[1])))
  node_f = np.vstack((node_f, node_f_pading))
  node_f = np.pad(node_f, [(0, (n_features - len(node_f))), (0, (n_features - len(node_f[0])))], mode='constant', constant_values=0)
  edge_index = np.hstack((edge_index1, edge_index2))
  edge_attr = np.row_stack((edge_attr1, edge_attr2))

  data = Data(x=torch.tensor(node_f, dtype=torch.float, device=device),
              edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
              edge_attr=torch.tensor(edge_attr,dtype=torch.float, device=device)
              )
  return data


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

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import torch
from torch.nn import Linear
import torch_geometric.transforms as T
plt.style.use("ggplot")

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


##################################
##         START SCRIPT         ##
##################################


# reads the smiles file line by line, removes whitespace 
# characters and splits it into a reasonable training_size 
with open('.idea/data/100kSmiles.rsmi','r') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
training_mol = content[1:(training_size+1)]

# filter out patentnumber, year, etc.
filtered_training_tmp = []
filtered_training_mol = []

for element in training_mol:
    tmp = element.split(' ', 1)[0]
    filtered_training_tmp.append(tmp)

for element in filtered_training_tmp:
    tmp = element.split('\t', 1)[0]
    if (re.search(r">", tmp)):
      filtered_training_mol.append(tmp)



# splits the smile into reaction components
reaction_smiles = []
reaction_smiles_dual = []
reaction_smiles_dual_sol = []


for element in filtered_training_mol:
    if ">>" in element:
        reaction_smiles.append(re.split('>>', element))       
    else:
        reaction_smiles_dual.append(re.split('>', element))

index = 0
while index < len(reaction_smiles):
  mol1 = Chem.MolFromSmiles(reaction_smiles[index][0])
  mol2 = Chem.MolFromSmiles(reaction_smiles[index][1])

  if mol1 is None:
    reaction_smiles.pop(index)
    break
  if mol2 is None:
    reaction_smiles.pop(index)
    break

  atoms1 = mol1.GetNumAtoms()
  atoms2 = mol2.GetNumAtoms()

  if atoms1 >= n_features:
    reaction_smiles.pop(index)
    break
  if atoms2 >= n_features:
    reaction_smiles.pop(index)
    break
  index += 1

index = 0
while index < len(reaction_smiles_dual):
  mol1 = Chem.MolFromSmiles(reaction_smiles_dual[index][0])
  mol2 = Chem.MolFromSmiles(reaction_smiles_dual[index][1])
  mol3 = Chem.MolFromSmiles(reaction_smiles_dual[index][2])

  if mol1 is None:
    reaction_smiles_dual.pop(index)
    break
  if mol2 is None:
    reaction_smiles_dual.pop(index)
    break
  if mol3 is None:
    reaction_smiles_dual.pop(index)
    break

  atoms1 = mol1.GetNumAtoms()
  atoms2 = mol2.GetNumAtoms()
  atoms3 = mol3.GetNumAtoms()

  if atoms1 >= n_features:
    reaction_smiles_dual.pop(index)
    break
  if atoms2 >= n_features:
    reaction_smiles_dual.pop(index)
    break
  if atoms1 + atoms2 >= n_features:
    reaction_smiles_dual.pop(index)
    break
  if atoms3 >= n_features:
    reaction_smiles_dual.pop(index)
    break
  index += 1

reaction_smiles_input = []
reaction_smiles_sol = []
reaction_smiles_sol_dual = []

for index in range(len(reaction_smiles)):
    reaction_smiles_input.append(reaction_smiles[index][0])
    reaction_smiles_sol.append(reaction_smiles[index][1])

for index in range(len(reaction_smiles_dual)):
    reaction_smiles_sol_dual.append(reaction_smiles_dual[index][2])
    

gcn_input = [mol2vec(m) for m in reaction_smiles_input]
gcn_input_dual = [mol2vec_dual(m) for m in reaction_smiles_dual]
gcn_sol = [mol2vec_sol(m) for m in reaction_smiles_sol]
gcn_sol_dual = [mol2vec_sol(m) for m in reaction_smiles_sol_dual]

for index, data in enumerate(gcn_input):
    data.y = gcn_sol[index]
 
for index, data in enumerate(gcn_input_dual):
    data.y = gcn_sol_dual[index]

gcn_input_final = (gcn_input + gcn_input_dual)

train_dataset = [m for m in gcn_input_final[:train_test_threshold]]
test_dataset = [m for m in gcn_input_final[train_test_threshold:]]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# definenet
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, 32, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.conv2 = GCNConv(32, 16, cached=False)
        self.conv3 = GCNConv(16, n_features, cached=False)
         
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x  


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)

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