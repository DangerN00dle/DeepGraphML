from __future__ import division
from __future__ import unicode_literals
import multiprocessing
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns


training_size = 5

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

# splits the smile into reaction components
reaction_smiles = []

for element in filtered_training_mol:
    if ">>" in element:
        reaction_smiles.append(re.split('>>', element))
        
    # else:
    #     reaction_smiles.append(re.split('>', element))
        
mol1 = Chem.MolFromSmiles(reaction_smiles[0][0])
mol2 = Chem.MolFromSmiles(reaction_smiles[0][1])

assignProperties(mol1)
assignProperties(mol2)

print(mol1)
print(mol2)









