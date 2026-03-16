
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

from attr.validators import max_len
from rdkit import Chem
from rdkit.Chem import Descriptors



parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', type=str, default='test' )
save_dir = parser.parse_args().save_dir

df= pd.read_csv('data/processed_data.csv')
# print(df.columns)

#Analysis for how balanced the dataset is

df["label"] = df["Rotation"].map({"+": 1, "-": 0})

label_counts = df["label"].value_counts()
print(label_counts)

plt.figure()
label_counts.sort_index().plot(kind="bar")
# plt.xticks(ticks=[0, 1], labels=["− (negative)", "+ (positive)"], rotation=0)
# plt.ylabel("Number of samples")
# plt.title("Class balance of optical rotation labels")
# plt.show()
# plt.savefig(os.path.join(save_dir,"Data Balance Analysis"), dpi=1200)



#analysis of heaviest atom
smiles = df['SMILES']

symbol = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca",
     "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr",
     "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i", "xe", "cs", "ba", "la",
     "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w ", "re", "os",
     "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am",
     "cm", "bk", "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn", "nh", "fl",
     "mc", "lv", "ts", "og"]


weight_list = []
atom_idex_list_unique=[]
num_atoms_list = []
atom_idex_dic=[]

for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()
    num_atoms_list.append(num_atoms)
    mol_with_h = Chem.AddHs(mol)
    exact_mw = Descriptors.ExactMolWt(mol_with_h)
    weight_list.append(exact_mw)
    for atom in mol.GetAtoms():
        atom_idex = atom.GetAtomicNum()
        if atom_idex != 6 :
            atom_idex_dic.append(atom_idex) #for plotting out the distribution
        if atom_idex not in atom_idex_list_unique:
            atom_idex_list_unique.append(atom_idex)


heaviest_atom=max(atom_idex_list_unique)
heaviest_symbol= symbol[heaviest_atom-1]

symbol_list_unique =[]
for atom_idex in atom_idex_list_unique:
    atom_symbol=symbol[atom_idex-1]
    symbol_list_unique.append(atom_symbol)


print(f"maximum molecular weight:{max(weight_list)}, minimum molecular weight:{min(weight_list)}")
avg_mol_weight = np.mean(weight_list)
print(f"average molecular weight:{avg_mol_weight}")
print(f'heaviest element:{heaviest_symbol}')
print(f"the elements that we have come across: {symbol_list_unique}")
print(f"the number of elements that we have come across: {len(symbol_list_unique)+1}")

from collections import Counter

length_counts= Counter(num_atoms_list)
x=list(length_counts.keys())
print(f'maximum length: {max(x)}, minimum length: {min(x)}')
# y=list(length_counts.values())
# x,y =zip(*sorted(length_counts.items()))
# plt.figure()
# plt.bar(x, y)
# plt.xlabel("Molecule length")
# plt.ylabel("Number of molecules")
# plt.title("Distribution of molecule lengths")
# plt.show()
# #scatter plot
# plt.figure()
# plt.scatter(x, y)
# plt.xlabel("Molecule length (number of atoms)")
# plt.ylabel("Number of molecules")
# plt.title("Distribution of molecule lengths")
# plt.show()



#
# symbols = [symbol[z - 1] for z in atom_idex_dic]
# element_counts = Counter(symbols)
# # get atomic number back from symbol index
# sorted_items = sorted(
#     element_counts.items(),
#     key=lambda x: symbol.index(x[0])
# )
#
# elements, counts = zip(*sorted_items)
#
#
# plt.figure(figsize=(8, 4))
# plt.bar(elements, counts)
# plt.xlabel("Element")
# plt.ylabel("Atom count")
# plt.title("Atomic composition distribution")
# plt.tight_layout()
# plt.show()



