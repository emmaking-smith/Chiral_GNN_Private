from networkx.algorithms.distance_measures import radius
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem
import pandas as pd
import numpy as np
import torch

df = pd.read_pickle("test.pickle")

# make sure you select a Series (not a DataFrame)
smiles_series = df['SMILES'].astype(str).fillna("")
print(smiles_series)



def morganfingerprint(mol: Chem.rdchem.Mol, atom_idx) -> np.ndarray:
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
    atom_fp_rdkit = fpgen.GetFingerprint(mol, fromAtoms=[atom_idx])
    # print(atom_fp_rdkit)
    atom_fp_array = np.zeros((1024,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(atom_fp_rdkit, atom_fp_array)
    atom_fp= atom_fp_array.tolist()
    return atom_fp


if __name__ == '__main__':

    for smile in smiles_series:
        mol = Chem.MolFromSmiles(smile)
        print(mol.GetNumAtoms())
        atom_features =[]
        for idx,atom in enumerate(mol.GetAtoms()):
            atom_morgan = morganfingerprint(mol=mol, atom_idx=idx)
            atom_features.append(atom_morgan)
            atom_features.append(1)
        torch.tensor(atom_features, dtype=torch.float).reshape((len(mol.GetAtoms()),-1))










