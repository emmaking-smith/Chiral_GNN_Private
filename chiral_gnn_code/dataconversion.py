# build_dataset.py
from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFP
from torch.fx.experimental.unification.multipledispatch.conflict import consistent
from smiles_to_geometric_data import Node_Info

def build_dataset(
    features: list[str],
    mfp_input: bool,
    pickle_path: str= 'data/processed_data_with_xyz.pickle',
    smiles_col: str = "SMILES",
    rotation_col: str = "Rotation",
    radius: int = 3,
    n_bits: int = 512,
) -> tuple[np.ndarray, np.ndarray]:

    df = pd.read_pickle(pickle_path)
    smiles = df[smiles_col]
    rotation = df[rotation_col]

    node_info = Node_Info()


    # Make labels: '+' -> 1, '-' -> 0
    y_series = rotation.map({"+": 1, "-": 0})
    y = y_series.to_numpy(dtype=np.float32)

    if mfp_input is True:
        mgen = rdFP.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        X = np.zeros((len(smiles), n_bits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            mfp = mgen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(mfp, X[i])

    else:
        x = []
        features_dict = {
            "atomic number": node_info.find_atomic_num,
            "hybridization": node_info.find_hybridization,
            "chirality type": node_info.find_chiral_type,
        }

        # the i-th row will store the corresponding mfp

        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            xyz_mol = df.iloc[i]['xyz']


            mol_features = []
            for id, atom in enumerate(mol.GetAtoms()):
                atom_features = []
                if 'atomic number' in features:
                    atom_number = features_dict['atomic number'](atom)
                    atom_features.append(atom_number)
                else:
                    atom_features = atom_features
                if 'hybridization' in features:
                    hybridization = features_dict['hybridization'](atom)
                    atom_features.append(hybridization)
                else:
                    atom_features = atom_features
                if 'chirality type' in features:
                    chirality_type = features_dict['chirality type'](atom)
                    atom_features.append(chirality_type)
                else:
                    atom_features = atom_features
                if 'xyz' in features:
                    xa, ya, za = xyz_mol[id]
                    atom_features.append(xa)
                    atom_features.append(ya)
                    atom_features.append(za)
                else:
                    atom_features = atom_features

                mol_features.append(atom_features)
                # atom_number = node_info.find_atomic_num(atom)
                # chirality_type = node_info.find_chiral_type(atom)
                # hybridization = node_info.find_hybridization(atom)
                # xa,ya,za = xyz_mol[id]
                # mol_features.append([atom_number, chirality_type, hybridization, xa, ya, za])
            x.append(np.array(mol_features))

            max_atom = max(m.shape[0] for m in x)
            feat_num = x[0].shape[1]

            x_3d = np.array([np.pad(m, ((0, max_atom - m.shape[0]), (0, 0)), mode='constant') for m in x])
            X = x_3d.reshape(x_3d.shape[0], max_atom * feat_num)











    return X, y

if __name__ == "__main__":
    X, y= build_dataset(features=[], mfp_input=True)
    print(X, X.shape)
    # print("y:", y.shape, y.dtype, y.nbytes / 1e6)

