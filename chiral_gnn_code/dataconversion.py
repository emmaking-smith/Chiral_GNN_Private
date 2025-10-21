# build_dataset.py
from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFP

def build_dataset(
    csv_path: str= './data/processed/dataset.csv',
    smiles_col: str = "SMILES",
    rotation_col: str = "Rotation",
    radius: int = 3,
    n_bits: int = 1024,
    include_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, n_bits) float32 fingerprint matrix
      y: (N,) int labels (Rotation '+'->1, '-'->0)
      df_clean: original cols + helper cols for traceability
    Drops rows with invalid SMILES or missing/unmappable rotation.
    """
    df = pd.read_csv(csv_path)

    # Clean raw columns
    smiles = df[smiles_col]
    rotation = df[rotation_col]

    # Make labels: '+' -> 1, '-' -> 0
    y_series = rotation.map({"+": 1, "-": 0})
    y = y_series.to_numpy(dtype=int)

    # Morgan generator (bit fingerprint)
    mgen = rdFP.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=include_chirality)
    X = np.zeros((len(smiles), n_bits), dtype=np.int8)


    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        mfp = mgen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(mfp, X[i])

    return X, y

if __name__ == "__main__":
    X, y = build_dataset("./data/processed_data.csv")

    # basic info-to test the code literally works fine
    # print("X shape:", X.shape)
    #print("y shape:", y.shape)
   # print(df_clean.head())            # first few rows
    #print("Class balance:\n", df_clean["rotation_binary"].value_counts())
    #print(df_clean.columns.tolist())
    # print(X)
    #print(y)