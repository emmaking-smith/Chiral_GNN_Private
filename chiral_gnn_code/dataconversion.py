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
    radius: int = 3,                    # ECFP4
    n_bits: int = 1028,
    include_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      X: (N, n_bits) float32 fingerprint matrix
      y: (N,) int labels (Rotation '+'->1, '-'->0)
      df_clean: original cols + helper cols for traceability
    Drops rows with invalid SMILES or missing/unmappable rotation.
    """
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{smiles_col}' not found. Got: {df.columns.tolist()}")
    if rotation_col not in df.columns:
        raise KeyError(f"Rotation column '{rotation_col}' not found. Got: {df.columns.tolist()}")

    # Clean raw columns
    smiles = df[smiles_col].astype(str).fillna("").str.strip()
    rotation_raw = df[rotation_col].astype(str).fillna("").str.strip()

    # Make labels: '+' -> 1, '-' -> 0
    # (adjust mapping here if your data uses different symbols)
    y_series = rotation_raw.map({"+": 1, "-": 0})

    # Morgan generator (bit fingerprint)
    mgen = rdFP.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=include_chirality)

    X = np.zeros((len(smiles), n_bits), dtype=np.int8)
    valid = np.ones(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if (mol is None) or pd.isna(y_series.iat[i]):
            valid[i] = False
            continue
        try:
            mfp = mgen.GetFingerprint(mol)                       # ExplicitBitVect
            DataStructs.ConvertToNumpyArray(mfp, X[i])           # 0/1 row
        except Exception:
            valid[i] = False

    # Keep only valid & labeled rows
    X = X[valid].astype("float32")
    y = y_series[valid].to_numpy(dtype=int)

    # Return a cleaned df for traceability (optional)
    df_clean = df.loc[valid].reset_index(drop=True)
    df_clean["rotation_binary"] = y

    return X, y, df_clean

if __name__ == "__main__":
    X, y, df_clean = build_dataset("./data/processed_data.csv")

    # basic info
   # print("X shape:", X.shape)
    print("y shape:", y.shape)
   # print(df_clean.head())            # first few rows
    #print("Class balance:\n", df_clean["rotation_binary"].value_counts())
   # print(df_clean.columns.tolist())
    print(y)