from networkx.algorithms.distance_measures import radius
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

df = pd.read_csv("./data/processed_data.csv")

# make sure you select a Series (not a DataFrame)
smiles_series = df['SMILES'].astype(str).fillna("")

fmgen = AllChem.GetMorganGenerator(radius=3,includeChirality=True)

mfps=[]
for x in smiles_series:
    mol = Chem.MolFromSmiles(x)
    mfp=fmgen.GetFingerprint(mol)
    mfps.append(mfp)








