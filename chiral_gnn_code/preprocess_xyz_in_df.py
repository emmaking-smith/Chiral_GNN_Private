'''
Adding the xyz coordinates to the dataframe for preprocessing.
'''

import pandas as pd
from rdkit import Chem

from smiles_to_geometric_data import Node_Info

def main():
    df = pd.read_csv('data/processed_data.csv', index_col=0)
    df['SMILES'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in df['SMILES']]
    coordinates = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        coordinates.append(Node_Info().standard_mol_xyz(mol))
    df['xyz'] = coordinates

    df.to_pickle('processed_data_with_xyz.pickle')

if __name__ == '__main__':
    main()