'''
Creating the adjacency matrices from the CCDC molecules.
'''

import numpy as np
import argparse
import ccdc

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        type=str,
                        )
    parser.add_argument('--chunk',
                        type=int)
    parser.add_argument('--save-name',
                        type=str)
    return parser.parse_args()

class Adjacency_Matrices():
    def __init__(self, max_molecule_size : int):
        self.max_molecule_size = max_molecule_size

    def create_bond_type_adj_matrix(self, mol : ccdc.molecule.Molecule, bond_type : str) -> np.array:
        '''
        Creates a bond-specific adjacency matrix for a given molecule.
        '''
        matrix = np.zeros((self.max_molecule_size, self.max_molecule_size))
        bonds = mol.bonds
        for bond in bonds:
            if str(bond.bond_type) == bond_type:
                matrix[bond.atoms[0].index, bond.atoms[1].index] = 1
                matrix[bond.atoms[1].index, bond.atoms[0].index] = 1
        return matrix

    def create_universal_node_matrix(self, mol : ccdc.molecule.Molecule) -> np.array:
        '''
        Creates the universal node matrix.
        '''
        uni_matrix = np.zeros((self.max_molecule_size, self.max_molecule_size))
        mol_size = len(mol.atoms)
        for i in range(mol_size):
            uni_matrix[i, mol_size] = 1
            uni_matrix[mol_size, i] = 1
        return uni_matrix

def main():
    import pandas as pd
    from ccdc.io import EntryReader
    from tqdm import tqdm
    args = init_args()

    df = pd.read_pickle(args.data_path)
    parser = EntryReader('CSD')

    ccdc_identifiers = df['ccdc_identifier'].tolist()
    ccdc_mols = [parser.molecule(x) for x in ccdc_identifiers]

    bond_types = [
        'Single',
        'Double',
        'Triple',
        'Aromatic',
        'Pi',
        'Quadruple',
        'Delocalised',
    ]

    longest_molecule = 101 # 100 + 1 for universal node.
    AM = Adjacency_Matrices(longest_molecule)

    matrices = []
    for mol in tqdm(ccdc_mols):
        mol.remove_hydrogens() # Removing hydrogens to keep indexing accurate.
        mol_bonds = [str(x.bond_type) for x in mol.bonds]
        mol_length = len(mol.atoms)
        if 'Unknown' in mol_bonds or mol_length > longest_molecule - 1:
            matrices.append(None)
        else:
            bond_type_matrices = [AM.create_bond_type_adj_matrix(mol=mol,
                                                                bond_type=x) for x in bond_types]
            bond_type_matrices += [AM.create_universal_node_matrix(mol=mol)]
            matrices.append(bond_type_matrices)

    df['adj_matrices'] = matrices
    df = df.loc[pd.isna(df['adj_matrices']) == False]
    df.to_pickle(args.save_name)

