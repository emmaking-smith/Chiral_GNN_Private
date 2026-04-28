'''
Preprocessing the optical rotation dataset.

NOTE: xyz is already present in the dataframe.
'''

import pandas as pd
import argparse
import os

from rdkit import Chem

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default='/exports/csce/eddie/chem/groups/King-Smith/EKS_musings/chiral_gnn_code/data/processed_data_with_xyz.pickle')
    parser.add_argument('--save-dir',
                        type=str,
                        default='/exports/csce/eddie/chem/groups/King-Smith/EKS_musings/chiral_gnn_code/data')
    return parser.parse_args()

class OR_Atom_Feats:
    def __init__(self):
        self.bond_types = [
            'Single',
            'Double',
            'Triple',
            'Aromatic',
            'Pi',
            'Quadruple',
            'Delocalised',
        ]

    def find_atomic_num(self, atom: Chem.rdchem.Atom) -> int:
        '''
        Finds the atomic numbers of the atoms in molecules
        '''
        return atom.GetAtomicNum()

    def make_atomic_num_entry(self, mol : Chem.rdchem.Mol) -> list[int]:
        '''
        Gets the atomic numbers for each atom in a molecule.
        '''
        return [self.find_atomic_num(x) for x in mol.GetAtoms()]

    def find_hybridization(self, atom: Chem.rdchem.Atom) -> int:
        '''
        Finds the hybridization of an atom,
        returns it as an integer.
        0 = unspecified
        1 = s
        2 = sp
        3 = sp2
        4 = sp3
        5 = sp2d
        6 = sp3d
        7 = sp3d2
        8 = other
        '''
        return int(atom.GetHybridization())

    def make_hybridization_entry(self, mol : Chem.rdchem.Mol) -> list[int]:
        '''
        Gets the hybridization for each atom in a molecule.
        '''
        return [self.find_hybridization(x) for x in mol.GetAtoms()]

    def find_chiral_type(self, atom: Chem.rdchem.Atom) -> int:
        '''
        Encodes chiralty type.
        0 = unspecified
        1 = clockwise
        2 = counterclockwise
        3 = other
        4 = tetrahedral
        5 = allene
        6 = square planar
        7 = trigonal bipyramidal
        8 = octahedral
        '''
        return int(atom.GetChiralTag())

    def make_chirality_entry(self, mol : Chem.rdchem.Mol) -> list[int]:
        '''
        Gets the chirality type for each atom in a molecule.
        '''
        return [self.find_chiral_type(x) for x in mol.GetAtoms()]

    def find_bond_indices(self, mol : Chem.rdchem.Mol, bond_type : str) -> list[tuple]:
        '''
        Find the indices for each bond type for a given molecule.
        '''
        bond_indices = []
        for bond in mol.GetBonds():
            if str(bond.GetBondType()) == bond_type.upper():
                bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        return bond_indices

    def make_adj_indices_dict(self, mol : Chem.rdchem.Mol) -> dict:
        '''
        Creates the dictionary for atom connectivity.
            { 'Single' : [(idx 0, idx 1), (idx 1, idx 2) ...], 'Double' : [...], ...}
        '''
        adj_indices_dict = {}
        for bond_type in self.bond_types:
            adj_indices_dict[bond_type] = self.find_bond_indices(mol, bond_type)
        return adj_indices_dict

    def process_df(self, df : pd.DataFrame) -> pd.DataFrame:
        '''
        Creates the atomic feature columns
        in the dataframe.
        '''
        smiles = df['SMILES'].tolist()
        mols = [Chem.MolFromSmiles(x) for x in smiles]

        # Features
        atomic_numbers = [self.make_atomic_num_entry(x) for x in mols]
        chirality = [self.make_chirality_entry(x) for x in mols]
        hybridization = [self.make_hybridization_entry(x) for x in mols]
        num_heavy_atoms = [len(x.GetAtoms()) for x in mols]

        # Adjacency Info
        adj_indices_dict = [self.make_adj_indices_dict(x) for x in mols]

        # Add info to dataframe.
        df['atomic_numbers'] = atomic_numbers
        df['chirality'] = chirality
        df['hybridization'] = hybridization
        df['num_heavy_atoms'] = num_heavy_atoms
        df['adj_indices_dict'] = adj_indices_dict

        return df

def main():
    args = init_args()

    df = pd.read_pickle(args.data)

    df = OR_Atom_Feats().process_df(df)
    df.to_pickle(os.path.join(args.save_dir, 'csd_preprocessed_data.pickle'))

if __name__ == '__main__':
    main()