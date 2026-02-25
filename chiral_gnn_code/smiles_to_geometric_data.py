'''
Functions that take a chemical SMILES string and turn
it into the correct format for pytorch geometric datasets.

'''
import torch
import numpy as np
from typing import Callable
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import rdFingerprintGenerator

class Node_Info:
    '''
    Gets the atomic information from the
    SMILES string.
    • Atom Idx
    • hybridization
    • chirality type
    • x,y,z
    '''
    def __init__(self):
        pass

    def find_atomic_num(self, atom : Chem.rdchem.Atom) -> int:
        '''
        Finds the atomic numbers of the atoms in molecules
        '''
        return atom.GetAtomicNum()

    def find_hybridization(self, atom : Chem.rdchem.Atom) -> int:
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

    def find_chiral_type(self, atom : Chem.rdchem.Atom) -> int:
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

    def standard_mol_xyz(self, mol : Chem.rdchem.Mol) -> np.array:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        try:
            AllChem.EmbedMolecule(mol, params)
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            principle_axes, moments = rdMolTransforms.ComputePrincipalAxesAndMoments(conf)
            rotation_matrix = principle_axes.T
            positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            center_of_mass = np.mean(positions, axis=0)
            centered_positions = positions - center_of_mass
            new_positions = np.dot(centered_positions, rotation_matrix)
        except:
            new_positions = np.zeros((len(mol.GetAtoms()), 3))
        return new_positions

class Create_Graph:
    def __init__(self, features : list[str]):
        '''
        Creating a pytorch geometric graph with
        specific features.

        Features can be:
        • atomic number
        • hybridization
        • chirality type
        • xyz
        '''
        self.features = features

        Feature_Functions = Node_Info()

        self.features_dict = {
            'atomic number' : Feature_Functions.find_atomic_num,
            'hybridization' : Feature_Functions.find_hybridization,
            'chirality type' : Feature_Functions.find_chiral_type,
        }

    def find_bond_begin_end_type(self, bond : Chem.rdchem.Bond) -> tuple[list[list[int]], str]:
        '''
        Finds the atom index of the beginning of the bond, the end
        of the bond, and the bond type.
        '''
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        type = bond.GetBondType()
        edge_pair = [[begin_idx, end_idx], [end_idx, begin_idx]]
        return edge_pair, type

    def find_edge_indices(self, mol : Chem.rdchem.Mol):
        '''
        Finds which atoms are bonded to one another.
        '''
        bonds = mol.GetBonds()
        edge_pairings = []
        edge_types = []
        for bond in bonds:
            bond_pair, bond_type = self.find_bond_begin_end_type(bond)
            edge_pairings += bond_pair
            edge_types.append(int(bond_type))
            edge_types.append(int(bond_type))
        return edge_pairings, edge_types

    def create_atomic_features(self, atom : Chem.rdchem.Atom,
                               atom_idx : int,
                               xyz_coordinates : np.array
                               ) -> list[list[int]]:
        atomic_features = []
        for feat in self.features:
            if feat != 'xyz':
                atomic_features.append(self.features_dict[feat](atom))

        if 'xyz' in self.features:
            atomic_features += xyz_coordinates[atom_idx].tolist()
        return atomic_features

    def create_node_features(self, mol : Chem.rdchem.Mol,
                             xyz_coordinates : np.array
                             ) -> list[list[int]]:
        all_atom_features = []

        for i, atom in enumerate(mol.GetAtoms()):
            all_atom_features.append(self.create_atomic_features(atom=atom,
                                                             xyz_coordinates=xyz_coordinates,
                                                             atom_idx=i)
                                     )
        return all_atom_features

    def smiles_to_graph(self, smiles : str,
                        xyz_coordinates : np.array
                        ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        Creating the edge list and node features for a single molecule.
        '''
        mol = Chem.MolFromSmiles(smiles)
        edge_tuples, bond_types = self.find_edge_indices(mol=mol)
        node_info = self.create_node_features(mol=mol, xyz_coordinates=xyz_coordinates)
        return (torch.tensor(edge_tuples, dtype=torch.long).reshape(-1,2),
                torch.tensor(node_info, dtype=torch.float).reshape((len(mol.GetAtoms()),-1)),
                torch.tensor(bond_types, dtype=torch.float).reshape((-1,1))
                )


class Chiral_MFP_Graph(Create_Graph):
    def __init__(self, radius : int=2, fpSize : int=2048):
        '''
        Creating a pytorch geometric graph
        using the atom-wise chiral Morgan fingerprint
        as each node's feature.

        Args:
            radius (int): radius for morgan fingerprint.
            fpSize (int): fingerprint size.
        '''
        super(Chiral_MFP_Graph, self).__init__(features=[])
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                                          fpSize=fpSize,
                                                                          includeChirality=True)

    def create_atom_wise_MFP(self, mol : Chem.rdchem.Mol, atom_idx : int) -> list[int]:
        '''
        Creating atom-wise Morgan Fingerprint.
        '''
        atom_i_fingerprint = self.morgan_generator.GetFingerprint(mol, fromAtoms=[atom_idx])
        return list(atom_i_fingerprint)

    # def create_atom_wise_counts_MFP(self, mol : Chem.rdchem.Mol, atom_idx : int) -> list[int]:
    #     '''
    #     Creating atom-wise Morgan Fingerprint.
    #     '''
    #     atom_i_fingerprint = self.morgan_generator.GetCountFingerprint(mol, fromAtoms=[atom_idx])
    #     return list(atom_i_fingerprint)

    def create_atom_wise_MFP_node_features(self, mol : Chem.rdchem.Mol) -> list[list[int]]:
        '''
        Create the atom-wise Morgan fingerprints for a molecule.
        '''
        all_atom_features = []

        for i in range(len(mol.GetAtoms())):
            all_atom_features.append(self.create_atom_wise_MFP(mol=mol,
                                                               atom_idx=i))
        return all_atom_features

    def smiles_to_MFP_graph(self, smiles : str) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        Creating the edge list and atom-wise Morgan fingerprints for a single molecule.
        '''
        mol = Chem.MolFromSmiles(smiles)
        edge_tuples, bond_types = self.find_edge_indices(mol=mol)
        node_info = self.create_atom_wise_MFP_node_features(mol=mol)
        return (torch.tensor(edge_tuples, dtype=torch.long).reshape(-1, 2),
                torch.tensor(node_info, dtype=torch.float).reshape((len(mol.GetAtoms()), -1)),
                torch.tensor(bond_types, dtype=torch.float).reshape((-1, 1))
                )

class Molformer_Graph(Create_Graph):
    def __init__(self):
        '''
        Creating a pytorch geometric graph
        using the different rootedAtAtom indices
        converted to molformer embeddings.
        '''
        super(Molformer_Graph, self).__init__(features=[])
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                               deterministic_eval=True,
                                               trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                                       trust_remote_code=True)

    def molformer_embedding(self, smiles : str) -> list[int]:
        '''
        Creating atom-wise Morgan Fingerprint.
        '''
        inputs = self.tokenizer([smiles], padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.tolist()  # list of length 768

    def create_atom_wise_molformer_nodes(self, mol : Chem.rdchem.Mol) -> list[list[int]]:
        '''
        Create the atom-wise Morgan fingerprints for a molecule.
        '''
        all_atom_features = []

        for i in range(len(mol.GetAtoms())):
            smiles_starting_at_index_i = Chem.MolToSmiles(mol, rootedAtAtom=i)
            all_atom_features.append(self.molformer_embedding(smiles=smiles_starting_at_index_i))
        return all_atom_features

    def smiles_to_molformer_graph(self, smiles : str) -> tuple[torch.tensor, torch.tensor]:
        '''
        Creating the edge list and atom-wise Morgan fingerprints for a single molecule.
        '''
        mol = Chem.MolFromSmiles(smiles)
        edge_tuples, bond_types = self.find_edge_indices(mol=mol)
        # node_info = self.create_atom_wise_molformer_nodes(mol=mol)
        return (torch.tensor(edge_tuples, dtype=torch.long).reshape(-1, 2),
                torch.tensor(bond_types, dtype=torch.float).reshape((-1, 1))
                )

class Morgan_Concat_Atom_Feats(Create_Graph):
    '''
    Concatenating a from atom Morgan
    with atom-wise features.
    '''
    def __init__(self, radius : int=2, fpSize : int=2048, features : list[str]=['atomic number', 'chirality type', 'hybridization']):
        super(Morgan_Concat_Atom_Feats, self).__init__(features=features)
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                                          fpSize=fpSize,
                                                                          includeChirality=True)

    def create_atom_wise_MFP(self, mol: Chem.rdchem.Mol, atom_idx: int) -> list[int]:
        '''
        Creating atom-wise Morgan Fingerprint.
        '''
        atom_i_fingerprint = self.morgan_generator.GetFingerprint(mol, fromAtoms=[atom_idx])
        return list(atom_i_fingerprint)

    def create_concat_node_feats(self, mol : Chem.rdchem.Mol) -> np.array:
        atom_features = self.create_node_features(mol=mol,
                                                  xyz_coordinates=None)
        morgan_features = [self.create_atom_wise_MFP(mol=mol, atom_idx=x) for x in range(len(mol.GetAtoms()))]
        node_features = np.concat((morgan_features, atom_features), axis=1)
        return node_features

    def smiles_to_MFP_concat_atom_feat_graph(self, smiles : str) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        Creating the edge list and atom-wise Morgan fingerprints for a single molecule.
        '''
        mol = Chem.MolFromSmiles(smiles)
        edge_tuples, bond_types = self.find_edge_indices(mol=mol)
        node_info = self.create_concat_node_feats(mol=mol)
        return (torch.tensor(edge_tuples, dtype=torch.long).reshape(-1, 2),
                torch.tensor(node_info, dtype=torch.float).reshape((len(mol.GetAtoms()), -1)),
                torch.tensor(bond_types, dtype=torch.float).reshape((-1, 1))
                )

import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk',
                        type=int)
    return parser.parse_args()


def main():
    import pandas as pd
    from tqdm import tqdm
    args = init_args()
    df = pd.read_pickle('data/processed_data_with_xyz.pickle')
    df = df[args.chunk:args.chunk+15000].reset_index(drop=True)
    MG = Molformer_Graph()
    node_infos = []
    for smiles in tqdm(df['SMILES']):
        node_infos.append(MG.create_atom_wise_molformer_nodes(Chem.MolFromSmiles(smiles)))
    df['node_info'] = node_infos
    df.to_pickle('data/processed_data_with_molformer_node_info_' + str(args.chunk) + '_' + str(args.chunk+15000) + '.pickle')

if __name__ == '__main__':
    main()