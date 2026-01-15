'''
Functions that take a chemical SMILES string and turn
it into the correct format for pytorch geometric datasets.

'''

import torch
import numpy as np
from typing import Union
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolTransforms, rdFingerprintGenerator


class Node_Info:
    '''
    Gets the atomic information from the
    SMILES string.
    • Atom Idx
    • hybridization
    • chirality type
    • x,y,z
    • neighbor "up" / "down" / "in plane"
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

    def morganfingerprint(self, mol : Chem.rdchem.Mol, atom_idx) -> list[int]:
        fpgen= rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
        atom_fp_rdkit = fpgen.GetFingerprint(mol,fromAtoms=[atom_idx])
        atom_fp_array = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(atom_fp_rdkit, atom_fp_array)
        atom_fp = atom_fp_array.tolist()
        return atom_fp


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
            'xyz' : Feature_Functions.standard_mol_xyz,
            'mpg': Feature_Functions.morganfingerprint
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

    def create_atomic_features(self, atom : Chem.rdchem.Atom, mol: Chem.rdchem.Mol,
                               xyz_coordinates : Union[np.array, None],
                               atom_idx : int,
                               label : int) -> list[list[int]]:
        atomic_features = []
        for feat in self.features:
            if feat == 'mpg':
                atom_fp = self.features_dict['mpg'](mol=mol, atom_idx=atom_idx)
                atom_fp.append(label)
                atomic_features.append(atom_fp)

            if feat != 'xyz' and feat != 'mpg':
                atomic_features.append(self.features_dict[feat](atom))

        if 'xyz' in self.features:
            atomic_features += xyz_coordinates[atom_idx].tolist()

        if 'mpg' not in self.features:
            atomic_features.append(label)

        return atomic_features

    def create_node_features(self, mol : Chem.rdchem.Mol, label) -> list[list[int]]:
        all_atom_features = []
        if 'xyz' in self.features:
            xyz_coordinates = self.features_dict['xyz'](mol)
        else:
            xyz_coordinates = None
        for i, atom in enumerate(mol.GetAtoms()):

            all_atom_features.append(self.create_atomic_features(atom=atom, mol=mol,
                                                             xyz_coordinates=xyz_coordinates,
                                                             atom_idx=i,
                                                                 label=label)
                                     )
        return all_atom_features

    def smiles_to_graph(self, smiles : str, label : int) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        Creating the edge list and node features for a single molecule.
        '''
        mol = Chem.MolFromSmiles(smiles)
        edge_tuples, bond_types = self.find_edge_indices(mol=mol)
        node_info = self.create_node_features(mol=mol, label=label)

        return (torch.tensor(edge_tuples, dtype=torch.long).reshape(-1,2),
                torch.tensor(node_info, dtype=torch.float).reshape((len(mol.GetAtoms()),-1)),
                torch.tensor(bond_types, dtype=torch.float).reshape((-1,1))
                )




