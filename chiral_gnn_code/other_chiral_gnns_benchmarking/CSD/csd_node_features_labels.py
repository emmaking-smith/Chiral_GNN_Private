'''
Get the molecular information from the CSD database.
• atomic number
• hybridization
• R vs S
• xyz
• SMILES string *** with CCDC atom indexing NOT RDKit indexing ***
• bond lengths
• mol angles
'''

import os
import numpy as np
import itertools
import argparse
import json

import ccdc
from ccdc.descriptors import MolecularDescriptors
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

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

class CCDC_Parsing:
    '''
    Functions needed to get atomic features.
    '''
    def __init__(self):
        self.feature_dict = {
            'atomic number' : self.find_atom_number,
            'chirality type': self.find_chirality,
            'hybridization' : self.find_hybridization,
            'xyz' : self.find_xyz
        }

    def find_atom_number(self, atom: ccdc.molecule.Atom) -> int:
        return atom.atomic_number

    def find_xyz(self, atom : ccdc.molecule.Atom) -> np.array:
        return np.array(atom.coordinates)

    def find_hybridization_non_metals(self, atom : ccdc.molecule.Atom) -> int:
        '''
        Non-metals take the most unsaturated bond as their hybridization.
        '''
        all_bonds = atom.bonds
        all_bonds = [str(x.bond_type) for x in all_bonds]
        if 'Triple' in all_bonds:
            hybridization = 2
        elif 'Double' in all_bonds or 'Pi' in all_bonds or 'Aromatic' in all_bonds:
            hybridization = 3
        else:
            hybridization = 4
        return hybridization

    def find_hybridization_metals(self, atom : ccdc.molecule.Atom) -> int:
        '''
        Metals use the number of neighbors to determine hybridization.
        THIS IS OBVIOUSLY NOT PERFECT BUT GOOD ENOUGH!
        '''
        number_of_neighbors = len(atom.neighbours)
        hybridization = 0
        if number_of_neighbors <= 2 and number_of_neighbors >= 1:
            hybridization = 2
        elif number_of_neighbors == 3:
            hybridization = 3
        elif number_of_neighbors == 4:
            hybridization = 5
        elif number_of_neighbors == 5:
            hybridization = 6
        elif number_of_neighbors == 6:
            hybridization = 7
        elif number_of_neighbors > 6:
            hybridization = 8
        return hybridization

    def find_hybridization(self, atom : ccdc.molecule.Atom) -> int:
        '''
        0 = unspecified
        1 = s (hydrogen)
        2 = sp (linear)
        3 = sp2 (trigonal planar)
        4 = sp3 (tetrahedral)
        5 = sp2d (square planar)
        6 = sp3d (trigonal bipyramidal)
        7 = sp3d2 (octahedral)
        8 = other
        '''
        if atom.is_metal == True:
            hybridization = self.find_hybridization_metals(atom)
        else:
            hybridization = self.find_hybridization_non_metals(atom)
        return hybridization

    def find_chirality(self, atom : ccdc.molecule.Atom) -> int:
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
        # The hybridization encoding -> chirality encoding
        # square planar hybrid. = 5, trigonal pyramidal hybrid. = 6, octahedral hybrid. = 7, other hybrid. = 8
        hybridization_dict = {5 : 6, 6 : 7, 7: 8, 8 : 3}
        is_stereocenter = atom.is_chiral
        chirality_type = 0
        if is_stereocenter == True:
            hybridization = self.find_hybridization(atom)
            # tetrahedral first.
            if hybridization == 4:
                chirality = atom.chirality
                if chirality == 'R':
                    chirality_type = 1
                elif chirality == 'S':
                    chirality_type = 2
                else:
                    chirality_type = 4
            elif hybridization < 4:
                chirality_type = 3
            # Then deal with metals.
            else:
                chirality_type = hybridization_dict[hybridization]
        return chirality_type

    def find_mol_features(self, mol : ccdc.molecule.Molecule, feature : str) -> list:
        '''
        Finding a specific feature for a given CCDC molecule
        '''
        features = []
        for atom in mol.atoms:
            if atom.atomic_symbol != 'H':
                features.append(self.feature_dict[feature](atom))
        return features

class CCDC_Labels:
    '''
    Creating the labels (bond lengths & angles)
    for each entry.
    '''
    def __init__(self):
        pass

    def find_bond_length(self, bond : ccdc.molecule.Bond) -> float:
        '''
        Find the bond length.
        '''
        return bond.length

    def find_angles(self, central_atom : ccdc.molecule.Atom) -> list[float]:
        '''
        Find angles associated with an atom.
        '''
        angles = []
        for i, j in itertools.combinations(range(len(central_atom.neighbours)), 2):
            if central_atom.neighbours[i].atomic_symbol != 'H' and central_atom.neighbours[j].atomic_symbol != 'H':
                angles.append(
                    MolecularDescriptors.atom_angle(central_atom.neighbours[i],
                               central_atom,
                               central_atom.neighbours[j])
                )
        return angles

    def pad(self, bond_or_angles_list : list[float], max_value : int) -> list[float]:
        '''
        Pads out bond lengths / angles for a single molecule to the maximum
        number of bonds / angles.
        '''
        padding_size = max_value - len(bond_or_angles_list)
        bond_or_angles_list = bond_or_angles_list + [0.] * padding_size
        return bond_or_angles_list

    def create_bond_list(self, mol : ccdc.molecule.Molecule) -> list[float]:
        '''
        Create the list of bond lengths of atoms to their non-hydrogen neighbors.
        '''
        bond_list = []
        for i in range(len(mol.atoms)):
            bonds_containing_atom = mol.atoms[i].bonds
            atom_bonds = []
            for bond in bonds_containing_atom:
                atom_bonds.append(self.find_bond_length(bond))
            bond_list.append(atom_bonds)
        return bond_list

    def create_angle_list(self, mol : ccdc.molecule.Molecule) -> list[float]:
        '''
        Create the list of bond angles of non-hydrogen atoms to their non-hydrogen neighbors.
        '''
        angle_list = []
        for atom in mol.atoms:
            angle_list.append(self.find_angles(atom))
        return angle_list

class Atomistic_Morgan_Fingerprints():
    def __init__(self, radius : int=2, fpSize : int=2048):
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                                          fpSize=fpSize,
                                                                          includeChirality=True)

    def create_rdkit_molecule(self, ccdc_mol : ccdc.molecule.Molecule) -> Chem.rdchem.Mol:
        '''
        Creates an rdkit molecule with the exact indexing as the ccdc molecule.
        '''
        sdf_string = ccdc_mol.to_string('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_string, sanitize=False)
        return rdkit_mol

    def create_single_fingerprint(self, mol: ccdc.molecule.Molecule) -> list[int]:
        '''
        Creating a single atom-wise chiral morgan fingerprint. ccdc_id refers to the
        CCDC identifier of the molecule because that is how the GSS mappings are named.
        '''
        rdkit_mol = self.create_rdkit_molecule(mol)
        fingerprint = []
        for i, atom in enumerate(mol.atoms):
            fingerprint.append(list(self.morgan_generator.GetFingerprint(rdkit_mol,
                                                                         fromAtoms=[i]
                                                                         )
                                    ))
        return fingerprint

    # def gss_output_to_index_dictionary(self, ccdc_id: str) -> dict:
    #     '''
    #     Takes in the output txt file from running the GSS and creates
    #     a dictionary where each key is the CCDC atom index and each
    #     value is the corresponding RDKit atom index.
    #
    #             { CCDC Atom 0 : RDKit Atom X, CCDC Atom 1 : RDKit Atom Y, ... }
    #
    #     All you need is the CCDC identifier of a given molecule (ccdc_id).
    #     '''
    #     ccdc_to_rdkit_indexing = {}
    #     with open(os.path.join(self.gss_mapping_path, 'mapping_' + ccdc_id + '.txt'), 'r') as f:
    #         for line in f:
    #             if line.startswith('mapping ='):
    #                 line = line.replace('mapping = ', '').strip()
    #                 line = line.split(') (')
    #                 for mapping in line:
    #                     mapping = mapping.strip('()')
    #                     key, value = mapping.split('->')
    #                     ccdc_to_rdkit_indexing[int(key.strip())] = int(value.strip())
    #     return ccdc_to_rdkit_indexing

    # def create_single_fingerprint(self, mol : ccdc.molecule.Molecule, ccdc_id : str) -> list[int]:
    #     '''
    #     Creating a single atom-wise chiral morgan fingerprint. ccdc_id refers to the
    #     CCDC identifier of the molecule because that is how the GSS mappings are named.
    #     '''
    #     ccdc_to_rdkit_index_dictionary = self.gss_output_to_index_dictionary(ccdc_id)
    #     rdkit_mol = Chem.MolFromSmiles(mol.smiles)
    #     fingerprint = []
    #     for i, atom in enumerate(mol.atoms):
    #         fingerprint.append(list(self.morgan_generator.GetFingerprint(rdkit_mol,
    #                                                                      fromAtoms=[ccdc_to_rdkit_index_dictionary[i]]
    #                                                                      )
    #                                 ))
    #     return fingerprint

def main():
    import pandas as pd
    from ccdc.io import EntryReader
    from tqdm import tqdm

    args = init_args()

    # Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ccdc_identifiers = np.load(args.data_path)
    # ccdc_identifiers = ccdc_identifiers[args.chunk : args.chunk + 100000]
    #
    # # Get CCDC molecules
    # parser = EntryReader('CSD')
    # ccdc_mols = []
    # for id in ccdc_identifiers:
    #     try:
    #         ccdc_mols.append(parser.molecule(id))
    #     except:
    #         pass
    #
    # del ccdc_identifiers
    #
    # # Create dataframe.
    # df = pd.DataFrame()
    #
    # atomic_numbers = []
    # chirality = []
    # hybridization = []
    # xyz = []
    # bond_lists = []
    # angle_lists = []
    # new_ids = []
    #
    # for mol in tqdm(ccdc_mols):
    #     try:
    #         mol.remove_hydrogens()
    #         bond_lists.append(CCDC_Labels().create_bond_list(mol))
    #         angle_lists.append(CCDC_Labels().create_angle_list(mol))
    #         atomic_numbers.append(CCDC_Parsing().find_mol_features(mol, 'atomic number'))
    #         chirality.append(CCDC_Parsing().find_mol_features(mol, 'chirality type'))
    #         hybridization.append(CCDC_Parsing().find_mol_features(mol, 'hybridization'))
    #         xyz.append(CCDC_Parsing().find_mol_features(mol, 'xyz'))
    #         new_ids.append(mol.identifier)
    #     except:
    #         pass
    #
    # df['ccdc_identifier'] = new_ids
    # df['atomic_numbers'] = atomic_numbers
    # df['chirality'] = chirality
    # df['hybridization'] = hybridization
    # df['xyz'] = xyz
    # df['bond_lengths'] = bond_lists
    # df['angles'] = angle_lists
    #
    # df.to_pickle(args.save_name +  '_' + str(args.chunk) + '_' + str(args.chunk + 100000) + '.pickle')
    # df.to_pickle(os.path.join(args.save_path, str(args.chunk) + '_' + str(args.chunk + 10000) + '.pickle'))


    '''
    chiral Morgan FP
    '''

    ccdc_identifiers = np.load(args.data_path)

    parser = EntryReader('CSD')
    ccdc_mols = [parser.molecule(x) for x in ccdc_identifiers]
    # ccdc_mols = ccdc_mols[args.chunk : args.chunk + 100000]
    AMF = Atomistic_Morgan_Fingerprints()

    sdf_strings = {}

    for mol in tqdm(ccdc_mols):
        mol.remove_hydrogens()
        sdf_strings[mol.identifier] = mol.to_string('sdf')

    with open(args.save_name, 'w') as f:
        json.dump(sdf_strings, f)

if __name__ == '__main__':
    main()

