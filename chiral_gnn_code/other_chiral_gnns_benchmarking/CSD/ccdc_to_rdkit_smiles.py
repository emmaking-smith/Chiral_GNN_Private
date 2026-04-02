'''
We need to make sure that to get the correct
atom-wise Morgan fingerprint that the rdkit
and ccdc atom numbering remains the same.
This shouldn't be too difficult as the trusty
Glasgow Subgraph Solver can come to our rescue.

USE conda ccdc3.10 environment.

ATTN:
• Convert Pi bonds to Single for appropriate subgraph matching
'''

import numpy as np
import os
import subprocess
import argparse

from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

import ccdc
from ccdc.io import EntryReader

# Setting up the GSS
env = os.environ.copy()
env['PATH'] = f"{env['PATH']}:/Users/emmaking-smith/Glasgow_Subgraph_Solver/glasgow-subgraph-solver/build/"

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        type=str,
                        )
    parser.add_argument('--chunk',
                        type=int)
    parser.add_argument('--save-dir',
                        type=str)
    return parser.parse_args()

class CCDC_Mol_GGS_File:
    def __init__(self, save_dir : str):
        self.bond_names = {
            'Single' : 'SINGLE',
            'Double' : 'DOUBLE',
            'Triple' : 'TRIPLE',
            'Aromatic' : 'AROMATIC',
            'Pi' : 'SINGLE',
            'Quadruple' : 'SINGLE'
        }
        self.save_dir = save_dir

    def make_edge_list(self, ccdc_mol : ccdc.molecule.Molecule) -> list:
        '''
        Creating the first half of the GSS file for
        ccdc molecules.
        '''
        bond_info = []
        for bond in ccdc_mol.bonds:
            # Skip the H's
            if bond.atoms[0].atomic_symbol != 'H' and bond.atoms[1].atomic_symbol != 'H':
                bond_info.append(str(bond.atoms[0].index) +
                                 ',' + str(bond.atoms[1].index) + ',' +
                                 self.bond_names[str(bond.bond_type)] +
                                 '\n'
                                 )
        return bond_info

    def make_atom_list(self, ccdc_mol : ccdc.molecule.Molecule) -> list:
        '''
        Creating the second half of the GSS file for
        ccdc molecules.
        '''
        atom_info = []
        for i, atom in enumerate(ccdc_mol.atoms):
            if atom.atomic_symbol != 'H':
                atom_info.append(str(i) + ',,' + atom.atomic_symbol + '\n')
        return atom_info

    def make_ggs_input(self, ccdc_mol : ccdc.molecule.Molecule) -> list:
        '''
        Get the atom numbering and bond info for the ccdc
        molecule.
        Returns a list that can be made saved as a .txt file
        for GSS processing.
        '''
        first_half = self.make_edge_list(ccdc_mol)
        second_half = self.make_atom_list(ccdc_mol)
        ggs_file = first_half + second_half
        return ggs_file

    def ccdc_mols_to_gss_files(self, ccdc_identifiers : list[str], ccdc_mols : list[ccdc.molecule.Molecule]) -> None:
        '''
        The list of ccdc identifiers to their molecules and their corresponding molecules.

        Saves out the files to the save_dir.
        '''
        for id, mol in zip(ccdc_identifiers, ccdc_mols):
            gss_file = self.make_ggs_input(mol)
            save_name = 'ccdc_' + str(id) + '.txt'
            with open(os.path.join(self.save_dir, save_name), 'w') as f:
                for line in gss_file:
                    f.write(line)

class RDKit_Mol_GGS_File:
    def __init__(self, save_dir : str):
        self.save_dir = save_dir

    def make_edge_list(self, rdkit_mol : Chem.rdchem.Mol) -> list:
        '''
        Creating the first half of the GSS file for
        rdkit molecules.
        '''
        bond_info = []
        for bond in rdkit_mol.GetBonds():
            bond_info.append(str(bond.GetBeginAtom().GetIdx()) + ',' +
                             str(bond.GetEndAtom().GetIdx()) + ',' +
                             str(bond.GetBondType()) +
                             '\n'
                             )
        return bond_info

    def make_atom_list(self, rdkit_mol : Chem.rdchem.Mol) -> list:
        '''
        Creating the second half of the GSS file for
        rdkit molecules.
        '''
        atom_info = []
        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            atom_info.append(str(i) + ',,' + atom.GetSymbol() + '\n')
        return atom_info

    def make_ggs_input(self, rdkit_mol : Chem.rdchem.Mol) -> list:
        '''
        Get the atom numbering and bond info for the rdkit
        molecule.
        Returns a list that can be made saved as a .txt file
        for GSS processing.
        '''
        first_half = self.make_edge_list(rdkit_mol)
        second_half = self.make_atom_list(rdkit_mol)
        ggs_file = first_half + second_half
        return ggs_file
    def rdkit_mols_to_gss_files(self, ccdc_identifiers : list[str], rdkit_mol : list[Chem.rdchem.Mol]) -> None:
        '''
        The list of ccdc identifiers to their molecules and their corresponding molecules.

        Saves out the files to the save_dir.
        '''
        for id, mol in zip(ccdc_identifiers, rdkit_mol):
            ggs_file = self.make_ggs_input(mol)
            save_name = 'rdkit_' + str(id) + '.txt'
            with open(os.path.join(self.save_dir, save_name), 'w') as f:
                for line in ggs_file:
                    f.write(line)

def gss_check(file : str) -> bool:
    '''
    Check the mapping worked.
    '''
    check = False
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('mapping ='):
                check = True
    return check

def has_delocalised_bonds(mol : ccdc.molecule.Molecule):
    '''
    Delocalised bonds are too difficult to deal with for mapping
    so we are removing molecules that have them.
    '''
    has_delocalised = False
    bond_types = [str(x.bond_type) for x in mol.bonds]
    if 'Delocalised' in bond_types:
        has_delocalised = True
    return has_delocalised

def main():
    # Set up parser.
    args = init_args()
    parser = EntryReader('CSD')

    # Load in the CCDC identifiers.
    ccdc_identifiers = np.load(args.data_path)
    ccdc_identifiers = ccdc_identifiers[args.chunk : args.chunk+100000]

    new_ccdc_identifiers = [] # updated list of identifiers
    ccdc_mols = []
    rdkit_mols = []
    for identifier in tqdm(ccdc_identifiers):
        try:
            ccdcmol = parser.molecule(identifier)
            if 'Unknown' not in ccdcmol.smiles:
                rdkitmol = Chem.MolFromSmiles(ccdcmol.smiles, sanitize=False)
                if has_delocalised_bonds(ccdcmol) == False:
                    ccdc_mols.append(ccdcmol)
                    rdkit_mols.append(rdkitmol)
                    new_ccdc_identifiers.append(identifier)
        except:
            pass

    # Setup save directory.
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.save_dir, 'mappings')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.save_dir, 'identifiers')).mkdir(parents=True, exist_ok=True)

    # Create the save GSS files.
    CCDC_Files = CCDC_Mol_GGS_File(save_dir=args.save_dir)
    RDKit_Files = RDKit_Mol_GGS_File(save_dir=args.save_dir)

    CCDC_Files.ccdc_mols_to_gss_files(new_ccdc_identifiers, ccdc_mols)
    RDKit_Files.rdkit_mols_to_gss_files(new_ccdc_identifiers, rdkit_mols)

    for id in tqdm(new_ccdc_identifiers):
        command = f"glasgow_subgraph_solver {os.path.join(args.save_dir, 'ccdc_' + str(id) + '.txt')} {os.path.join(args.save_dir, 'rdkit_' + str(id) + '.txt')} > {os.path.join(args.save_dir, 'mappings', 'mapping_' + str(id) + '.txt')}"
        subprocess.run(command,
                       shell=True,
                       env=env,
                       executable='/bin/bash')

    removals = []
    # Checking the mappings and removing all that are not properly mapped.
    for file in os.listdir(os.path.join(args.save_dir, 'mappings')):
        idx = os.path.splitext(file)[0].split('mapping_')[1]
        if gss_check(os.path.join(args.save_dir, 'mappings', file)) == False:
            removals.append(idx)

    # Updating the ccdc_identifiers.
    updated_ccdc_identifiers = new_ccdc_identifiers.copy()
    for item in removals:
        updated_ccdc_identifiers.remove(item)
    assert len(updated_ccdc_identifiers) == len(new_ccdc_identifiers) - len(removals)

    print(f'Removed {len(removals)} from CCDC list.')
    print(f'{len(updated_ccdc_identifiers)} remain...')
    np.save(os.path.join(args.save_dir, 'identifiers', 'updated_ccdc_identifiers_' + str(args.chunk) + '_' + str(args.chunk+100000) + '.npy'), updated_ccdc_identifiers)

if __name__ == '__main__':
    main()



