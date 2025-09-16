'''
Finding the direction of specific rotation for
Reaxys molecules.
'''

import numpy as np
import pandas as pd
from rdkit import Chem

def get_polarization(names : list[str]) -> list[str]:
    '''
    Finds all recorded directions of rotation for
    a SMILES string.
    '''
    polarization = []
    names = names.split(';')
    names = [x.strip() for x in names]
    for name in names:
        if '(+)' in name:
            polarization.append('+')
        if '(-)' in name:
            polarization.append('-')
        else:
            polarization.append('')
    return polarization

def get_sign(polarization : list[str]) -> str:
    '''
    Determines the rotation for a SMILES string.
    If multiple are present or no sign is present,
    records separate messages.
    '''
    if '+' in polarization and '-' in polarization:
        sign = 'Check manually! Both (+) and (-)'
    elif '+' in polarization:
        sign = '+'
    elif '-' in polarization:
        sign = '-'
    else:
        sign = 'no sign present'
    return sign

def get_inchi(smiles : str) -> str:
    '''
    Converts the SMILES string to an Inchi string.
    '''
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol)


def main():
    df = pd.read_pickle('data/combined_reaxys_data.pickle')
    # Remove duplicated SMILES.
    df = df.drop_duplicates(subset=['SMILES'], keep='first')

    # Remove SMILES without stereochemical conformation info.
    df = df.loc[df['SMILES'].str.contains('@', case=False)]
    df = df.loc[df['SMILES'] != '[H][C@]12CON[C]1(=COC2)C1=CC=CC=C1F']
    df = df.loc[df['SMILES'] != 'Cl.CC[N@@H]1(O)C=C2C(C(=O)N3CC4=C(CN5CCC(C)CC5)OC(C)=C(Cl)C=C4C=C23)C2=C1CC(=O)CC2']
    # Get the direction of rotation.
    df['Rotation'] = [get_sign(get_polarization(x)) for x in df['Chemical Name']]

    # Reset index.
    df = df.reset_index(drop=True)

    # Add InCHI
    inchis = []
    for s in df['SMILES']:
        try:
            inchis.append(get_inchi(s))
        except:
            inchis.append(np.nan)

    df['InCHI'] = inchis

    df = df.loc[(df['Rotation'] == '+') | (df['Rotation'] == '-')]

    df.to_pickle('data/processed_data.pickle')

if __name__ == '__main__':
    main()