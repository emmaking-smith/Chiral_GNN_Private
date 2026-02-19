'''
Testing out fundamental chemical features -> classic ML does.

• atomic number
• hybridization
• chirality type (e.g., R/S)
• atomic coordinates (x,y,z)
'''

import pandas as pd
import numpy as np
import argparse

from rdkit import Chem
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from smiles_to_geometric_data import Create_Graph

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',
                        type=int)
    parser.add_argument('--model',
                        type=str,
                        choices=['RF', 'ExtraTrees', 'GradBoost', 'SVM'])
    parser.add_argument('--features',
                        nargs='+',
                            choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                        help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    return parser.parse_args()

class Chem_Feature_Embeddings:
    def __init__(self, features : list[str], longest_molecule : int):
        '''
        Turn the SMILES strings into mol feature vectors where:
            mol_features = [atom 0 feat 0, atom 0 feat 1, atom 0 feat 2, ... , atom N feat M]
        Padded out to longest molecule.
        '''

        self.processing = Create_Graph(features=features)
        self.longest_molecule = longest_molecule

    def embed_one_smiles(self, smiles : str, xyz_coordinates : np.array) -> np.array:
        '''
        Embed one SMILES string. You get the xyz coordinates from the dataframe.
        '''
        mol = Chem.MolFromSmiles(smiles)
        mol_features = np.array(self.processing.create_node_features(mol=mol,
                                                            xyz_coordinates=xyz_coordinates))
        mol_features = mol_features
        mol_features = self.pad_mol_vector(mol_features)
        return mol_features

    def pad_mol_vector(self, mol_features : np.array) -> np.array:
        '''
        Pads with zeros the mol features so that:
             mol_features = [atom 0 feat 0, atom 0 feat 1, atom 0 feat 2, ... , atom N feat M]
         becomes:
            mol_features = [atom 0 feat 0, atom 0 feat 1, atom 0 feat 2, ... , atom N feat M, 0, 0, 0, ...]
        '''

        padding = np.zeros((self.longest_molecule - mol_features.shape[0], mol_features.shape[1]))
        return np.concat((mol_features, padding)).reshape((-1))

    def create_inputs(self, df : pd.DataFrame) -> np.array:
        '''
        Since we'll be changing the features, make
        the inputs on demand.
        '''
        inputs = []
        for i in range(len(df)):
            inputs.append(self.embed_one_smiles(smiles=df.loc[i, 'SMILES'],
                                                xyz_coordinates=df.loc[i, 'xyz']))
        inputs = np.array(inputs).reshape((len(df), -1))
        return inputs

def predictions(train_inputs : np.array,
                train_labels : np.array,
                test_inputs : np.array,
                model ) -> np.array:
    model.fit(train_inputs, train_labels)
    preds = model.predict(test_inputs)
    return preds

def scorings(test_labels : np.array, predictions : np.array) -> tuple[float, float, float, float]:
    '''
    F-score, precision, recall, accuracy.
    '''
    f_score = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    return f_score, precision, recall, accuracy

def find_longest_molecule(df : pd.DataFrame) -> int:
    '''
    Finding the longest (most heavy atoms) molecule
    in the dataframe.
    '''
    mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
    mol_sizes = [x.GetNumHeavyAtoms() for x in mols]
    longest_molecule = max(mol_sizes)
    return longest_molecule

def main():
    # Run the RF, SVM, ExtraTrees, GradientBoost
    df = pd.read_pickle('../data/processed_data_with_xyz.pickle')

    longest_molecule = find_longest_molecule(df) # It is 207.

    # fold = 0
    # model = RandomForestClassifier()

    model_zoo = {'RF': RandomForestClassifier(),
                 'ExtraTrees': ExtraTreesClassifier(),
                 'GradBoost': GradientBoostingClassifier(),
                 'SVM': SVC()}

    args = init_args()
    fold = args.fold
    model = model_zoo[args.model]
    features = args.features

    CEF = Chem_Feature_Embeddings(features=features,
                                  longest_molecule=longest_molecule)

    cv = KFold(n_splits=5, shuffle=True, random_state=3)
    idxs = np.array(df.index)
    train_idxs, test_idxs = list(cv.split(idxs))[fold]

    train_df = df.loc[train_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)

    train_inputs = CEF.create_inputs(train_df)
    test_inputs = CEF.create_inputs(test_df)

    train_labels = np.array([1 if x == '+' else 0 for x in train_df['Rotation']]).reshape((len(train_df),))
    test_labels = np.array([1 if x == '+' else 0 for x in test_df['Rotation']]).reshape((len(test_df),))

    preds = predictions(train_inputs, train_labels, test_inputs, model)
    f_score, precision, recall, accuracy = scorings(test_labels, preds)
    print('*' * 10)
    print(f'{model} (Fold: {fold})')
    print(f'\t F-Score: {f_score}')
    print(f'\t Precision: {precision}')
    print(f'\t Recall: {recall}')
    print(f'\t Accuracy: {accuracy}')
    print('*' * 10)


if __name__ == '__main__':
    main()




