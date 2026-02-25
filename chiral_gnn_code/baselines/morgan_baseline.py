'''
Testing out how chiral Morgan fingerprints -> classic ML does.
pfSize = 512 - same size as NN
'''

import pandas as pd
import numpy as np
import argparse

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',
                        type=int)
    parser.add_argument('--model',
                        type=str,
                        choices=['RF', 'ExtraTrees', 'GradBoost', 'SVM'])
    return parser.parse_args()

class Morgan_Embeddings:
    def __init__(self, radius=2, fpSize=512):
        '''
        Turn the SMILES strings into chiral Morgan fingerprint.
        '''

        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                                      fpSize=fpSize,
                                                                      includeChirality=True)

    def embed_one_smiles(self, smiles : str) -> np.array:
        '''
        Embed one smiles string.
        '''
        mol = Chem.MolFromSmiles(smiles)
        morgan_fp = self.morgan_generator.GetFingerprint(mol)
        return np.array(morgan_fp)

    def create_morgan_embeddings(self, df : pd.DataFrame) -> pd.DataFrame:
        '''
        Creating adding the molformer tensors (as numpy arrays)
        to the existing dataframe.
        '''
        morgan_fingerprints = [self.embed_one_smiles(smiles=x) for x in df['SMILES']]
        df['Morgan_FP'] = morgan_fingerprints
        return df

    def create_new_df(self) -> None:
        df = pd.read_pickle('data/processed_data_with_xyz.pickle')
        new_df = self.create_morgan_embeddings(df)
        new_df.to_pickle('data/processed_data_with_xyz_and_morgan_2048_fingerprints.pickle')

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

def main():
    Morgan_Embeddings().create_new_df()

    # Run the RF, SVM, ExtraTrees, GradientBoost
    df = pd.read_pickle('data/processed_data_with_xyz_and_morgan_2048_fingerprints.pickle')

    # fold = 0
    # model = RandomForestClassifier()

    model_zoo = {'RF': RandomForestClassifier(),
                 'ExtraTrees': ExtraTreesClassifier(),
                 'GradBoost': GradientBoostingClassifier(),
                 'SVM': SVC()}

    args = init_args()
    fold = args.fold
    model = model_zoo[args.model]

    cv = KFold(n_splits=5, shuffle=True, random_state=3)
    idxs = np.array(df.index)
    train_idxs, test_idxs = list(cv.split(idxs))[fold]

    train_df = df.loc[train_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)

    train_inputs = np.array(train_df['Morgan_FP'].tolist()).reshape((len(train_df), -1))
    train_labels = np.array([1 if x == '+' else 0 for x in train_df['Rotation']]).reshape((len(train_df), ))

    test_inputs = np.array(test_df['Morgan_FP'].tolist()).reshape((len(test_df), -1))
    test_labels = np.array([1 if x == '+' else 0 for x in test_df['Rotation']]).reshape((len(test_df), ))

    preds = predictions(train_inputs, train_labels, test_inputs, model)
    f_score, precision, recall, accuracy = scorings(test_labels, preds)
    print('*'*10)
    print(f'{model} (Fold: {fold})')
    print(f'\t F-Score: {f_score}')
    print(f'\t Precision: {precision}')
    print(f'\t Recall: {recall}')
    print(f'\t Accuracy: {accuracy}')
    print('*'*10)

if __name__ == '__main__':
    main()