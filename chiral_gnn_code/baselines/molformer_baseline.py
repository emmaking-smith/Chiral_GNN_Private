'''
Testing out how Molformer -> classic ML does.

NOTE: NEEDS TRANSFORMERS==4.36.2!!!
'''

import pandas as pd
import numpy as np
import argparse
import torch
import os

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',
                        type=int)
    parser.add_argument('--model',
                        type=str,
                        choices=['RF', 'ExtraTrees', 'GradBoost', 'SVM'])
    return parser.parse_args()

class MolFormer_Embeddings:
    def __init__(self):
        '''
        Turn the SMILES strings into MolFormer pytorch tensors.
        '''

        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                               deterministic_eval=True,
                                               trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                                       trust_remote_code=True)

    def embed_one_smiles(self, smiles : str) -> torch.tensor:
        '''
        Embed one smiles string.
        '''
        inputs = self.tokenizer([smiles], padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output  # tensor size 1 x 768

    def create_molformer_embeddings(self, df : pd.DataFrame) -> pd.DataFrame:
        '''
        Creating adding the molformer tensors (as numpy arrays)
        to the existing dataframe.
        '''
        molformer_embeddings = [self.embed_one_smiles(smiles=x).numpy() for x in df['SMILES']]
        df['Molformer_Embeddings'] = molformer_embeddings
        return df

    def create_new_df(self) -> None:
        df = pd.read_pickle('../data/processed_data_with_xyz.pickle')
        new_df = self.create_molformer_embeddings(df)
        new_df.to_pickle('data/processed_data_with_xyz_and_molformer_embeddings.pickle')

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
    # MolFormer_Embeddings().create_new_df()

    # Run the RF, SVM, ExtraTrees, GradientBoost
    df = pd.read_pickle('data/processed_data_with_xyz_and_molformer_embeddings.pickle')

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

    train_inputs = np.array(train_df['Molformer_Embeddings'].tolist()).reshape((len(train_df), -1))
    train_labels = np.array([1 if x == '+' else 0 for x in train_df['Rotation']]).reshape((len(train_df), ))

    test_inputs = np.array(test_df['Molformer_Embeddings'].tolist()).reshape((len(test_df), -1))
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




