'''
Baselines for concatenation
of the Morgan + Molformer.
'''

import pandas as pd
import numpy as np
import argparse

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
    # Get both dataframes.
    molformer_df = pd.read_pickle('../data/processed_data_with_xyz_and_molformer_embeddings.pickle')
    morgan_df = pd.read_pickle('../data/processed_data_with_xyz_and_morgan_fingerprints.pickle')

    morgan_molformer = np.concat((
        np.array(morgan_df['Morgan_FP'].tolist()).reshape((len(molformer_df), -1)),
        np.array(molformer_df['Molformer_Embeddings'].tolist()).reshape((len(morgan_df), -1))
    ), axis=1)

    model_zoo = {'RF': RandomForestClassifier(),
                 'ExtraTrees': ExtraTreesClassifier(),
                 'GradBoost': GradientBoostingClassifier(),
                 'SVM': SVC()}

    args = init_args()
    fold = args.fold
    model = model_zoo[args.model]

    cv = KFold(n_splits=5, shuffle=True, random_state=3)
    idxs = np.array(molformer_df.index)
    train_idxs, test_idxs = list(cv.split(idxs))[fold]

    train_df = molformer_df.loc[train_idxs].reset_index(drop=True)
    test_df = molformer_df.loc[test_idxs].reset_index(drop=True)

    del molformer_df, morgan_df

    train_inputs = morgan_molformer[train_idxs]
    train_labels = np.array([1 if x == '+' else 0 for x in train_df['Rotation']]).reshape((len(train_df), ))

    test_inputs = morgan_molformer[test_idxs]
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