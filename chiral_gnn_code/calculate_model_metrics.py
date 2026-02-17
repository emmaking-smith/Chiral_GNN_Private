'''
Calculating the F-Score, Precision, Recall, and Accuracy
of each model from its directory path.
'''

import os
import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        help='Path to directory containing pred.pickle file OR pred.npy and true.npy files.')
    return parser.parse_args()

def scorings(true : np.array, predictions : np.array) -> tuple[float, float, float, float]:
    '''
    F-score, precision, recall, accuracy.
    '''
    f_score = f1_score(true, predictions)
    precision = precision_score(true, predictions)
    recall = recall_score(true, predictions)
    accuracy = accuracy_score(true, predictions)
    return f_score, precision, recall, accuracy

def main():
    args = init_args()
    try:
        df = pd.read_pickle(os.path.join(args.path, 'pred.pickle'))
        true = np.array([1 if x == '+' else 0 for x in df['Rotation']])
        preds = df['pred']
    except:
        preds = np.load(os.path.join(args.path, 'pred.npy'))
        true = np.load(os.path.join(args.path, 'true.npy'))
    preds = np.array([1 if x >= 0.5 else 0 for x in preds])

    f_score, precision, recall, accuracy = scorings(true, preds)
    fold = args.path.split('/')[4].split('_')[-1]
    model = args.path.split('/')[2]

    print('*'*10)
    print(f'{model} (Fold: {fold})')
    print(f'\t F-Score: {f_score}')
    print(f'\t Precision: {precision}')
    print(f'\t Recall: {recall}')
    print(f'\t Accuracy: {accuracy}')
    print('*'*10)

if __name__ == '__main__':
    main()