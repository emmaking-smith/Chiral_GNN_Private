import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def settings():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        choices=['rf', 'gpc', 'SVM', 'ExtraTrees', 'GradientBoosting'],
                        help='Choose one of the available options: rf, gpc, SVM, ExtraTrees, GradientBoosting')
    parser.add_argument('--save-dir', type=str )
    parser.add_argument('--inputs',
                        nargs='*',
                        choices=['atomic number', 'hybridization', 'chirality type', 'xyz','mpg'],
                        help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz, mpg')

    model_name = parser.parse_args().model_name
    feats = parser.parse_args().inputs.copy()
    feats.sort()
    feats = '_'.join(feats).replace(' ', '-')
    # set up the save directory
    save_dir = Path(parser.parse_args().save_dir, model_name, feats)


    return model_name,feats,save_dir

def average(model_name, feats, save_dir):
    all_test_losses = []
    all_f1_scores = []

    for fold, fold_dir in enumerate(os.listdir(save_dir)):
        file_dir = os.path.join(save_dir, "fold_" + str(fold), "model_prediction_result.csv")
        df = pd.read_csv(file_dir)
        f1 = f1_score(y_true=df["true"], y_pred=df["prediction"])
        test_loss = log_loss(y_true=df["true"], y_pred=df["proba"])
        all_test_losses.append(test_loss)
        all_f1_scores.append(f1)

    print(f"Average {model_name} with {feats} test loss:  {np.mean(all_test_losses)}")
    print(f"Average Average {model_name} with {feats} f1 score: {np.mean(all_f1_scores)}")


def main():
    model_name, save_dir, feats = settings()
    average(model_name, save_dir, feats)

if __name__ == "__main__":
    main()






