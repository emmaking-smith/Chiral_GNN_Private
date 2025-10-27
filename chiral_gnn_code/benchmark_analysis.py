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
    parser.add_argument('--save-dir', type=str, )

    model_name = parser.parse_args().model_name
    # set up the save directory
    save_dir = Path(parser.parse_args().save_dir, model_name)


    return model_name, save_dir

def average(model_name, save_dir):
    all_test_losses = []
    all_f1_scores = []

    for fold, fold_dir in enumerate(os.listdir(save_dir)):
        file_dir = os.path.join(save_dir, "fold_" + str(fold), "model_prediction_result.csv")
        df = pd.read_csv(file_dir)
        f1 = f1_score(y_true=df["true"], y_pred=df["prediction"])
        test_loss = log_loss(y_true=df["true"], y_pred=df["proba"])
        all_test_losses.append(test_loss)
        all_f1_scores.append(f1)

    print(f"Average test loss: {np.mean(all_test_losses)}")
    print(f"Average f1 score: {np.mean(all_f1_scores)}")


def main():
    model_name, save_dir = settings()
    average(model_name, save_dir)

if __name__ == "__main__":
    main()






