import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score


def preset():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',
                    nargs='+',
                    choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                    help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    ap.add_argument('--model-name', choices=['GCN', 'GAT', 'SAGE', 'GIN', 'NN'])
    ap.add_argument('--random-seed', type=int)
    ap.add_argument('--save-dir', type=str)


    args = ap.parse_args()
    feats = ap.parse_args().features.copy()
    feats.sort()
    feats = '_'.join(feats).replace(' ', '-')



    model_name = args.model_name
    random_seed = args.random_seed
    folder_path = Path(os.path.join(str(args.save_dir), model_name, feats, str(random_seed)))

    return folder_path

def process_pickle(pred_path: Path):

    df_pred = pd.read_pickle(Path(pred_path))
    # using the map function to convert the rotation label to 1/0, follow the rule that the dic provide to convert the series' label
    y_true = df_pred["Rotation"].map({"+": 1, "-": 0})
    # convert the prediction value to int, if the value >= 0.5, it will be true and the output value is set to int so the final output will be 1,
    # otherwise, < 0.5, False, 0, this step is for calculating the F1 score, convert the data type of prediction similar to the label
    y_pred = (df_pred["pred"] >= 0.5).astype(int)
    # print(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return f1

def read_test_loss(log_path: Path):
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        last_line = lines[-1]
        match = re.search(r"Fold\s+\d+.*?Mean\s+Test\s+Loss\s*:\s*([\d\.]+)", last_line)
        if match:
            test_loss=float(match.group(1))

    return test_loss

def average (folder_path):
    all_test_losses = []
    all_f1_scores = []

    for fold, fold_dir in enumerate(os.listdir(folder_path)):
        log_path = Path(os.path.join(str(folder_path),"fold_"+str(fold),"epoch_loss.log"))
        pred_path = Path(os.path.join(folder_path,"fold_"+str(fold),"pred.pickle"))
        f1 = process_pickle(pred_path)
        fold_loss = read_test_loss(log_path)
        all_test_losses.append(fold_loss)
        all_f1_scores.append(f1)



    print(f'{folder_path}, Average test loss:', np.mean(all_test_losses))
    print(f'{folder_path}, Average f1 score:', np.mean(all_f1_scores))





def main():
    folder_path= preset()
    average(folder_path)

if __name__ == "__main__":
    main()
