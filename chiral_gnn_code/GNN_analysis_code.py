import re
import argparse
from pathlib import Path
import pandas as pd
import os
from sklearn.metrics import f1_score
# import matplotlib.pyplot as plt

def preset():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',
                    nargs='+',
                    choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                    help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    ap.add_argument('--model-name', choices=['GCN', 'GAT', 'SAGE', 'GIN', 'NN'])
    ap.add_argument('--random-seed', type=int)
    ap.add_argument('--fold-number', type=int, nargs='+', default=[0,1,2,3,4])
    ap.add_argument('--save-dir', type=str)


    args = ap.parse_args()
    feats = ap.parse_args().features.copy()
    feats.sort()
    feats = '_'.join(feats).replace(' ', '-')



    model_name = args.model_name
    random_seed = args.random_seed
    folder_path = Path(os.path.join(str(args.save_dir), model_name, feats, str(random_seed)))
    fold_number = args.fold_number

    return folder_path, fold_number

def process_pickle(pred_path: Path):

    df_pred = pd.read_pickle(Path(pred_path))
    y_true = df_pred["Rotation"].map({"+": 1, "-": 0})
    y_pred = (df_pred["pred"] >= 0.5).astype(int)
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

def average (folder_path, fold_number):
    all_test_losses = []
    all_f1_scores = []

    for fold in fold_number:
        log_path = Path(os.path.join(folder_path,"fold_"+str(fold),"epoch_loss.log"))
        pred_path = Path(os.path.join(folder_path,"fold_"+str(fold),"pred.pickle"))
        f1 = process_pickle(pred_path)
        fold_loss = read_test_loss(log_path)
        all_test_losses.append({'fold': fold, 'test_loss': fold_loss})
        all_f1_scores.append({'fold': fold, 'f1_score': f1})

    df_loss = pd.DataFrame(all_test_losses).sort_values(["fold"])
    df_f1_scores = pd.DataFrame(all_f1_scores).sort_values(["fold"])
    df_result = pd.merge(df_loss, df_f1_scores, on="fold")


    average_test_loss = df_result["test_loss"].mean()
    average_f1_scores = df_result["f1_score"].mean()
    df_result.loc["mean"]=["mean",average_test_loss, average_f1_scores]
    df_result.to_csv(os.path.join(folder_path, "analysis.csv"), index=False)

    print(f'{folder_path} Average test loss:', average_test_loss)
    print(f'{folder_path} Average f1 score:', average_f1_scores)


    # plt.figure()
    # plt.scatter(df_result["fold"], df_result["test_loss"], label="test", marker="o")
    # plt.axhline(y=average_test_loss, color="r", linestyle="--", label="average test loss")
    # plt.xlabel("fold")
    # plt.ylabel("test loss")
    # plt.savefig(os.path.join(folder_path,"test_loss.png"))
    # plt.figure()
    # plt.scatter(df_f1_scores["fold"], df_f1_scores["f1_score"], label="f1", marker="x")
    # plt.axhline(y=average_f1_scores, color="r", linestyle="--", label="average f1 score")
    # plt.xlabel("fold")
    # plt.ylabel("f1 score")
    # plt.savefig(os.path.join(folder_path,"f1_score.png"))




def main():
    folder_path, fold_number = preset()
    average(folder_path, fold_number)

if __name__ == "__main__":
    main()
