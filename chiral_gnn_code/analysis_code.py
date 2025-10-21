# -*- coding: utf-8 -*-

"""
Plot train/val loss curves from epoch_loss.log and prediction distribution from pred_pickle file
- Saves a PNG curve and a CSV table for epoch_loss.log
- Saves a PNG and csv table for pred_pickle file.
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os


#Parse epoch and loss numbers from the log file into two tidy DataFrames.

def parse_log(log_path: Path):

    data_train = []
    data_val = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_train = re.search(r".*Epoch\s+(\d+)\s*\|\s*Mean\s+Train\s+Loss\s*:\s*([\d\.]+)", line)
            if m_train:
                data_train.append((int(m_train.group(1)), "train", float(m_train.group(2))))
                continue
            m_val = re.search(r".*Epoch\s+(\d+)\s*\|\s*Mean\s+Val\s+Loss\s*:\s*([\d\.]+)", line)
            if m_val:
                data_val.append((int(m_val.group(1)), "val", float(m_val.group(2))))

    df_train = pd.DataFrame(data_train, columns=["epoch", "train", "loss"]).sort_values(["epoch", "train"])
    df_val = pd.DataFrame(data_val, columns=["epoch", "val", "loss"]).sort_values(["epoch", "val"])
    return df_train, df_val

def read_pickle(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(Path(pred_path))
    return pred

def read_test_loss(log_path: Path):
    test_loss=[]
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        last_line = lines[-1]
        match = re.search(r"Fold\s+(\d+).*?Mean\s+Test\s+Loss\s*:\s*([\d\.]+)", last_line)
        if match:
            test_loss.append(("fold",int(match.group(1)), "loss",float(match.group(2))))

    return test_loss


#test loss average
def average_test_loss (model_name, feats, random_seed):
    all_test_losses = []

    for i in range(5):
        log_path = Path(os.path.join("Result",model_name, feats, str(random_seed),"fold_"+str(i),"/epoch_loss.log"))
        fold_loss = read_test_loss(log_path)
        all_test_losses.append(fold_loss)

    df_test = pd.DataFrame(all_test_losses, columns=["fold", "test_loss"]).sort_values(["fold"])
    average_test_loss = df_test["test_loss"].mean()
    plt.scatter(df_test["fold"], df_test["test_loss"], label="test", marker="o")
    plt.axhline(y=average_test_loss, color="r", linestyle="--", label="average test loss")
    plt.xlabel("fold")
    plt.ylabel("test loss")
    plt.savefig("test_loss.png")
    return  df_test






def preset():
    ap = argparse.ArgumentParser(description="Plot train/val loss from epoch_loss.log")
    ap.add_argument("--log",
                    default="./epoch_loss.log",
                    help="Path to .../fold_x/epoch_loss.log")
    ap.add_argument("--out", default='prediction.png', help="Output image filename (default: prediction.png)")
    ap.add_argument("--pred", default="pred.pickle", help="Path to .../fold_x/pred.pickle")
    ap.add_argument('--features',
                    nargs='+',
                    choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                    help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    ap.add_argument('--model_name', choices=['GCN', 'GAT', 'SAGE', 'GIN', 'NN'])
    ap.add_argument('--random_seed', type=int)
    # ap.add_argument("--smooth", type=int, default=1, help="Moving average window (>=1; 1 = no smoothing)")

    args = ap.parse_args()
    feats = ap.parse_args().features.copy()
    feats.sort()
    feats = '_'.join(feats).replace(' ', '-')

    log_path = Path(args.log)
    pred_path = Path(args.pred)
    df_pred = read_pickle(pred_path)
    df_train, df_val = parse_log(log_path)
    csv_out = Path(args.out).with_suffix(".csv")
    model_name = args.model_name
    random_seed = args.random_seed

    return df_pred, df_train, df_val, args, feats, log_path, csv_out, model_name, random_seed


def plot_and_result(df_pred, df_train, df_val, log_path, csv_out, df_test):


    # Plot (matplotlib only, single figure, no explicit colors)
    plt.figure(num='loss.png', figsize=(8, 5))
    if "loss" in df_train.columns:
        plt.plot(df_train["epoch"], df_train["loss"], label="Train Loss", linewidth=2)
    if "loss" in df_val.columns:
        plt.plot(df_val['epoch'], df_val["loss"], label="Val Loss", linewidth=2)

    print(df_train)
    print(df_val)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({log_path.parent.name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_log.png', dpi=500)
    print(f"Saved figure:loss_log.png")

    plt.figure(num='prediction', figsize=(8, 5))
    plt.scatter(df_pred.index, df_pred["pred"], label="prediction", marker="x")
    plt.ylim(0, 1)
    plt.ylabel("prediction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('prediction.png', dpi=500)
    print(f"Saved figure: prediction.png")

    # Also save the numeric table for further analysis

    df_pred.to_csv(csv_out, index=False)
    df_test.to_csv(csv_out, index=False)


    df_train.to_csv("train_log_data.csv", index=True)
    df_val.to_csv('validation_log_data.csv', index=True)
    print(f"Saved data: train_log_data.csv, validation_log_data.csv")

def main():
    df_pred, df_train, df_val, args, feats, log_path, csv_out, model_name, random_seed = preset()
    df_test = average_test_loss(model_name, feats, random_seed)
    plot_and_result(df_pred, df_train, df_val, log_path, csv_out, df_test)



if __name__ == "__main__":
    main()
