# -*- coding: utf-8 -*-

"""
Plot train/val loss curves from epoch_loss.log and prediction distribution from pred_pickle file
- Saves a PNG curve and a CSV table for epoch_loss.log
- Saves a PNG for pred_pickle file.
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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



#Parse pred.pickle file.

def read_pickle(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_pickle(Path(pred_path))
    return pred


def main():
    ap = argparse.ArgumentParser(description="Plot train/val loss from epoch_loss.log")
    ap.add_argument("--log",
                    default="./epoch_loss.log",
                    help="Path to .../fold_x/epoch_loss.log")
    ap.add_argument("--out", default='loss_curve.png', help="Output image filename (default: loss_curve.png)")
    ap.add_argument("--pred", default="pred.pickle", help="Path to .../fold_x/pred.pickle")
    #ap.add_argument("--smooth", type=int, default=1, help="Moving average window (>=1; 1 = no smoothing)")
    args = ap.parse_args()

    log_path = Path(args.log)
    pred_path = Path(args.pred)
    df_pred = read_pickle(pred_path)
    df_train, df_val = parse_log(log_path)


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
    plt.savefig(args.out, dpi=500)
    print(f"Saved figure: {args.out}")

    plt.figure(num='prediction', figsize=(8, 5))
    plt.scatter(df_pred.index, df_pred["pred"], label="prediction", marker="x")
    plt.ylim(0, 1)
    plt.ylabel("prediction")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig('prediction.png', dpi=500)
    print(f"Saved figure: prediction.png")

    # Also save the numeric table for further analysis
    csv_out = Path(args.out).with_suffix(".csv")
    df_pred.to_csv(csv_out, index=True)
    print(f"Saved data: {csv_out}")


if __name__ == "__main__":
    main()
