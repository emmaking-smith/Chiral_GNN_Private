、import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay


def preset():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',
                    nargs='+',
                    choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                    help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    ap.add_argument('--model-name', choices=['GCN', 'GAT', 'SAGE', 'GIN', 'Attentive'])
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

def gather_label(pred_path: Path):

    df_pred = pd.read_pickle(Path(pred_path))
    # using the map function to convert the rotation label to 1/0, follow the rule that the dic provide to convert the series' label
    y_true = df_pred["Rotation"].map({"+": 1, "-": 0}).astype(int)
    df_pred["pred"] = df_pred["pred"].apply(lambda x: x.item())

    # convert the prediction value to int, if the value >= 0.5, it will be true and the output value is set to int so the final output will be 1,
    # otherwise, < 0.5, False, 0, this step is for calculating the F1 score, convert the data type of prediction similar to the label
    y_pred = (df_pred["pred"]>=0.5).astype(int)



    return y_true, y_pred

def read_test_loss(log_path: Path):
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        last_line = lines[-1]
        match = re.search(r"Fold\s+\d+.*?Mean\s+Test\s+Loss\s*:\s*([\d\.]+)", last_line)
        if match:
            test_loss=float(match.group(1))

    return test_loss

def average_loss (folder_path):
    all_test_losses = []


    for fold, fold_dir in enumerate(os.listdir(folder_path)):
        log_path = Path(os.path.join(str(folder_path),"fold_"+str(fold),"epoch_loss.log"))
        fold_loss = read_test_loss(log_path)
        all_test_losses.append(fold_loss)

    average_test_loss = np.mean(all_test_losses)
    return average_test_loss



def classification_metrics_manual(y_true, y_pred):
    tp = tn = fp = fn = 0


    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return precision, recall, accuracy, f1



def get_f1_score(y_true, y_pred):


    precision=precision_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred)
    accuracy=accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, accuracy, f1


def fold_score(folder_path):

    fold_precision_score = []
    fold_recall_score = []
    fold_accuracy_score = []
    fold_f1_score = []
    for fold, fold_dir in enumerate(os.listdir(folder_path)):
        pred_path = Path(os.path.join(folder_path,"fold_"+str(fold),"pred.pickle"))
        y_true, y_pred = gather_label(pred_path)
        # _, _, _, f1 = classification_metrics_manual(y_true, y_pred)
        f1=f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        fold_f1_score.append(f1)
        fold_precision_score.append(precision)
        fold_recall_score.append(recall)
        fold_accuracy_score.append(accuracy)

    average_f1_score = (np.sum(fold_f1_score))/5
    average_precision = (np.sum(fold_precision_score))/5
    average_recall = (np.sum(fold_recall_score))/5
    average_accuracy = (np.sum(fold_accuracy_score))/5
    return average_f1_score, average_precision, average_recall, average_accuracy



def main(folder_path):

    average_test_loss = average_loss(folder_path)
    all_y_true = []
    all_y_pred = []
    for fold, fold_dir in enumerate(os.listdir(folder_path)):
        pred_path = Path(os.path.join(folder_path, "fold_" + str(fold), "pred.pickle"))
        y_true, y_pred = gather_label(pred_path)
        all_y_true.extend(list(y_true))
        all_y_pred.extend(list(y_pred))

    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues,values_format='d')
    # plt.title("Confusion Matrix for the GraphSAGE Model with the Full Feature Set")
    # plt.title("Confusion Matrix for the GIN Model Without Atomic Coordinates")
    # plt.title("Confusion Matrix for the GIN Model Without Chirality Type")
    # plt.title("Confusion Matrix for the GIN Model Without Hybridization")
    plt.title("Confusion Matrix for the GraphSAGE Model Without Atomic Number")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    # #manual calculating
    # # precision, recall, accuracy, f1 = classification_metrics_manual(all_y_true, all_y_pred)
    #
    # #sklearn packages
    # # precision, recall, accuracy, f1 = get_f1_score(all_y_true, all_y_pred)
    # print(f'precision:{precision}, recall: {recall}, accuracy: {accuracy}, f1:{f1}, ')

    # average_f1_score, average_precision, average_recall, average_accuracy = fold_score(folder_path)
    # print(f'average_test_loss:{average_test_loss},'
    #       f'average_f1:{average_f1_score}, average_precision:{average_precision}, average_recall:{average_recall}, average_accuracy:{average_accuracy}')

if __name__ == "__main__":
    #testing
    random_seed=3
    save_dir='results'
    model_name='Sage'
    feats='atomic-number_chirality-type_xyz'
    folder_path = Path(os.path.join(str(save_dir), model_name, feats, str(random_seed)))
    # interative
    # folder_path = preset()
    main(folder_path)
