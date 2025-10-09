from sklearn.gaussian_process import GaussianProcessClassifier
import os
from dataconversion import build_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, f1_score
from matplotlib import colormaps
X, y, df_clean = build_dataset("./data/processed_data.csv")


def randomforestclassification(X, y, save_dir):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=3,
        stratify=y
    )

    rf.fit(train_inputs, train_labels)
    rf_preds = rf.predict(test_inputs)
    rf_pred_proba = rf.predict_proba(test_inputs)

    df_rf_pred=pd.DataFrame({"true": test_labels, "prediction": rf_preds, "pred_proba": rf_pred_proba[:,1]})
    df_rf_pred.to_csv(Path(save_dir, "Random Forest predictions.csv"))

    print(f"Random forest accuracy: {accuracy_score(test_labels, rf_preds)}")
    print(f"Random forest roc_auc_score:{roc_auc_score(test_labels, rf_pred_proba[:,1])}")
    print(f"Random forest loss:{log_loss(test_labels, rf_pred_proba)}")
    print(f"Random forest f1_score:{f1_score(test_labels, rf_preds)}")


    matrix = confusion_matrix(test_labels, rf_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Reds'))
    plt.title('Random Forest Predictions')
    plt.savefig(os.path.join(save_dir, "Random Forest Classifier Result.png"), dpi=300)

def gaussianclassification(X, y, save_dir):
    gpc=GaussianProcessClassifier(random_state=3)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=3,
        stratify=y
    )
    gpc.fit(train_inputs, train_labels)
    gpc.score(test_inputs, test_labels)



    gpc_preds = gpc.predict(test_inputs)
    gpc_preds_proba = gpc.predict_proba(test_inputs)

    df_gpc_pred= pd.DataFrame({"true": test_labels, "prediction": gpc_preds, "proba": gpc_preds_proba[:,1]})
    df_gpc_pred.to_csv(Path(save_dir, "Gaussian Process predictions.csv"))

    print(f"Gaussian accuracy: {accuracy_score(test_labels, gpc_preds)}")
    print(f"Gaussian roc_auc_score:{roc_auc_score(test_labels, gpc_preds_proba[:,1])}")
    print(f"Gaussian loss:{log_loss(test_labels, gpc_preds_proba)}")
    print(f"Gaussian f1_score:{f1_score(test_labels, gpc_preds)}")

    matrix = confusion_matrix(test_labels, gpc_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Blues'))
    plt.title('Gaussian Process result')
    plt.savefig(os.path.join(save_dir,"Gaussian Process Classifier result.png"), dpi=300)


def main():
    save_dir= 'results/benchmark/'
    os.makedirs(save_dir, exist_ok=True)

    randomforestclassification(X, y, save_dir)
    gaussianclassification(X, y, save_dir)

if __name__ == "__main__":
    main()