"""Random forest model works fine" Gaussian process currently does not work"""

import argparse


from sklearn.gaussian_process import GaussianProcessClassifier
import os
from sklearn.model_selection import KFold

from sklearn.svm import SVC


from sklearn.preprocessing import StandardScaler
from torch_geometric.nn.aggr import scaler

from dataconversion import build_dataset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, f1_score
from matplotlib import colormaps


parser = argparse.ArgumentParser()
parser.add_argument('--fold',
                    type=int,
                    choices=list(range(5)),
                    help='Choose one of the available options: 0-4')
parser.add_argument('--model-name',
                    type=str,
                    choices=['rf', 'gpc', 'SVM', 'ExtraTrees', 'GradientBoosting'],
                    help='Choose one of the available options: rf, gpc, SVM, ExtraTrees, GradientBoosting')
parser.add_argument('--scaler',
                    type=bool,
                    choices=[True, False],
                    help='Choose to use the scaler or not depending on the model')
#gather the input: model_name, scaler, fold_number
model_name = parser.parse_args().model_name
scaler = parser.parse_args().scaler
fold = parser.parse_args().fold
#set up the save directory
save_dir = 'results/benchmark/'
file_dir = os.path.join(save_dir,model_name,'fold_' + str(fold))
os.makedirs(file_dir, exist_ok=True)
#get the data
X, y = build_dataset("./data/processed_data.csv")
#cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=3)
train_idx, test_idx = list(cv.split(X))[fold]
train_X, test_X = X[train_idx], X[test_idx]
train_y, test_y = y[train_idx], y[test_idx]

#choose the correct model

models = {'rf': RandomForestClassifier(n_estimators=100, random_state=3),
          'gpc': GaussianProcessClassifier(random_state=3),
          'SVM': SVC(random_state=3, probability=True),
          'ExtraTrees': ExtraTreesClassifier(random_state=3),
          'GradientBoosting': GradientBoostingClassifier(random_state=3)}

if model_name in models:
    model = models[model_name]
else:
    print("No model selected")
    model = None



#fit the data to the model
def benchmark(train_X, test_X, train_y, model, scaler):

    if scaler == True:
        train_X = StandardScaler().fit_transform(train_X)
        test_X = StandardScaler().fit_transform(test_X)
    else:
        train_X, test_X = train_X, test_X

    model.fit(train_X, train_y)
    model_pred = model.predict(test_X)
    model_pred_proba = model.predict_proba(test_X)
    print(f'fold={fold}, {accuracy_score(test_y, model_pred)}')

    return model_pred, model_pred_proba

    # if model_name == 'rf':
    # model = RandomForestClassifier(random_state=3)
    # elif model_name == 'gpc':
    # model = GaussianProcessClassifier(random_state=3)
    # elif model_name == 'SVM':
    # model = SVC(random_state=3)
    # elif model_name == 'ExtraTrees':
    # model = ExtraTreesClassifier(random_state=3)
    # elif model_name == 'GradientBoosting':
    # model = GradientBoostingClassifier(random_state=3)
    # else:
    # print("No model selected")

    # idxs_train, idxs_test = train_test_split(idxs, test_size=0.2, random_state=3)
    # test_X, test_y = X[idxs_test], y[idxs_test]

#summarize the result of the model
def benchmark_result(test_y, model_pred, model_pred_proba, model_name, file_dir):
    f1 = f1_score(test_y, model_pred)
    roc_auc = roc_auc_score(test_y, model_pred_proba[:, 1])

    matrix = confusion_matrix(test_y, model_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Blues'))
    plt.title(f'{model_name} result')
    plt.savefig(os.path.join(file_dir,"confusion_matrix.png"), dpi=500)
    performance_result = pd.DataFrame([{"Model": model_name, "Accuracy": accuracy_score(test_y, model_pred),
                                      'ROC AUC': roc_auc,
                                      'F1 Score': f1, }])

    df_result = pd.DataFrame({"true": test_y, "prediction": model_pred, "proba": model_pred_proba[:, 1]})
    performance_result.to_csv(os.path.join(file_dir,"model_performance_result.csv"))
    df_result.to_csv(os.path.join(file_dir, "model_prediction_result.csv"))

    return performance_result, df_result


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

    Random_forest_accuracy = {accuracy_score(test_labels, rf_preds)}
    Random_forest_roc_auc_score = {roc_auc_score(test_labels, rf_pred_proba[:,1])}
    Random_forest_loss={log_loss(test_labels, rf_pred_proba)}
    Random_forest_f1_score= {f1_score(test_labels, rf_preds)}

    matrix = confusion_matrix(test_labels, rf_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Reds'))
    plt.title('Random Forest Predictions')
    plt.savefig(os.path.join(save_dir, "Random Forest Classifier Result.png"), dpi=300)

    df_rf_result = pd.DataFrame([{"Model": "Random Forest",
                                 "Accuracy": Random_forest_accuracy,
                                 "ROC AUC": Random_forest_roc_auc_score,
                                 "Log Loss": Random_forest_loss,
                                 "F1 Score": Random_forest_f1_score }])
#save_to_csv
    df_rf_result.to_csv(os.path.join(save_dir, "Random forest table.csv"), index=False)

    return Random_forest_f1_score, Random_forest_accuracy, Random_forest_loss, Random_forest_roc_auc_score, df_rf_result





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

    Gaussian_accuracy = {accuracy_score(test_labels, gpc_preds)}
    Gaussian_roc_auc_score = {roc_auc_score(test_labels, gpc_preds_proba[:,1])}
    Gaussian_loss = {log_loss(test_labels, gpc_preds_proba)}
    Gaussian_f1_score = {f1_score(test_labels, gpc_preds)}

    matrix = confusion_matrix(test_labels, gpc_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Blues'))
    plt.title('Gaussian Process result')
    plt.savefig(os.path.join(save_dir,"Gaussian Process Classifier result.png"), dpi=300)

    return Gaussian_accuracy , Gaussian_roc_auc_score, Gaussian_loss, Gaussian_f1_score




def main():
    model_pred, model_pred_proba = benchmark(train_X,test_X,train_y, model, scaler)
    benchmark_result(test_y, model_pred, model_pred_proba, model_name, file_dir)


if __name__ == "__main__":
    main()