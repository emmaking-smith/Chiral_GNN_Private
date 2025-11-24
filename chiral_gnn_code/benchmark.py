
import argparse

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
import os
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from dataconversion import build_dataset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,log_loss
from matplotlib import colormaps
from pathlib import Path


def settings():
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
    parser.add_argument('--save-dir', type=str, )
    parser.add_argument('--data', type=str, )
    parser.add_argument('--features',
                    nargs='*',
                    choices=['atomic number', 'hybridization', 'chirality type', 'xyz'], default=[],
                    help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    parser.add_argument('--morganfingerprint', type=bool,default=False,choices=[True,False], help='if want to use Morgan fingerprint as input')



    # gather the input: model_name, scaler, fold_number
    model_name = parser.parse_args().model_name
    scaler = parser.parse_args().scaler
    fold = parser.parse_args().fold

    # set up the save directory
    save_dir = parser.parse_args().save_dir
    if parser.parse_args().morganfingerprint == True:
        file_dir = os.path.join(save_dir, model_name,'mpg', 'fold_' + str(fold))
    else:
        feats = parser.parse_args().features.copy()
        feats.sort()
        feats = '_'.join(feats).replace(' ', '-')
        file_dir = os.path.join(save_dir,model_name, feats, 'fold_' + str(fold))

    Path(file_dir).mkdir(exist_ok=True,parents=True)

    # get the data
    X, y = build_dataset(features=parser.parse_args().features, pickle_path=parser.parse_args().data, mfp_input=parser.parse_args().morganfingerprint)
    print(X.shape)
    # cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=3)
    train_idx, test_idx = list(cv.split(X))[fold]
    train_X, test_X = X[train_idx], X[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]


    # choose the correct model

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

    return train_X, train_y, test_X, test_y, model, scaler, fold, file_dir,model_name


#fit the data to the model
def benchmark(train_X, test_X, train_y, model, scaler, fold, test_y):

    if scaler == True:
        train_X = StandardScaler().fit_transform(train_X)
        test_X = StandardScaler().fit_transform(test_X)
    else:
        train_X, test_X = train_X, test_X

    if model == GaussianProcessClassifier(random_state=3):
        pca = PCA(n_components=50)
        train_X = pca.fit_transform(train_X)
        test_X = pca.transform(test_X)
    else:
        train_X,test_X = train_X, test_X

    model.fit(train_X, train_y)
    # print(model.kernel_)
    # print(model.base_estimator_.X_train_.dtype)
    # print(model.base_estimator_.y_train_.dtype)
    model_pred = model.predict(test_X)
    model_pred_proba = model.predict_proba(test_X)
    print(f'fold={fold}, accuracy_score: {accuracy_score(test_y, model_pred)}')

    return model_pred, model_pred_proba

#summarize the result of the model
def benchmark_result(test_y, model_pred, model_pred_proba, model_name, file_dir):
    f1 = f1_score(test_y, model_pred)
    roc_auc = roc_auc_score(test_y, model_pred_proba[:, 1])
    loss = log_loss(test_y, model_pred_proba)

    matrix = confusion_matrix(test_y, model_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=colormaps.get_cmap('Blues'))
    plt.title(f'{model_name} result')
    plt.savefig(os.path.join(file_dir,"confusion_matrix.png"), dpi=500)
    performance_result = [{"Model": model_name, "Accuracy": accuracy_score(test_y, model_pred),
                                      'ROC AUC': roc_auc,
                                      'F1 Score': f1, 'test loss': loss}]
    print(performance_result)


    # performance_result.to_csv(os.path.join(file_dir, "model_performance_result.csv"))
    # #checking the prediction，if applying to the larger dataset could be deactivated
    df_result = pd.DataFrame({"true": test_y, "prediction": model_pred, "proba": model_pred_proba[:, 1]})
    df_result.to_csv(os.path.join(file_dir, "model_prediction_result.csv"))

    return f1



def main():
    train_X, train_y, test_X, test_y, model, scaler, fold, file_dir, model_name= settings()
    model_pred, model_pred_proba = benchmark(train_X,test_X,train_y, model, scaler,fold, test_y)
    benchmark_result(test_y, model_pred, model_pred_proba, model_name, file_dir)


if __name__ == "__main__":
    main()