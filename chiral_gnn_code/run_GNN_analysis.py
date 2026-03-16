from pathlib import Path
import os

import pandas as pd

from GNN_analysis_code import fold_score, average_loss

model_list=['GCN', 'GIN','GAT', 'Sage','Attentive']
feature_set_list = [
    'atomic-number_hybridization',
    'atomic-number_chirality-type',
    'atomic-number_xyz',
    'chirality-type_hybridization',
    'hybridization_xyz',
    'chirality-type_xyz',
    'atomic-number_chirality-type_hybridization',
    'atomic-number_hybridization_xyz',
    'atomic-number_chirality-type_xyz',
    'chirality-type_hybridization_xyz',
    'atomic-number_chirality-type_hybridization_xyz'
]
save_dir='results_rerun'
random_seed=3



def main(model_list,feature_set_list,save_dir, random_seed):
    rows = []

    for model_name in model_list:
        for feats in feature_set_list:
            folder_path = Path(os.path.join(save_dir, model_name, feats, str(random_seed)))

            loss = average_loss(folder_path)
            # precision, recall, accuracy, f1 = get_f1_score(folder_path)
            f1, precision, recall, accuracy = fold_score(folder_path)

            rows.append({
                'model': model_name,
                'features': feats,
                'loss': loss,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1': f1
            })

    df = pd.DataFrame(rows)

    return df



if __name__ == '__main__':
    df=main(model_list,feature_set_list,save_dir,random_seed)
    df.to_csv('summary_results_fold_rerun.csv', index=False)

