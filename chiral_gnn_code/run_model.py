'''
Training the GNN models on the set of features desired.

5-Fold Cross Validation
'''

import os
import numpy as np
import pandas as pd
import argparse
import random
from pathlib import Path
import logging
import torch

from torch_geometric.loader import DataLoader

from .torch_geometric_model_loading import Geometric_Models, train_one_epoch, validate_test_one_epoch
from .geometric_dataset import ChiralGNN_Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default='data/processed_data.csv')
    parser.add_argument('--model-name',
                        type=str,
                        choices=['GCN', 'GAT', 'SAGE', 'GIN', 'NN'],
                        help='Choose one of the available options: GCN, GAT, SAGE, GIN, NN.')
    parser.add_argument('--random_seed',
                        type=int)
    parser.add_argument('--features',
                        nargs='+',
                        choices=['atomic number', 'hybridization', 'chirality type', 'xyz'],
                        help='Choose one or more of the available options: atomic number, hybridization, chirality type, xyz')
    parser.add_argument('--epochs',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--hidden_layer_size',
                        type=int,
                        default=128)
    parser.add_argument('--save-dir',
                        type=str)
    return parser.parse_args()

def main():
    args = init_args()

    data = 'data/processed_data.csv'
    model_name = 'GCN'
    epochs = 100
    lr = 1e-3
    batch_size = 64
    features = ['atomic number', 'hybridization', 'chirality type', 'xyz']
    random_seed = 0
    save_dir = 'results'
    hidden_layer_size = 128

    df = pd.read_csv(args.data)

    np.random.seed(random_seed)
    np.random.shuffle(df)
    idxs = np.array_split(df.index, 5)

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    log_file = os.path.join(save_dir, 'epoch_loss.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Split into training and testing / validation cross val sets.
    for fold in range(len(idxs)):
        test_idxs = idxs[fold]
        train_val_idxs = np.delete(idxs, test_idxs)
        train_idxs = train_val_idxs[0:int(np.floor(len(train_val_idxs) * 0.1))]
        val_idxs = train_val_idxs[int(np.floor(len(train_val_idxs) * 0.1)) : ]
        train_df = df.loc[train_idxs].reset_index(drop=True)
        val_df = df.loc[val_idxs].reset_index(drop=True)
        test_df = df.loc[test_idxs].reset_index(drop=True)

        # Set up dataset & datalaoder.
        train_dataset = ChiralGNN_Dataset(df=train_df,
                                          features=features)
        val_dataset = ChiralGNN_Dataset(df=val_df,
                                          features=features)
        test_dataset = ChiralGNN_Dataset(df=test_df,
                                         features=features)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False)

        # Set up model & optimizer.
        input_layer_size = train_dataset[0]['x'].size()[-1]
        model = Geometric_Models(input_layer_size=input_layer_size,
                                 hidden_layer_size=hidden_layer_size,
                                 output_layer_size=1,
                                 model_name=model_name)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr)

        model.to(device)

        # Train - Val Loop.
        for epoch in range(epochs):
            train_losses = train_one_epoch(model, train_dataloader, optimizer)
            logger.debug('Epoch %d | Mean Train Loss : %.3f', epoch, np.mean(train_losses))
            val_losses = validate_test_one_epoch(model, val_dataloader)
            logger.debug('Epoch %d | Mean Val Loss : %.3f', epoch, np.mean(val_losses))

        # Testing.
        test_losses = validate_test_one_epoch(model, test_dataloader)
        test_df['pred'] = test_losses.tolist()
        logger.debug('*** Fold %d *** Mean Test Loss : %.3f', fold, np.mean(test_losses))

        # Save out model and preds.
        torch.save(model.state_dict(), os.path.join(save_dir, 'fold_' + str(fold), 'model_state_dict'))
        test_df.to_pickle(os.path.join(save_dir, 'fold_' + str(fold), 'pred.pickle'))


if __name__ == '__main__':
    main()