'''
Training the GNN models on the set of features desired.

5-Fold Cross Validation
'''

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging
import torch

from torch_geometric.loader import DataLoader

from torch_geometric_model_loading import Geometric_Models, train_one_epoch, validate_test_one_epoch
from geometric_dataset import ChiralGNN_Dataset

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
    parser.add_argument('--cv-fold',
                        type=int,
                        help='Which fold of the cross validation should be left out in testing?')
    parser.add_argument('--cv',
                        type=int,
                        help='Number of folds in cross validation.',
                        default=5)
    parser.add_argument('--save-dir',
                        type=str)
    return parser.parse_args()

def logger_setup(fold : int, save_dir : str) -> logging.Logger:
    '''
    Returns a specific logger for each fold.
    '''
    log_file = os.path.join(save_dir, 'fold_' + str(fold), 'epoch_loss.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger(f'fold_{fold}')
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()  # Clear existing handlers

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def main():
    args = init_args()

    # data = 'data/processed_data.csv'
    # model_name = 'GCN'
    # epochs = 10
    # lr = 1e-3
    # batch_size = 64
    # features = ['atomic number', 'hybridization', 'chirality type', 'xyz']
    # random_seed = 0
    # save_dir = 'results'
    # hidden_layer_size = 128
    # cross_val_folds = 5

    df = pd.read_csv(args.data)
    np.random.seed(args.random_seed)
    idxs = np.array(df.index)
    np.random.shuffle(idxs)
    idxs = np.array_split(idxs, args.cv)

    # Make the directory for this fold.
    Path(os.path.join(args.save_dir, 'fold_' + str(args.fold))).mkdir(exist_ok=True, parents=True)

    # Get the logger ready.
    logger = logger_setup(args.fold, args.save_dir)

    # Splitting into training, validation, and testing.
    test_idxs = idxs[args.fold]
    train_test_idxs = idxs.copy()
    del train_test_idxs[args.fold]

    # Every other idx not in the test fold becomes part of the
    # training or validation fold.
    train_test_idxs = np.concatenate(train_test_idxs).reshape([-1])
    train_idxs = train_test_idxs[0:len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1))]
    val_idxs = train_test_idxs[len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1)) : ]

    assert set(train_idxs).intersection(test_idxs) == set()
    assert set(train_idxs).intersection(val_idxs) == set()
    assert set(val_idxs).intersection(test_idxs) == set()

    train_df = df.loc[train_idxs].reset_index(drop=True)
    val_df = df.loc[val_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)

    # Set up dataset & datalaoder.
    train_dataset = ChiralGNN_Dataset(df=train_df,
                                      features=args.features)
    val_dataset = ChiralGNN_Dataset(df=val_df,
                                      features=args.features)
    test_dataset = ChiralGNN_Dataset(df=test_df,
                                     features=args.features)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False)

    # Set up model & optimizer.
    input_layer_size = train_dataset[0]['x'].size()[-1]
    model = Geometric_Models(input_layer_size=input_layer_size,
                             hidden_layer_size=args.hidden_layer_size,
                             output_layer_size=1,
                             model_name=args.model_name)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr)

    model.to(device)

    # Train - Val Loop.
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, train_dataloader, optimizer)
        logger.debug('Epoch %d | Mean Train Loss : %.3f', epoch, np.mean(train_losses))
        val_losses = validate_test_one_epoch(model, val_dataloader)
        logger.debug('Epoch %d | Mean Val Loss : %.3f', epoch, np.mean(val_losses))

    # Testing.
    test_losses = validate_test_one_epoch(model, test_dataloader)
    test_df['pred'] = test_losses
    logger.debug('*** Fold %d *** Mean Test Loss : %.3f', fold, np.mean(test_losses))

    # Save out model and preds.
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(args.fold), 'model_state_dict'))
    test_df.to_pickle(os.path.join(args.save_dir, 'fold_' + str(args.fold), 'pred.pickle'))

if __name__ == '__main__':
    main()