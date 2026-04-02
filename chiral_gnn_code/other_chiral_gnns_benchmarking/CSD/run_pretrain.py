'''
Run the pretraining, save out the model weights, longest_molecule, and ranked atom list.
'''

import pandas as pd
import numpy as np
import os
import argparse
import torch
import logging

from typing import Union
from torch.utils.data import DataLoader
from pathlib import Path

from csd_datasets import CSD_Atom_Feat_Dataset
from models.pretrain_finetune_model import Pretrain_Finetune_MPNN, train_one_epoch, validate_test_one_epoch

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        type=str,
                        help='Directory name to save out files.')
    parser.add_argument('--train_df',
                        type=str,
                        help='Path to train dataframe pickle file.')
    parser.add_argument('--val_df',
                        type=str,
                        help='Path to val dataframe pickle file.')
    parser.add_argument('--test_df',
                        type=str,
                        help='Path to test dataframe pickle file.')
    parser.add_argument('--finetune_df',
                        type=str,
                        help='Path to finetuning dataframe pickle file.')
    parser.add_argument('--fold',
                        type=int)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--epochs',
                        type=int,
                        default=100)
    parser.add_argument('--message_passes',
                        type=int,
                        default=3)
    parser.add_argument('--message_size',
                        type=int,
                        default=128)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--ranked_unique_atoms',
                        type=str,
                        default='data/ranked_unique_atoms.npy',
                        help='Path to the npy file which contains all atoms in the dataset ordered in decreasing frequency.')
    parser.add_argument('--pretrain_model_path',
                        type=str,
                        help='Path to the model dict file for finetuning OR None if pretraining.')
    parser.add_argument('--max_num_angles',
                        type=int,
                        default=45,
                        help='Maximum number of angles possible for atom-wise labels.')
    parser.add_argument('--max_bonds',
                        type=int,
                        default=10,
                        help='Maximum number of bonds possible for each atom.')
    parser.add_argument('--max_molecule_size',
                        type=int,
                        default=101,
                        help='Maximum number atoms in any given molecule.')
    return parser.parse_args()

def logger_setup(fold : Union[int, None], save_dir : str, pretrain : bool) -> logging.Logger:
    '''
    Returns a specific logger for each fold.
    '''
    save_name = 'pretrain_' if pretrain == True else 'finetune_'
    log_file = os.path.join(save_dir, save_name + 'epoch_loss.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    if pretrain == True:
        logger = logging.getLogger(f'fold_{fold}')
    else:
        logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear existing handlers

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def main():
    args = init_args()

    # Environment setup.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain = True if args.pretrain_model_path is None else False

    # Load in dataframes.
    if pretrain == True:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        train_df = pd.read_pickle(args.train_df)
        val_df = pd.read_pickle(args.val_df)
        test_df = pd.read_pickle(args.test_df)
        logger = logger_setup(None, args.save_dir, pretrain)
    else:
        args.save_dir = str(os.path.join(args.save_dir, str(args.random_seed), 'fold_' + str(args.fold)))
        Path(args.save_dir).mkdir(exist_ok=True, parents=True)
        finetune_df = pd.read_pickle(args.finetune_df)
        np.random.seed(0)
        idxs = np.array(finetune_df.index)
        np.random.shuffle(idxs)
        idxs = np.array_split(idxs, 5)
        logger = logger_setup(args.fold, args.save_dir, pretrain)

        # Splitting into training, validation, and testing.
        test_idxs = idxs[args.fold]
        train_test_idxs = idxs.copy()
        del train_test_idxs[args.fold]

        # Every other idx not in the test fold becomes part of the
        # training or validation fold.
        train_test_idxs = np.concatenate(train_test_idxs).reshape([-1])
        train_idxs = train_test_idxs[0:len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1))]
        val_idxs = train_test_idxs[len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1)):]

        assert set(train_idxs).intersection(test_idxs) == set()
        assert set(train_idxs).intersection(val_idxs) == set()
        assert set(val_idxs).intersection(test_idxs) == set()

        train_df = finetune_df.loc[train_idxs].reset_index(drop=True)
        val_df = finetune_df.loc[val_idxs].reset_index(drop=True)
        test_df = finetune_df.loc[test_idxs].reset_index(drop=True)

        del finetune_df

    # Create datasets and dataloaders.
    train_dataset = CSD_Atom_Feat_Dataset(df=train_df,
                 max_num_angles=args.max_num_angles,
                 max_molecule_size=args.max_molecule_size,
                 max_bonds=args.max_bonds)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    val_dataset = CSD_Atom_Feat_Dataset(df=val_df,
                                          max_num_angles=args.max_num_angles,
                                          max_molecule_size=args.max_molecule_size,
                                          max_bonds=args.max_bonds)
    val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    test_dataset = CSD_Atom_Feat_Dataset(df=test_df,
                                        max_num_angles=args.max_num_angles,
                                        max_molecule_size=args.max_molecule_size,
                                        max_bonds=args.max_bonds)
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=train_dataset.collate_fn)

    # Create model and optimizer.
    model = Pretrain_Finetune_MPNN(message_size=args.message_size,
                                   message_passes=args.message_passes,
                                   ranked_unique_atoms=np.load(args.ranked_unique_atoms).tolist(),
                                   pretrain_model_path=args.pretrain_model_path
                                   )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    model.to(device)

    # Training loop.
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, train_loader, optimizer, pretrain)
        logger.debug('Epoch %d | Mean Train Loss : %.3f', epoch, np.mean(train_losses))
        val_losses, _ = validate_test_one_epoch(model, val_loader, pretrain)
        logger.debug('Epoch %d | Mean Val Loss : %.3f', epoch, np.mean(val_losses))

    # Testing.
    test_losses, test_preds = validate_test_one_epoch(model, test_loader, pretrain)
    test_df['pred'] = test_preds
    logger.debug('*** Mean Test Loss *** : %.3f',  np.mean(test_losses))

    # Save out model and preds.
    save_name = 'pretrain_' if pretrain == True else 'finetune_'
    torch.save(model.state_dict(), os.path.join(args.save_dir, save_name + 'model_state_dict'))

    # To save on space, just save out the true.npy and pred.npy
    if pretrain == False:
        ground_truth = [1 if x == '+' else 0 for x in test_df['Rotation']]
        predictions = [float(x[0]) for x in test_df['pred']]
        np.save(os.path.join(args.save_dir, 'true.npy'), ground_truth)
        np.save(os.path.join(args.save_dir, 'pred.npy'), predictions)

if __name__ == '__main__':
    main()