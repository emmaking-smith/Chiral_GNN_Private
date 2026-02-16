'''
Getting ChiGNN working from Yan et. al.

https://pubs.acs.org/doi/10.1021/acs.jcim.4c02259

Use chignn3.10 conda environment.
'''

# Dataset formation
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import os
import logging

from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from model import EKS_eval, CHIGraphModel, train

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',
                        type=int,
                        help='Which fold of the cross validation should be left out in testing?')
    parser.add_argument('--cv',
                        type=int,
                        help='Number of folds in cross validation.',
                        default=5)
    parser.add_argument('--save-dir',
                        type=str)
    parser.add_argument('--random-seed',
                        type=int,
                        default=0)
    parser.add_argument('--chunk',
                        type=int)
    return parser.parse_args()

def logger_setup(fold : int, save_dir : str) -> logging.Logger:
    '''
    Returns a specific logger for each fold.
    '''
    log_file = os.path.join(save_dir, 'epoch_loss.log')
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

def Construct_dataset(data_index, labels, all_smiles):
    big_index = []
    data_graphs = []
    data_graph_informations = []
    for i in tqdm(range(len(all_smiles))):
        if labels[i] == '+':
            y = torch.Tensor([0.])
        else:
            y = torch.Tensor([1.])
        data_index_int = torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        if y[0] > 60:
            big_index.append(i)
            continue
        smiles = all_smiles[i]

        try:
            mol = smiles_to_3d_mol(smiles)
            data = mol_to_data(mol)

            edge_index = data.edge_index
            data = to_edge_graph(data)
            data.pos = None

        except:
            print(i, smiles)
            continue

        data_graphs.append(data)
        data_graph_information = Data(edge_index=edge_index, y=y, data_index=data_index_int)
        data_graph_informations.append(data_graph_information)
    return data_graphs, data_graph_informations, big_index

def main():
    args = init_args()

    # df = pd.read_csv('data/processed_data.csv')
    # df = df[args.chunk:args.chunk+15000].reset_index(drop=True)
    # all_smiles = df['SMILES'].values
    # labels = df['Rotation'].values
    # index = df.index
    #
    # data_graphs, data_graph_informations, big_index = Construct_dataset(data_index=index,
    #                                                                     labels=labels,
    #                                                                     all_smiles=all_smiles)
    #
    # with open('data/ChiGNN/graph_data_' + str(args.chunk) + '_' + str(args.chunk + 15000) + '.pkl', 'wb') as f:
    #     pickle.dump((data_graphs, data_graph_informations, big_index), f)


    # Running the model on our data.
    with open('data/ChiGNN/EKS_graph_data.pkl', 'rb') as f:
        data_graphs, data_graph_informations, big_index = pickle.load(f)

    total_num = len(data_graphs)

    # Using the default arguments from ChiGNN
    num_layers = 3
    graph_pooling = 'sum'
    emb_dim = 128
    drop_ratio = 0
    save_test = 'store_true'
    batch_size = 2048
    epochs = 1500
    weight_decay = 0.00001
    early_stop = 10
    num_workers = 0
    dataset_root = 'data/ChiGNN'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Using our dataset splits.
    save_dir = os.path.join(args.save_dir, str(args.random_seed), 'fold_' + str(args.fold))

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    np.random.seed(args.random_seed)
    idxs = np.arange(total_num)
    np.random.shuffle(idxs)
    idxs = np.array_split(idxs, args.cv)

    test_idxs = idxs[args.fold]
    train_test_idxs = idxs.copy()
    del train_test_idxs[args.fold]

    train_test_idxs = np.concatenate(train_test_idxs).reshape([-1])
    train_idxs = train_test_idxs[0:len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1))]
    val_idxs = train_test_idxs[len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1)):]

    assert set(train_idxs).intersection(test_idxs) == set()
    assert set(train_idxs).intersection(val_idxs) == set()
    assert set(val_idxs).intersection(test_idxs) == set()

    # Get the logger ready.
    logger = logger_setup(args.fold, save_dir)

    # Yan et al.'s dataset set up.
    train_data_graphs = []
    valid_data_graphs = []
    test_data_graphs = []
    train_data_graph_informations = []
    valid_data_graph_informations = []
    test_data_graph_informations = []

    for i in test_idxs:
        test_data_graphs.append(data_graphs[i])
        test_data_graph_informations.append(data_graph_informations[i])
    for i in val_idxs:
        valid_data_graphs.append(data_graphs[i])
        valid_data_graph_informations.append(data_graph_informations[i])
    for i in train_idxs:
        train_data_graphs.append(data_graphs[i])
        train_data_graph_informations.append(data_graph_informations[i])

    train_loader_data_graphs = DataLoader(train_data_graphs, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
    train_loader_data_graph_informations = DataLoader(train_data_graph_informations, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
    valid_loader_data_graphs = DataLoader(valid_data_graphs, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
    valid_loader_data_graph_informations = DataLoader(valid_data_graph_informations, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
    test_loader_data_graphs = DataLoader(test_data_graphs, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    test_loader_data_graph_informations = DataLoader(test_data_graph_informations, batch_size=batch_size,
                                                     shuffle=False, num_workers=num_workers)

    # Yan et al.'s model set up.
    nn_params = {
        'num_tasks': 1,
        'num_layers': num_layers,
        'emb_dim': emb_dim,
        'drop_ratio': drop_ratio,
        'graph_pooling': graph_pooling,
        'descriptor_dim': 1
    }

    model = CHIGraphModel(**nn_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    criterion_fn = torch.nn.BCELoss()

    # Training
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, device, train_loader_data_graphs, train_loader_data_graph_informations, optimizer,
                           criterion_fn, train_data_graphs, batch_size=batch_size)
        if (epoch + 1) % 100 == 0:
            val_loss, _, _ = EKS_eval(model, device, valid_loader_data_graphs,
                                valid_loader_data_graph_informations,
                                criterion_fn, valid_data_graphs, batch_size)

            logger.debug('Epoch %d | Mean Train Loss : %.3f', epoch, train_loss)
            logger.debug('Epoch %d | Mean Val Loss : %.3f', epoch, val_loss)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_' + str(epoch + 1) + '.pth'))

    # Testing
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_15000.pth'), map=device))
    _, test_predictions, ground_truth = EKS_eval(model, device, test_loader_data_graphs,
                                test_loader_data_graph_informations,
                                criterion_fn, test_data_graphs, 1)

    np.save(os.path.join(save_dir, 'preds.npy'), test_predictions)
    np.save(os.path.join(save_dir, 'true.npy'), ground_truth)


if __name__ == '__main__':
    main()




