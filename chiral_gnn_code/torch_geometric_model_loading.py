'''
Setting up various torch geometric models.
'''

import numpy as np
import pandas as pd
import torch
import torch_geometric

from typing import Union
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, NNConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Geometric_Models(torch.nn.Module):
    '''
    Loads a standardized model from a given name.
    '''
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super().__init__()
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.graph_convoluation_layer_dict = {
            'GCN' : GCNConv,
            'GAT' : GATConv,
            'SAGE' : SAGEConv,
            'GIN' : GINConv,
            'NN' : NNConv
        }

    def forward(self, node_info : torch.tensor,
                edge_index : torch.tensor,
                model_name : str) -> torch.tensor:
        conv1 = self.graph_convoluation_layer_dict[model_name](self.input_layer_size,
                                                               self.hidden_layer_size)
        conv2 = self.graph_convoluation_layer_dict[model_name](self.hidden_layer_size,
                                                               self.output_layer_size)
        graph_embedding = conv1(node_info, edge_index)
        graph_embedding = torch.nn.GELU()(graph_embedding)
        output = conv2(graph_embedding, edge_index)
        return torch.nn.SoftMax()(output)

    def calculate_loss(self, predicted : torch.tensor,
                       true : torch.tensor) -> torch.tensor:
        return torch.nn.BCELoss()(predicted, true)

def train_one_epoch(model : torch.nn.Module,
                    dataloader : torch_geometric.loader.DataLoader,
                    gnn_name : str,
                    optimizer : torch.optim.optimizer) -> np.array:
    '''
    Training the generic model (Geometric Models) one epoch.
    '''
    model.train()
    losses = []
    for data in dataloader:
        node_info = data['x'].to(device)
        edge_idxs = data['edge_indices'].to(device)
        true_values = data['y'].to(device)
        optimizer.zero_grad()
        preds = model(node_info, edge_idxs, gnn_name)
        loss = model.calculate_loss(preds, true_values)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy)
    return losses

def validate_test_one_epoch(model : torch.nn.Module,
                            dataloader : torch_geometric.loader.DataLoader,
                            gnn_name : str) -> np.array:
    '''
    Validating or testing on a single epoch.
    '''
    model.evaluate()
    losses = []
    for data in dataloader:
        node_info = data['x'].to(device)
        edge_idxs = data['edge_indices'].to(device)
        true_values = data['y'].to(device)
        preds = model(node_info, edge_idxs, gnn_name)
        loss = model.calculate_loss(preds, true_values)
        losses.append(loss.cpu().detach().numpy())
    return losses


