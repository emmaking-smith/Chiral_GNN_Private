'''
Setting up various torch geometric models.

UPDATE 17 SEP: Works on dummy main() code.
'''

import numpy as np
import pandas as pd
import torch
import torch_geometric

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, NNConv, global_mean_pool
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, GIN, AttentiveFP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Geometric_Models_2(torch.nn.Module):
    def __init__(self,
                 input_layer_size : int,
                 hidden_layer_size : int,
                 output_layer_size : int,
                 num_message_passes : int,
                 model_name : str):
        super().__init__()

        self.model_name = model_name

        graph_dict = {
            'GCN' : GCN,
            'GAT' : GAT,
            'SAGE' : GraphSAGE,
            'GIN' : GIN,
            'Attentive' : AttentiveFP
        }

        if model_name == 'Attentive':
            self.gnn = graph_dict[model_name](in_channels=input_layer_size,
                                              hidden_channels=hidden_layer_size,
                                              out_channels=output_layer_size,
                                              num_layers=num_message_passes,
                                              edge_dim=1,
                                              num_timesteps=3)
        else:
            self.gnn = graph_dict[model_name](in_channels=input_layer_size,
                                              hidden_channels=hidden_layer_size,
                                              out_channels=output_layer_size,
                                              num_layers=num_message_passes)

    def forward(self, batch : torch_geometric.data.batch.Batch) -> torch.tensor:
        graph_embeddings = self.gnn(x=batch['x'],
                                    edge_index=batch['edge_index'],
                                    edge_attr=batch['edge_attr'],
                                    batch=batch['batch'])
        if self.model_name != 'Attentive':
            graph_embeddings = global_mean_pool(graph_embeddings, batch['batch'])
        output = torch.nn.Sigmoid()(graph_embeddings).reshape([-1])
        return output

    def calculate_loss(self, predicted : torch.tensor,
                       true : torch.tensor) -> torch.tensor:
        return torch.nn.BCELoss()(predicted, true)

# class Geometric_Models(torch.nn.Module):
#     '''
#     Loads a standardized model from a given name.
#     '''
#     def __init__(self, input_layer_size : int,
#                  hidden_layer_size : int,
#                  output_layer_size : int,
#                  model_name : str):
#         super().__init__()
#         self.input_layer_size = input_layer_size
#         self.hidden_layer_size = hidden_layer_size
#         self.output_layer_size = output_layer_size
#
#         self.graph_convolution_layer_dict = {
#             'GCN' : GCNConv,
#             'GAT' : GATConv,
#             'SAGE' : SAGEConv,
#             'GIN' : GINConv,
#             'NN' : NNConv
#         }
#
#         self.conv_layer1 = self.graph_convolution_layer_dict[model_name](self.input_layer_size,
#                                                                          self.hidden_layer_size)
#         self.conv_layer2 = self.graph_convolution_layer_dict[model_name](self.hidden_layer_size,
#                                                                          self.hidden_layer_size)
#         self.conv_layer3 = self.graph_convolution_layer_dict[model_name](self.hidden_layer_size,
#                                                                          self.hidden_layer_size)
#         self.final_layer = torch.nn.Linear(self.hidden_layer_size, self.output_layer_size)
#
#     def forward(self, batch : torch_geometric.data.batch.Batch) -> torch.tensor:
#         node_info = batch['x']
#         edge_index = batch['edge_index']
#         edge_attr = batch['edge_attr']
#         graph_embeddings = torch.nn.ReLU()(self.conv_layer1(x=node_info,
#                                                             edge_index=edge_index,
#                                                             edge_attr=edge_attr))
#         graph_embeddings = torch.nn.ReLU()(self.conv_layer2(x=graph_embeddings,
#                                                             edge_index=edge_index,
#                                                             edge_attr=edge_attr))
#         graph_embeddings = torch.nn.ReLU()(self.conv_layer3(x=graph_embeddings,
#                                                             edge_index=edge_index,
#                                                             edge_attr=edge_attr))
#         output = self.final_layer(graph_embeddings)
#         # Average pool for each molecule.
#         output = global_mean_pool(output, batch['batch'])
#         return torch.nn.Sigmoid()(output).reshape([-1])
#
#     def calculate_loss(self, predicted : torch.tensor,
#                        true : torch.tensor) -> torch.tensor:
#         return torch.nn.BCELoss()(predicted, true)

def train_one_epoch(model : torch.nn.Module,
                    dataloader : torch_geometric.loader.DataLoader,
                    optimizer : torch.optim) -> list:
    '''
    Training the generic model (Geometric Models) one epoch.
    '''
    model.train()
    losses = []
    for batch in dataloader:
        batch = batch.to(device)
        true_values = batch['y'].to(device).float()
        optimizer.zero_grad()
        preds = model(batch)
        loss = model.calculate_loss(preds, true_values)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    return losses

def validate_test_one_epoch(model : torch.nn.Module,
                            dataloader : torch_geometric.loader.DataLoader,
                            ) -> tuple[list, list]:
    '''
    Validating or testing on a single epoch.
    '''
    model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            true_values = batch['y'].to(device).float()
            preds = model(batch)
            loss = model.calculate_loss(preds, true_values)
            losses.append(loss.cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
    return losses, predictions

def main():
    from geometric_dataset import ChiralGNN_Dataset
    from torch_geometric.loader import DataLoader

    df = pd.read_pickle('data/processed_data_with_xyz.pickle')
    df = df[0:3]
    features = ['atomic number', 'hybridization', 'chirality type', 'xyz']
    model_name = 'Attentive'
    dataset = ChiralGNN_Dataset(df=df, features=features)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    input_layer_size = dataset[0]['x'].size()[-1]
    hidden_layer_size = 3
    output_layer_size = 1
    model = Geometric_Models_2(input_layer_size=input_layer_size,
                             hidden_layer_size=hidden_layer_size,
                             output_layer_size=output_layer_size,
                             num_message_passes=3,
                             model_name=model_name)

    lr = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr)
    model = model.to(device)
    for epoch in range(10):
        print(epoch)
        batch_losses = train_one_epoch(model, dataloader, optimizer)
        print('train loss', np.mean(batch_losses))
        batch_val, _ = validate_test_one_epoch(model, dataloader)
        print('val loss', np.mean(batch_val))
        print()

    # df = pd.read_pickle('data/processed_data_with_xyz.pickle')
    # df = df[0:3]
    # features = ['atomic number', 'hybridization', 'chirality type', 'xyz']
    # model_name = 'NN'
    # dataset = ChiralGNN_Dataset(df=df, features=features)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    # input_layer_size = dataset[0]['x'].size()[-1]
    # hidden_layer_size = 3
    # output_layer_size = 1
    # model = Geometric_Models(input_layer_size=input_layer_size,
    #                          hidden_layer_size=hidden_layer_size,
    #                          output_layer_size=output_layer_size,
    #                          model_name=model_name)
    # lr = 1e-3
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                                        lr=lr)
    # model = model.to(device)
    # epochs = 10
    # for epoch in range(epochs):
    #     print(epoch)
    #     batch_losses = train_one_epoch(model, dataloader, optimizer)
    #     print('train loss', np.mean(batch_losses))
    #     batch_val, _ = validate_test_one_epoch(model, dataloader)
    #     print('val loss', np.mean(batch_val))
    #     print()

if __name__ == '__main__':
    main()