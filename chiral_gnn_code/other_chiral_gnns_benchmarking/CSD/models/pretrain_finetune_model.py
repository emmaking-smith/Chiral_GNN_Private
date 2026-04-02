'''
The model to pretrain on CSD or finetune on optical
'''


import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union
from .base_mpnn import MPNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Pretrain_Finetune_MPNN(nn.Module):
    def __init__(self, message_size : int,
                 message_passes : int,
                 ranked_unique_atoms : list[int],
                 pretrain_model_path : Union[str, None]):
        super(Pretrain_Finetune_MPNN, self).__init__()

        self.message_size = message_size
        self.message_passes = message_passes
        self.top_5_unique_atoms = ranked_unique_atoms[0:5]
        self.pretrain_model_path = pretrain_model_path

        # Set up the MPNN
        self.MPNN = MPNN(message_size=self.message_size,
                         message_passes=self.message_passes,
                         ranked_unique_atoms=ranked_unique_atoms)

        # If the pretrain model path is None, we are pretraining.
        # Else, we load in the trained model.
        if self.pretrain_model_path is not None:
            mpnn_trained_state_dict = self.gen_states()
            self.MPNN.load_state_dict(mpnn_trained_state_dict)
            for param in self.MPNN.parameters():
                param.requires_grad = False

            # Output layers are finetune layers.
            self.output_layers = nn.Sequential(
            nn.Linear(self.message_size, self.message_size),
            nn.ReLU(),
            nn.Linear(self.message_size, 1),
            nn.Sigmoid()
        )

        else:
            self.output_layers = nn.Sequential(
                nn.Linear(self.message_size, self.message_size),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size),
                nn.ReLU(),
                nn.Linear(self.message_size, N),
                nn.ReLU()
            )

    def gen_states(self):
        '''
        Creates a new state_dict for loading the MPNN params to
        create the pretrained model.
        '''
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.pretrain_model_path, map_location=device)
        for key, value in state_dict.items():
            if 'MPNN' in key:
                new_key = key.split('MPNN.')[1]
                new_state_dict[new_key] = value
        return new_state_dict

    def forward(self, g : torch.tensor,
                h : torch.tensor):
        '''
        Args:
            g: the adjacency matricies packet for each molecule.
                has size num molecules in rxns x longest molecule x longest molecule

            h: the features vector for each molecule in the rxns.
               has size num molecules in rxns x longest molecule x num features

        Returns:
            pred: EITHER the predicted bonds and angles (pretraining) OR
                  the direction of rotation of a molecule (finetuning)
                  with 1 being (+) and 0 being (-).
        '''

        # Run each batch through the MPNN.
        embeddings = self.MPNN(g, h)
        embeddings = torch.sum(embeddings, dim=1) # check this!!

        pred = self.output_layers(embeddings)
        return pred

    def calculate_pretrain_loss(self, predictions, true):
        return nn.MSELoss()(predictions, true)

    def calculate_finetune_loss(self, predictions, true):
        return nn.BCELoss()(predictions, true)

def train_one_epoch(model : torch.nn.Module,
                    dataloader : torch.utils.data.DataLoader,
                    optimizer : torch.optim,
                    pretrain : bool) -> list:
    '''
    Training the CSD model one epoch.
    '''
    model.train()
    losses = []
    for batch in dataloader:
        adj_matrices = batch[0].to(device).float()
        feature_vectors = batch[1].to(device).float()
        true_values = batch[2].to(device).float()
        optimizer.zero_grad()
        preds = model(adj_matrices, feature_vectors)
        if pretrain == True:
            loss = model.calculate_pretrain_loss(preds, true_values)
        else:
            loss = model.calculate_finetune_loss(preds, true_values)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    return losses

def validate_test_one_epoch(model : torch.nn.Module,
                    dataloader : torch.utils.data.DataLoader,
                    pretrain : bool) -> tuple[list, list]:
    '''
    Validating or testing on a single epoch.
    '''
    model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            adj_matrices = batch[0].to(device).float()
            feature_vectors = batch[1].to(device).float()
            true_values = batch[2].to(device).float()
            preds = model(adj_matrices, feature_vectors)
            if pretrain == True:
                loss = model.caluculate_pretrain_loss(preds, true_values)
            else:
                loss = model.calculate_finetune_loss(preds, true_values)
            losses.append(loss.cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
    return losses, predictions