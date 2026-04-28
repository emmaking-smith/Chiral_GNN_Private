'''
Datasets for the different feature vectors.
'''

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

class CSD_Atom_Feat_Dataset(Dataset):
    def __init__(self,
                 df : pd.DataFrame,
                 max_num_angles : int,
                 max_molecule_size : int,
                 max_bonds : int,
                 pretrain : bool,
                 ):
        '''
        df = dataframe created from csd_node_features_labels module
        max_num_angles = maximum number of angles in the entire dataset (for padding labels)
        max_bonds = maximum number of bonds an atom has in the entire dataset (for padding labels)
        max_molecule_size = longest molecule in dataset (for padding adj matrix)
        bond_types = the list of bond types present in the dataset.
        pretrain = T/F is this pretraining or not (finetuning)?
        '''
        super(CSD_Atom_Feat_Dataset, self).__init__()
        self.df = df
        self.max_num_angles = max_num_angles
        self.max_molecule_size = max_molecule_size
        self.max_bonds = max_bonds
        self.pretrain = pretrain
        self.bond_types = [
                    'Single',
                    'Double',
                    'Triple',
                    'Aromatic',
                    'Pi',
                    'Quadruple',
                    'Delocalised'
                    ]

    def __len__(self):
        return len(self.df)

    def pad(self, bond_or_angles_list : list[float], max_value : int) -> list[float]:
        '''
        Pads out bond lengths / angles for a single molecule to the maximum
        number of bonds / angles.
        '''
        padding_size = max_value - len(bond_or_angles_list)
        bond_or_angles_list = bond_or_angles_list + [0.] * padding_size
        return bond_or_angles_list

    def __getitem__(self, idx : int) -> tuple[torch.tensor, torch.tensor]:
        # Get the features.
        atomic_numbers = np.array(self.df.loc[idx, 'atomic_numbers'])
        chirality = np.array(self.df.loc[idx, 'chirality'])
        hybridization = np.array(self.df.loc[idx, 'hybridization'])
        if self.pretrain == True:
            xyz = np.array(self.df.loc[idx, 'xyz'])
        else:
            xyz = np.array(self.df.loc[idx, 'xyz'][0:self.df.loc[idx, 'num_heavy_atoms']])

        # Create the feature vector.
        node_features = torch.concat((torch.tensor(atomic_numbers).unsqueeze(0).t(),
                                      torch.tensor(chirality).unsqueeze(0).t(),
                                      torch.tensor(hybridization).unsqueeze(0).t(),
                                      torch.tensor(xyz)), dim=1).float()

        # Padding feature vector.
        node_features = torch.concat((
            node_features,
            torch.zeros((self.max_molecule_size - node_features.size()[0], node_features.size()[1]))
        ))

        # Make the adjacency matrices.
        matrices = [self.make_bond_type_adj_matrix(
            self.df.loc[idx, 'adj_indices_dict'], x) for x in self.bond_types]
        matrices += [self.make_universal_bond_matrix(self.df.loc[idx, 'num_heavy_atoms'])]
        matrices = torch.tensor(matrices)

        # Create labels.
        if self.pretrain == True:
            # Create the bond angles vector.
            angles = self.df.loc[idx, 'angles']
            angles = list(
                map(
                    lambda x: self.pad(x, self.max_num_angles),
                    angles
                )
            )
            angles = torch.tensor(angles)
            angles = torch.cat((
                angles,
                torch.zeros((self.max_molecule_size - angles.size()[0], angles.size()[1])),
                                ))

            # Create the bond lengths vector
            bond_lengths = self.df.loc[idx, 'bond_lengths']
            bond_lengths = list(
                map(
                    lambda x: self.pad(x, self.max_bonds),
                    bond_lengths
                )
            )
            bond_lengths = torch.tensor(bond_lengths)
            bond_lengths = torch.cat((
                bond_lengths,
                torch.zeros((self.max_molecule_size - bond_lengths.size()[0], bond_lengths.size()[1]))
            ))

            # Create labels = concatenation of bond lengths and angles.
            labels = torch.cat((bond_lengths, angles), dim=1)

        else:
            rotation = self.df.loc[idx, 'Rotation']
            if rotation == '+':
                rotation = 1
            else:
                rotation = 0
            labels = torch.tensor([rotation]).float()

        return (matrices, node_features), labels

    def make_bond_type_adj_matrix(self, adj_indices_dict : dict, bond_type : str) -> np.array:
        '''
        Creating the adjacency matrix from the atomic index tuples
        that indicate which set of atoms have a specific bond type
        connecting them.
        '''
        matrix = np.zeros((self.max_molecule_size, self.max_molecule_size))
        bond_type_dict = adj_indices_dict[bond_type]
        if len(bond_type_dict) > 0:
            for tuple in bond_type_dict:
                matrix[tuple[0], tuple[1]] = 1
                matrix[tuple[1], tuple[0]] = 1
        return matrix

    def make_universal_bond_matrix(self, mol_size : int) -> np.array:
        '''
        Creating the adjacency matrix from the atomic index tuples
        that indicate which set of atoms have a specific bond type
        connecting them.
        '''
        uni_matrix = np.zeros((self.max_molecule_size, self.max_molecule_size))
        for i in range(mol_size):
            uni_matrix[i, mol_size] = 1
            uni_matrix[mol_size, i] = 1
        return uni_matrix

    def collate_fn(self, batch):
        '''
        Collating batches.
        '''
        batch_size = len(batch)
        (matrices, node_features), labels = batch[0]
        feature_length = node_features.size()[1]

        for i in range(1, batch_size):
            matrices = torch.cat((matrices, batch[i][0][0]), dim=0)
            node_features = torch.cat((node_features, batch[i][0][1]), dim=0)
            labels = torch.cat((labels, batch[i][1]), dim=0)

        # Reshaping.
        matrices = matrices.reshape((
            batch_size, -1, self.max_molecule_size, self.max_molecule_size
        ))
        node_features = node_features.reshape((
            batch_size, self.max_molecule_size, feature_length
        ))
        if self.pretrain == True:
            labels = labels.reshape((
                batch_size, self.max_molecule_size, -1
            ))
        else:
            labels = labels.reshape((batch_size, -1))

        return matrices, node_features, labels










