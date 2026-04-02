'''
Datasets for the different feature vectors.
'''

import pandas as pd
from torch.utils.data import Dataset
import torch

class CSD_Atom_Feat_Dataset(Dataset):
    def __init__(self,
                 df : pd.DataFrame,
                 max_num_angles : int,
                 max_molecule_size : int,
                 max_bonds : int,
                 ):
        '''
        df = dataframe created from csd_node_features_labels module
        max_num_angles = maximum number of angles in the entire dataset (for padding labels)
        max_bonds = maximum number of bonds an atom has in the entire dataset (for padding labels)
        max_molecule_size = longest molecule in dataset (for padding adj matrix)
        bond_types = the list of bond types present in the dataset.
        '''
        super(CSD_Atom_Feat_Dataset, self).__init__()
        self.df = df
        self.max_num_angles = max_num_angles
        self.max_molecule_size = max_molecule_size
        self.max_bonds = max_bonds

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
        atomic_numbers = self.df.loc[idx, 'atomic_numbers']
        chirality = self.df.loc[idx, 'chiralty']
        hybridization = self.df.loc[idx, 'hybridization']
        xyz = self.df.loc[idx, 'xyz']

        # Create the feature vector.
        node_features = torch.concat((torch.tensor(atomic_numbers).unsqueeze(0).t(),
                                      torch.tensor(chirality).unsqueeze(0).t(),
                                      torch.tensor(hybridization).unsqueeze(0).t(),
                                      torch.tensor(xyz)), dim=1).float()

        # Padding feature vector.
        node_features = torch.concat((
            torch.zeros((self.max_molecule_size - node_features.size()[0], node_features.size()[1])),
            node_features
        ))

        # Get the adjacency matrices.
        matrices = self.df.loc[idx, 'adj_matrices']

        # Create the bond angles vector.
        angles = self.df.loc[idx, 'angles']
        angles = list(
            map(
                lambda x:self.pad(x, self.max_num_angles),
                angles
            )
        )
        angles = torch.tensor(angles)
        angles = torch.cat((
            torch.zeros((self.max_molecule_size - angles.size()[0], angles.size()[1])),
                            angles
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
            torch.zeros((self.max_molecule_size - bond_lengths.size()[0], bond_lengths.size()[1])),
            angles
        ))

        # Create labels = concatenation of bond lengths and angles.
        labels = torch.cat((bond_lengths, angles), dim=1)

        return (matrices, node_features), labels

    def collate_fn(self, batch):
        '''
        Collating batches.
        '''
        batch_size = len(batch)
        (matrices, node_features), labels = batch[0]
        longest_molecule = matrices.size()[0]
        feature_length = node_features.size()[1]

        for i in range(1, batch_size):
            matrices = torch.cat((matrices, batch[i][0][0]), dim=0)
            node_features = torch.cat((node_features, batch[i][0][1]), dim=0)
            labels = torch.cat((labels, batch[i][1]), dim=0)

        # Reshaping.
        matrices = matrices.reshape((
            batch_size, -1, longest_molecule, longest_molecule
        ))
        node_features = node_features.reshape((
            batch_size, longest_molecule, feature_length
        ))
        labels = labels.reshape((
            batch_size, longest_molecule, -1
        ))

        return matrices, node_features, labels










