'''
Pytorch Geometric Dataset
'''

import torch
from torch_geometric.data import Dataset, Data
from smiles_to_geometric_data import Create_Graph

class ChiralGNN_Dataset(Dataset):
    def __init__(self, df, features):
        super().__init__()
        self.df = df
        self.processing = Create_Graph(features=features)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx : int):
        smiles = self.df.loc[idx, 'SMILES']
        rotation = self.df.loc[idx, 'Rotation']
        if rotation == '+':
            rotation = 1
        else:
            rotation = 0
        edge_tuples, node_info, bond_types = self.processing.smiles_to_graph(smiles=smiles, label=rotation)

        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types,
                        y=torch.tensor([rotation]))
        return idx_data