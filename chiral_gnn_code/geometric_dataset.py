'''
Pytorch Geometric Dataset
'''

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData

from smiles_to_geometric_data import Create_Graph, Chiral_MFP_Graph, Molformer_Graph

class Molformer_Dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.processing = Molformer_Graph()

    def len(self) -> int:
        return len(self.df)

    def get(self, idx: int) -> BaseData:
        smiles = self.df.loc[idx, 'SMILES']
        rotation = self.df.loc[idx, 'Rotation']
        if rotation == '+':
            rotation = 1
        else:
            rotation = 0
        edge_tuples, bond_types = self.processing.smiles_to_molformer_graph(smiles=smiles)
        node_info = self.df.loc[idx, 'node_info']
        node_info = torch.tensor(node_info, dtype=torch.float).reshape((-1, 768))
        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types,
                        y=torch.tensor([rotation]))
        return idx_data


class ChiralGNN_Dataset_MorganFP(Dataset):
    def __init__(self, df, radius=2, fpSize=512):
        super().__init__()
        self.df = df
        self.processing = Chiral_MFP_Graph(radius=radius, fpSize=fpSize)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx : int):
        smiles = self.df.loc[idx, 'SMILES']
        rotation = self.df.loc[idx, 'Rotation']
        if rotation == '+':
            rotation = 1
        else:
            rotation = 0
        edge_tuples, node_info, bond_types = self.processing.smiles_to_MFP_graph(smiles=smiles)

        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types,
                        y=torch.tensor([rotation]))
        return idx_data

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
        edge_tuples, node_info, bond_types = self.processing.smiles_to_graph(smiles=smiles,
                                                                             xyz_coordinates=self.df.loc[idx, 'xyz']
                                                                             )

        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types,
                        y=torch.tensor([rotation]))
        return idx_data