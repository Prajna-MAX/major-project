# gnn_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data as GeomData

ATOM_TYPES = ["C","N","O","S","F","Cl","Br","I","P"]


def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    # ---- SAFETY: handle invalid SMILES with dummy graph ---- #
    if mol is None:
        x = torch.zeros((1, 12), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
        return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # ---- Node features ---- #
    node_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        onehot = [1.0 if symbol == a else 0.0 for a in ATOM_TYPES]
        degree = atom.GetDegree()
        charge = atom.GetFormalCharge()
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        node_feats.append(onehot + [degree, charge, aromatic])

    # ---- Edge features ---- #
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bt = bond.GetBondType()
        if bt == rdchem.BondType.SINGLE:
            attr = [1,0,0,0]
        elif bt == rdchem.BondType.DOUBLE:
            attr = [0,1,0,0]
        elif bt == rdchem.BondType.TRIPLE:
            attr = [0,0,1,0]
        elif bond.GetIsAromatic():
            attr = [0,0,0,1]
        else:
            attr = [0,0,0,0]

        # undirected edges
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr += [attr, attr]

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return GeomData(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GNN_ToxModel(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=128, num_tasks=12, layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_tasks)
        )

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)

        graph_embedding = global_add_pool(x, graph.batch)
        return self.readout(graph_embedding)
