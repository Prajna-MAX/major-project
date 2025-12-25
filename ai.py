# ai_gnn.py — TRAIN GNN ONLY
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

from model import GNN_ToxModel, mol_to_graph

TARGET_COLS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4


class ToxDataset(Dataset):
    def __init__(self, df):
        self.smiles = df["smiles"].tolist()
        self.labels = df[TARGET_COLS].fillna(0).values.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = mol_to_graph(self.smiles[idx])
        return graph, torch.tensor(self.labels[idx])


def collate_fn(batch):
    graphs = [g[0] for g in batch]
    labels = torch.stack([g[1] for g in batch])
    return Batch.from_data_list(graphs), labels


def auc_score(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for graph, y in loader:
            graph = graph.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(graph)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    aucs = []
    for i in range(len(TARGET_COLS)):
        if len(np.unique(trues[:,i])) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(trues[:,i], preds[:,i]))
    return np.nanmean(aucs)


def main():
    df = pd.read_csv("tox21.csv")
    df = df.dropna(subset=TARGET_COLS, how="all")
    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)

    dataset = ToxDataset(df)
    n_val = int(0.1 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset)-n_val, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = GNN_ToxModel(num_tasks=len(TARGET_COLS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_auc = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader)

        for graph, y in pbar:
            graph = graph.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(graph)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        val_auc = auc_score(model, val_loader)
        print(f"Epoch {epoch} | Val AUC = {val_auc:.4f}")

        torch.save(model.state_dict(), f"gnn_epoch_{epoch}.pt")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "gnn_best.pt")
            print("Saved best model.")

    print("Training complete. Best AUC =", best_auc)


if __name__ == "__main__":
    main()
