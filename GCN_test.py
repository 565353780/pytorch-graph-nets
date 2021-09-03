#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCN_Trainer:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.device = None
        self.model = None
        self.dataset = None
        self.data_loader = None
        self.optimizer = None
        return

    def loadModel(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = GCN().to(self.device)
        self.model.train()
        return

    def loadDataset(self, dataset, batch_size):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        return

    def train(self, epoch_num):
        data = self.dataset[0].to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(epoch_num):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            print("\rTrain epoch = " + str(epoch+1) + "/" + str(epoch_num), end="")
        print()
        return

    def detect(self, data):
        self.model.eval()
        data = data.to(self.device)
        print(data)
        pred = self.model(data).argmax(dim=1)
        print(pred.shape)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(acc))
        return

class GCN_Detector:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.device = None
        self.model = None
        return

    def loadModel(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        return

    def detect(self, data):
        data = data.to(self.device)
        pred = self.model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(acc))
        return

if __name__ == '__main__':
    dataset = Planetoid(
        root='/home/chli/Cora',
        name='Cora')
    batch_size = 32

    gcn_trainer = GCN_Trainer()

    gcn_trainer.loadModel()
    gcn_trainer.loadDataset(dataset, batch_size)

    gcn_trainer.train(200)

    graph = dataset[0]
    gcn_trainer.detect(graph)

