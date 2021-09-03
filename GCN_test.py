#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class GCN(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        self.optimizer = None
        return

    def loadModel(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN().to(device)
        self.model.train()
        return

    def loadDataset(self, dataset):
        self.dataset = dataset
        return

    def train(self):
        data = self.dataset[0].to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(200):
            self.optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            print("\rTrain epoch = " + str(epoch) + "/" + str(200), end="")
        print()
        return

    def detect(self, data):
        self.model.eval()
        data = data.to(device)
        pred = self.model(data).argmax(dim=1)
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
        self.model.eval()
        pred = self.model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(acc))
        return

if __name__ == '__main__':
    dataset = Planetoid(root='/home/chli/Cora', name='Cora')

    gcn_trainer = GCN_Trainer()

    gcn_trainer.loadModel()
    gcn_trainer.loadDataset(dataset)

    gcn_trainer.train()

    gcn_trainer.detect(gcn_trainer.dataset[0])

