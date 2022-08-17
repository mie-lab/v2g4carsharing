import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


from torch.utils.data import Dataset


class ModeChoiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, ind):
        x = self.features[ind]
        y = self.labels[ind]
        return x, y


class MLP(nn.Module):
    def __init__(self, inp_size, out_size):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(inp_size, 128)
        self.linear_2 = nn.Linear(128, out_size)

    def forward(self, x):
        hidden = torch.relu(self.linear_1(x))
        out = self.linear_2(hidden)
        return out.softmax(dim=1)


def train_model(model, epochs, train_loader, test_loader, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()  # TODO: currently softmax in output layer, criterion here with enropy

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 1000 == 0:
                print("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_num, loss.item()))

        with torch.no_grad():
            test_losses = []
            for batch_num, input_data in enumerate(test_loader):
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y)
                test_losses.append(loss.item())

        print(
            f"\n Epoch {epoch} | TRAIN Loss {sum(losses) / len(losses)} | TEST loss {sum(test_losses) / len(test_losses)} \n"
        )


def test_model(model, X_test, y_test, col_labels):
    model.eval()
    out = model(torch.from_numpy(X_test).float()).detach().numpy()
    conf_matrix = confusion_matrix(y_test, out, labels=col_labels)  # confusion_matrix(y_true, y_pred)
    print(col_labels)
    print(conf_matrix)  # row is true label
    print()
    # sensitivity / specificity
    print(col_labels)
    print("sensitivity", conf_matrix.diagonal() / np.sum(conf_matrix, axis=1))
    print("specificity", conf_matrix.diagonal() / np.sum(conf_matrix, axis=0))
