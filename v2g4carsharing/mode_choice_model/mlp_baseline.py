import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split


from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def prepare_data(trips, min_number_trips=500):
    dataset = trips.drop(["geom", "geom_origin", "geom_destination"], axis=1)
    print("Dataset raw", len(dataset))
    # only include frequently used modes
    nr_trips_with_mode = trips[[col for col in trips.columns if col.startswith("Mode")]].sum()
    included_modes = list(nr_trips_with_mode[nr_trips_with_mode > min_number_trips].index.tolist())
    print("included_modes", included_modes)
    # TODO: group into public transport, slow transport, car, shared car
    dataset = dataset[dataset[included_modes].sum(axis=1) > 0]
    print("after removing other modes:", len(dataset))

    # only get feature and label columns
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    dataset = dataset[feat_cols + included_modes]

    # drop columns with too many nans:
    max_unavail = 0.1  # if more than 10% are NaN
    feature_avail_ratio = pd.isna(dataset).sum() / len(dataset)
    features_not_avail = feature_avail_ratio[feature_avail_ratio > max_unavail].index
    dataset.drop(features_not_avail, axis=1, inplace=True)
    print("dataset len now", len(dataset))

    # remove other NaNs (the ones because of missing origin or destination ID)
    dataset.dropna(inplace=True)
    print("dataset len after dropna", len(dataset))

    # convert features to array
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    feat_array = np.array(dataset[feat_cols], dtype=np.float64)
    # normalize
    feat_array_normed = (feat_array - np.mean(feat_array, axis=0)) / np.std(feat_array, axis=0)

    labels = np.array(dataset[included_modes])
    print("labels", labels.shape, "features", feat_array_normed.shape)
    labels_max = np.argmax(labels, axis=1)

    return feat_array_normed, labels


def train_model(model, epochs, train_loader, test_loader):
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


def test(model, X_test, y_test, col_labels):
    out = model(torch.from_numpy(X_test).float())
    conf_matrix = confusion_matrix(y_test, out, labels=col_labels)  # confusion_matrix(y_true, y_pred)
    print(col_labels)
    print(conf_matrix)  # row is true label
    print()
    # sensitivity / specificity
    print(col_labels)
    print("sensitivity", conf_matrix.diagonal() / np.sum(conf_matrix, axis=1))
    print("specificity", conf_matrix.diagonal() / np.sum(conf_matrix, axis=0))


if __name__ == "__main__":

    epochs = 10
    batch_size = 8

    # load data
    trips = pd.read_csv(os.path.join("../data/mobis", "trips_enriched.csv"))

    # prepare data
    feat_array_normed, labels = prepare_data(trips)

    # make dataset
    X_train, X_test, y_train, y_test = train_test_split(feat_array_normed, labels, test_size=0.2, shuffle=False)
    train_set = ModeChoiceDataset(X_train, y_train)
    test_set = ModeChoiceDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # train
    model = MLP(feat_array_normed.shape[1], labels.shape[1]).to(device)
    train_model(model, epochs, train_loader, test_loader)
