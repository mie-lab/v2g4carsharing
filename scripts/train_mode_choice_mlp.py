import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
import torch
from torch.utils.data import DataLoader

from v2g4carsharing.mode_choice_model.mlp_baseline import ModeChoiceDataset, train_model, test_model, MLP
from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    epochs = 2
    batch_size = 8

    # load data
    trips = pd.read_csv(os.path.join("../data/mobis", "trips_enriched.csv"))

    # prepare data
    features, labels = prepare_data(trips)

    feat_array_normed = np.array(features, dtype=np.float64)
    label_array = np.array(labels, dtype=np.float64)

    # make dataset
    X_train, X_test, y_train, y_test = train_test_split(feat_array_normed, label_array, test_size=0.2, shuffle=False)
    train_set = ModeChoiceDataset(X_train, y_train)
    test_set = ModeChoiceDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # train
    model = MLP(feat_array_normed.shape[1], label_array.shape[1]).to(device)
    train_model(model, epochs, train_loader, test_loader, device=device)

    test_model(model, X_test, y_test, labels.columns)
