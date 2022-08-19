import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
import torch
from torch.utils.data import DataLoader

from v2g4carsharing.mode_choice_model.mlp_baseline import ModeChoiceDataset, train_model, test_model, MLP
from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_random_forest(trips_mobis, trips_sim=None):

    # prepare data
    features, labels = prepare_data(trips_mobis, return_normed=False)
    rf_wrapper = RandomForestWrapper()
    rf_wrapper.fit(features, labels)

    # save model
    rf_wrapper.save()

    if trips_sim is not None:
        # Testing --> print unique
        features_sim = np.array(trips_sim[rf_wrapper.feat_columns])
        pred_sim = rf_wrapper.rf.predict(features_sim)
        pred_sim_str = rf_wrapper.label_meanings[pred_sim]
        uni, counts = np.unique(pred_sim_str, return_counts=True)
        print({u: c for u, c in zip(uni, counts)})
    # # Test prediction for single rows
    # for i, row in trips_sim.iterrows():
    #     out_test = rf_wrapper(row)
    #     print(out_test)
    #     if i > 10:
    #         break


def fit_mlp(trips_mobis):
    epochs = 2
    batch_size = 8

    # preprocess: get relevant features, remove nan rows, normalize
    features, labels, mean_std = prepare_data(trips_mobis)
    # to array
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


if __name__ == "__main__":

    # load data
    trips_mobis = pd.read_csv(os.path.join("..", "data", "mobis", "trips_features.csv"))
    trips_sim = pd.read_csv(os.path.join("../data/mobis", "trips_features.csv"))

    fit_random_forest(trips_mobis, trips_sim)

