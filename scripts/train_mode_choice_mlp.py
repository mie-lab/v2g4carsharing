import os
import sys
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from v2g4carsharing.mode_choice_model.mlp_baseline import ModeChoiceDataset, train_model, test_model, MLP
from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper, rf_tuning
from v2g4carsharing.mode_choice_model.evaluate import plot_confusion_matrix, mode_share_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_random_forest(trips_mobis, out_path=os.path.join("outputs", "mode_choice_model"), model_save_name="rf_sim"):

    f = open(os.path.join(out_path, "stdout_random_forest.txt"), "w")
    sys.stdout = f

    # prepare data
    features, labels = prepare_data(trips_mobis, return_normed=False)

    # Tuning and reporting test data performance
    labels_max_str = np.array(labels.columns)[np.argmax(np.array(labels), axis=1)]
    X_train, X_test, y_train, y_test = train_test_split(features, labels_max_str)
    # find best max depth
    best_acc = 0
    for max_depth in [20, 30, 50, None]:
        acc = rf_tuning(X_train, X_test, y_train, y_test, max_depth=max_depth)
        # save best parameter
        if acc > best_acc:
            best_acc = acc
            final_max_depth = max_depth
    # report test data performance
    rf_tuning(X_train, X_test, y_train, y_test, max_depth=final_max_depth, plot_confusion=True, out_path=out_path)

    # Fit on whole training data:
    rf_wrapper = RandomForestWrapper(max_depth=final_max_depth)
    rf_wrapper.fit(features, labels)

    # save train accuracy
    train_pred = rf_wrapper(features)
    plot_confusion_matrix(train_pred, labels_max_str, traintest="TRAIN", out_path=out_path)

    # save model
    rf_wrapper.save(save_name=model_save_name)

    f.close()


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_name", type=str, default="test_rf", help="name to save model",
    )
    parser.add_argument(
        "-i",
        "--in_path_mobis",
        type=str,
        default=os.path.join("..", "data", "mobis", "trips_features.csv"),
        help="path to mobis feature dataset",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=os.path.join("outputs", "mode_choice_model"),
        help="path to save training results",
    )
    args = parser.parse_args()

    # out path is a new directory with the model name
    out_path = os.path.join(args.out_path, args.save_name)
    os.makedirs(out_path, exist_ok=True)

    # load data
    trips_mobis = pd.read_csv(args.in_path_mobis)

    fit_random_forest(trips_mobis, out_path=out_path, model_save_name=args.save_name)

