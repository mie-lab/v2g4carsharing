import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from v2g4carsharing.mode_choice_model.mlp_baseline import ModeChoiceDataset, train_model, test_model, MLP
from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_random_forest(trips_mobis, trips_sim=None, out_path=os.path.join("outputs", "mode_choice_model")):
    def simple_rf_test(X_train, X_test, y_train, y_test, max_depth=None, plot_confusion=False):
        rf = RandomForestClassifier(max_depth=max_depth)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        car_sharing_pred = y_pred[y_test == "Mode::CarsharingMobility"]
        car_sharing_acc = sum(car_sharing_pred == "Mode::CarsharingMobility") / len(car_sharing_pred)
        print(f"Max depth {max_depth} bal accuracy {acc} car sharing sensitivity {car_sharing_acc}")
        if plot_confusion:
            plot_confusion_matrix(y_pred, y_test, traintest="TEST")
        return car_sharing_acc

    def plot_confusion_matrix(pred, labels_max_str, traintest="TRAIN"):
        print("----- ", traintest, "results")
        print("Acc:", accuracy_score(pred, labels_max_str))
        print("Balanced Acc:", balanced_accuracy_score(pred, labels_max_str))
        for confusion_mode in [None, "true", "pred"]:
            name = "" if confusion_mode is None else confusion_mode
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ConfusionMatrixDisplay.from_predictions(
                labels_max_str, pred, normalize=confusion_mode, xticks_rotation="vertical", values_format="2.2f", ax=ax
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, f"random_forest_{traintest}_confusion_{name}.png"))

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
        acc = simple_rf_test(X_train, X_test, y_train, y_test, max_depth=max_depth)
        # save best parameter
        if acc > best_acc:
            best_acc = acc
            final_max_depth = max_depth
    # report test data performance
    simple_rf_test(X_train, X_test, y_train, y_test, max_depth=final_max_depth, plot_confusion=True)

    # Fit on whole training data:
    rf_wrapper = RandomForestWrapper(max_depth=final_max_depth)
    rf_wrapper.fit(features, labels)

    # save train accuracy
    train_pred = rf_wrapper(features)
    plot_confusion_matrix(train_pred, labels_max_str, traintest="TRAIN")

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

    # load data
    trips_mobis = pd.read_csv(os.path.join("..", "data", "mobis", "trips_features.csv"))
    trips_sim = pd.read_csv(os.path.join("../data/mobis", "trips_features.csv"))

    fit_random_forest(trips_mobis, trips_sim)

