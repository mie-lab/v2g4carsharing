import os
import sys
import pandas as pd
import argparse
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper, rf_tuning
from v2g4carsharing.mode_choice_model.evaluate import plot_confusion_matrix, feature_importance_plot

def test_random_forest(trips_mobis, model_path=os.path.join("trained_models", "xgb_model.p"), out_path=os.path.join("outputs", "mode_choice_model")):

    # prepare data
    # drop_columns = [col for col in trips_mobis.columns if col.startswith("feat_prev_Mode")]  # ["feat_caraccess"]
    # features, labels = prepare_data(trips_mobis, return_normed=False, drop_columns=drop_columns)
    # print("fitting on features:", features.columns)

    # load model
    import pickle
    with open(model_path, "rb") as infile:
        mode_choice_model = pickle.load(infile)

    # included modes
    # with open("config.json", "r") as infile:
    #     included_modes = json.load(infile)["included_modes"]

    labels = trips_mobis[mode_choice_model.label_meanings]

    # Tuning and reporting test data performance
    labels_max_str = np.array(labels.columns)[np.argmax(np.array(labels), axis=1)]
    # X_train, X_test, y_train, y_test = train_test_split(features, labels_max_str, random_state=1)
    
    inp_rf = np.array(trips_mobis[mode_choice_model.feat_columns])
    pred = mode_choice_model.rf.predict(inp_rf)
    y_pred = np.array(mode_choice_model.label_meanings)[pred]
    
    # y_pred = rf_wrapper(trips_mobis)
    acc = balanced_accuracy_score(labels_max_str, y_pred)
    print("Balanced accuracy (test)", acc)
    print("Distribution predicted", np.unique(y_pred, return_counts=True))
    print("Distribution ground truth", np.unique(labels_max_str, return_counts=True))

    # y_pred = rf_wrapper(X_test)
    # acc = balanced_accuracy_score(y_test, y_pred)
    # print("Balanced accuracy (test)", acc)
    # y_pred = rf_wrapper(X_train)
    # acc = balanced_accuracy_score(y_train, y_pred)
    # print("Balanced accuracy (train)", acc)
    plot_confusion_matrix(y_pred, labels_max_str, traintest="TEST", out_path=out_path)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_load_path", type=str, default="trained_models/xgb_model.p", help="name to save model",
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
        default=os.path.join("outputs", "mode_choice_model", "test"),
        help="path to save training results",
    )
    args = parser.parse_args()

    # out path is a new directory with the model name
    out_path = os.path.join(args.out_path)
    os.makedirs(out_path, exist_ok=True)

    # load data
    trips_mobis = pd.read_csv(args.in_path_mobis)

    # # removing spatial outliers --> helps but not changing a lot
    # trips_mobis = trips_mobis[(trips_mobis["distance_to_station_origin"] < 50000) & (trips_mobis["distance_to_station_destination"] < 50000) ]
    
    # restrict to winter times
    # print(len(trips_mobis))
    # trips_mobis = trips_mobis[trips_mobis["started_at_origin"] > "2022-10-01"]
    # print(len(trips_mobis))
    
    # restrict to the users that we trained the model on (or not trained on)
    trips_current = pd.read_csv("../data/mobis/trips.csv")
    currently_used_users = trips_current["user_id"].unique()
    print(len(trips_mobis))
    trips_mobis = trips_mobis[~trips_mobis["person_id"].isin(currently_used_users)]
    
    print("Tested on trips:", len(trips_mobis))
    print("Number of unique users", trips_mobis["person_id"].nunique())

    test_random_forest(trips_mobis, model_path=args.model_load_path, out_path=out_path)

