import os
import sys
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from v2g4carsharing.mode_choice_model.mlp_baseline import ModeChoiceDataset, train_model, test_model, MLP
from v2g4carsharing.mode_choice_model.prepare_train_data import prepare_data
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper, rf_tuning
from v2g4carsharing.mode_choice_model.evaluate import plot_confusion_matrix, feature_importance_plot
from v2g4carsharing.mode_choice_model.simple_choice_models import LogisticRegressionWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_random_forest(trips_mobis, out_path=os.path.join("outputs", "mode_choice_model"), model_save_name="rf_sim"):

    # prepare data
    drop_columns = [col for col in trips_mobis.columns if col.startswith("feat_prev_Mode") or col in [
        # Version 2: without swiss-specific: 'feat_halbtax', 'feat_ga',

        # VERSION 3: without any specific
        #    'feat_age', 'feat_sex', 'feat_caraccess', 'feat_employed',
        #    'feat_purpose_destination_home',
        #    'feat_purpose_destination_leisure', 'feat_purpose_destination_work',
        #    'feat_purpose_destination_shopping',
        #    'feat_purpose_destination_education', 'feat_purpose_origin_home',
        #    'feat_purpose_origin_leisure', 'feat_purpose_origin_work',
        #    'feat_purpose_origin_shopping', 'feat_purpose_origin_education',
        #    'feat_pt_accessibilityorigin', 'feat_pt_accessibilitydestination',
    ]]  # ["feat_caraccess"]
    features, labels = prepare_data(trips_mobis, return_normed=False, drop_columns=drop_columns)
    print("fitting on features:", features.columns)

    # Tuning and reporting test data performance
    labels_max_str = np.array(labels.columns)[np.argmax(np.array(labels), axis=1)]
    X_train, X_test, y_train, y_test = train_test_split(features, labels_max_str, random_state=1)
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

    # print most important features
    feature_importance_plot(rf_wrapper.feat_columns, rf_wrapper.rf.feature_importances_, out_path=out_path)

    # save model
    rf_wrapper.save(save_name=model_save_name)

    # # for debugging:
    # trips_sim = pd.read_csv("../data/simulated_population/sim_2019/trips_features.csv")
    # features_sim = np.array(trips_sim[rf_wrapper.feat_columns])
    # pred_sim = rf_wrapper.rf.predict(features_sim)
    # labels_sim = rf_wrapper.label_meanings[pred_sim]
    # carsharing_share_mobis = (np.sum(labels_max_str == "Mode::CarsharingMobility")) / len(labels_max_str)
    # carsharing_share_sim = (np.sum(labels_sim == "Mode::CarsharingMobility")) / len(labels_sim)
    # print(carsharing_share_mobis, carsharing_share_sim, np.unique(labels_sim, return_counts=True))
    # print("Ratio  of sim mode share vs mobis mode share (should be 1)", carsharing_share_sim / carsharing_share_mobis)


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


def fit_logistic_regression(trips_mobis, model_save_name):
    cutoff = 0.5
    # binary task --> how many to include
    balance_ratio = 2
    nr_carsharing = sum(trips_mobis["Mode::CarsharingMobility"] > cutoff)

    subset = trips_mobis[trips_mobis["Mode::CarsharingMobility"] < cutoff].index.values
    inds = np.random.permutation(len(subset))
    subset = subset[inds[: int(nr_carsharing * balance_ratio)]]
    data = pd.concat((trips_mobis.loc[subset], trips_mobis[trips_mobis["Mode::CarsharingMobility"] > cutoff]))

    features = data[[col for col in data.columns if col.startswith("feat")]]

    data["label"] = pd.NA
    data.loc[data["Mode::CarsharingMobility"].values > cutoff, "label"] = "Mode::CarsharingMobility"
    data["label"].fillna("Mode::Other", inplace=True)

    model = LogisticRegression()
    model.fit(features, data["label"])
    wrapped_model = LogisticRegressionWrapper()
    wrapped_model.model = model
    wrapped_model.feat_columns = model.feature_names_in_
    wrapped_model.label_meanings = model.classes_

    # no split necessary because the model cannot overfit
    pred = wrapped_model(features)
    print("----- TEST results")
    print("Acc:", accuracy_score(pred, data["label"]))
    print("Balanced Acc:", balanced_accuracy_score(pred, data["label"]))

    print(classification_report(pred, data["label"]))

    wrapped_model.save(model_save_name)


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

    f = open(os.path.join(out_path, "stdout_model_train_test.txt"), "w")
    sys.stdout = f

    if "logistic" in args.save_name:
        fit_logistic_regression(trips_mobis, model_save_name=args.save_name)
    else:
        fit_random_forest(trips_mobis, out_path=out_path, model_save_name=args.save_name)

    f.close()

