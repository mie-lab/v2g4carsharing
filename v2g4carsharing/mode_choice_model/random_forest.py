import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from v2g4carsharing.mode_choice_model.evaluate import plot_confusion_matrix


def rf_tuning(X_train, X_test, y_train, y_test, max_depth=None, plot_confusion=False, out_path=None):
    rf = RandomForestClassifier(max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    car_sharing_pred = y_pred[y_test == "Mode::CarsharingMobility"]
    car_sharing_acc = sum(car_sharing_pred == "Mode::CarsharingMobility") / len(car_sharing_pred)
    print(f"Max depth {max_depth} bal accuracy {acc} car sharing sensitivity {car_sharing_acc}")
    if plot_confusion:
        plot_confusion_matrix(y_pred, y_test, traintest="TEST", out_path=out_path)
    return car_sharing_acc


class RandomForestWrapper:
    def __init__(self, max_depth=20) -> None:
        self.rf = xgb.XGBClassifier(max_depth=max_depth)
        # self.rf = RandomForestClassifier(max_depth=max_depth)

    def __call__(self, feature_vec):
        if not hasattr(self, "feat_columns"):
            raise RuntimeError("Forest must first be fitted!")
        # # prev version
        # feature_vec = feature_vec[self.feat_columns]
        # if len(feature_vec.shape) < 2:
        #     feature_vec = pd.DataFrame(feature_vec).T

        # select only the relevant feature columns
        feature_vec = np.array(feature_vec[self.feat_columns])
        # expand dims in case it's only one row
        if len(feature_vec.shape) < 2:
            feature_vec = feature_vec.reshape(1, -1)
        # predict
        pred_label = self.rf.predict(feature_vec)
        pred_label_str = self.label_meanings[pred_label]
        if len(pred_label_str) == 1:
            # if it's only one row, we return only the String, not an array
            return pred_label_str[0]
        return pred_label_str

    def fit(self, features, labels):
        self.feat_columns = features.columns
        self.label_meanings = np.array(labels.columns)
        labels_max = np.argmax(np.array(labels), axis=1)
        self.rf.fit(features, labels_max)

    def save(self, save_name="rf_test"):
        with open(os.path.join("trained_models", save_name + ".p"), "wb") as outfile:
            pickle.dump(self, outfile)

