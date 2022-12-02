import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


def simple_mode_choice(feature_vec):
    distance = feature_vec["distance"]
    if distance > 5 and np.random.rand() < 0.1:
        return "Mode::CarsharingMobility"
    return "Mode::Car"


def distance_dependent_mode_choice(feature_vec):
    distance = feature_vec["distance"]
    distance_to_station = feature_vec["distance_to_station_origin"]
    if distance < 2 * distance_to_station:
        return "Mode::Car"
    if np.random.rand() < 0.1:
        return "Mode::CarsharingMobility"
    return "Mode::Car"


class LogisticRegressionWrapper:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, features, labels):
        self.model.fit(features, labels)
        self.feat_columns = self.model.feature_names_in_
        self.label_meanings = self.model.classes_

    def __call__(self, feature_vec):

        if not hasattr(self, "feat_columns"):
            raise RuntimeError("Forest must first be fitted!")
        feature_vec = np.array(feature_vec[self.feat_columns])
        # expand dims in case it's only one row
        if len(feature_vec.shape) < 2:
            feature_vec = feature_vec.reshape(1, -1)
        pred_label = self.model.predict(feature_vec)
        if len(pred_label) == 1:
            # if it's only one row, we return only the String, not an array
            return pred_label[0]
        return pred_label

    def save(self, save_name="logistic_regression"):
        with open(os.path.join("trained_models", save_name + ".p"), "wb") as outfile:
            pickle.dump(self, outfile)
