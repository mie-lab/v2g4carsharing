import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestWrapper:
    def __init__(self, max_depth=20) -> None:
        self.rf = RandomForestClassifier(max_depth=max_depth)

    def __call__(self, feature_vec):
        if not hasattr(self, "feat_columns"):
            raise RuntimeError("Forest must first be fitted!")
        if len(feature_vec.shape) < 2:
            feature_vec = np.array(feature_vec[self.feat_columns]).reshape(1, -1)
        pred_label = self.rf.predict(feature_vec)
        pred_label_str = self.label_meanings[pred_label]
        return pred_label_str

    def fit(self, features, labels):
        self.feat_columns = features.columns
        self.label_meanings = np.array(labels.columns)
        labels_max = np.argmax(np.array(labels), axis=1)
        self.rf.fit(features, labels_max)

    def save(self, save_name="rf_test"):
        with open(os.path.join("trained_models", save_name + ".p"), "wb") as outfile:
            pickle.dump(self, outfile)

