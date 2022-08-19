import numpy as np


def simple_mode_choice(feature_vec):
    distance = feature_vec["distance"]
    if distance > 5 and np.random.rand() < 0.1:
        return "shared"
    return "car"


def distance_dependent_mode_choice(feature_vec):
    distance = feature_vec["distance"]
    distance_to_station = feature_vec["distance_to_station_origin"]
    if distance < 2 * distance_to_station:
        return "car"
    if np.random.rand() < 0.1:
        return "shared"
    return "car"
