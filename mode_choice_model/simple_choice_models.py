import numpy as np


def simple_mode_choice(distance_to_act):
    if distance_to_act > 5 and np.random.rand() < 0.1:
        return "shared"
    return "car"


def distance_dependent_mode_choice(distance_to_act, distance_to_station):
    if distance_to_act < 2 * distance_to_station:
        return "car"
    if np.random.rand() < 0.1:
        return "shared"
    return "car"
