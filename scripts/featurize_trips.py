import os
import argparse
from v2g4carsharing.mode_choice_model.features import ModeChoiceFeatures

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_path",
        type=str,
        default=os.path.join("..", "data", "simulated_population", "sim_2022"),
        help="path to preprocessed trips",
    )
    args = parser.parse_args()
    # Set paths:
    # for simulated: "../data/simulated_population/sim_2022"
    # for mobis: "../data/mobis/"

    in_path = args.in_path

    feat_collector = ModeChoiceFeatures(in_path)
    feat_collector.add_all_features()
    feat_collector.save()

