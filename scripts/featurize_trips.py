import os
import argparse
import pandas as pd
import numpy as np
from v2g4carsharing.mode_choice_model.features import ModeChoiceFeatures


def compare_feature_distributions(mobis_feat_path, sim_feat_path):
    mobis_feat = pd.read_csv(mobis_feat_path)
    sim_feat = pd.read_csv(sim_feat_path)

    feat_comp_table = []
    for col in mobis_feat.columns:
        if "feat" in col and col in sim_feat.columns:
            z = (np.mean(sim_feat[col]) - np.mean(mobis_feat[col])) / np.std(mobis_feat[col])
            print()
            print(col)
            print(
                "mobis:",
                round(np.mean(mobis_feat[col]), 2),
                "sim:",
                round(np.mean(sim_feat[col]), 2),
                "STD:",
                round(np.std(mobis_feat[col]), 2),
            )
            if abs(z) > 1.96:
                print("ATTENTION: significant z value")
            feat_comp_table.append({
                    "mobis_mean": np.mean(mobis_feat[col]),
                    "mobis_std": np.std(mobis_feat[col]),
                    "sim_mean": np.mean(sim_feat[col]),
                    "sim_std": np.std(sim_feat[col]),
                    "z value": z
                })
        pd.DataFrame(feat_comp_table)


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

