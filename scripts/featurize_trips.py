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
            r2 = lambda x: round(x, 2)
            feat_comp_table.append(
                {
                    "feat": col,
                    "MOBIS (mean)": r2(np.mean(mobis_feat[col])),
                    "Simulated population (mean)": r2(np.mean(sim_feat[col])),
                    "MOBIS std": r2(np.std(mobis_feat[col])),
                    "Simulated population std": r2(np.std(sim_feat[col])),
                    "Z value (sim - mobis mean) / mobis std": r2(z),
                }
            )
        pd.DataFrame(feat_comp_table).to_csv(os.path.join("outputs", "feature_comparison.csv"))


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
    parser.add_argument(
        "-k", "--keep_geom", action="store_true", help="if flag set, keep the geometry column",
    )
    args = parser.parse_args()
    # Set paths:
    # for simulated: "../data/simulated_population/sim_2022"
    # for mobis: "../data/mobis/"

    in_path = args.in_path
    print("Removing geometry?", not args.keep_geom)

    feat_collector = ModeChoiceFeatures(in_path)
    feat_collector.add_all_features()
    feat_collector.save(remove_geom=(not args.keep_geom))

    # compare_feature_distributions(
    # "../data/mobis/trips_features.csv", "../data/simulated_population/sim_2022/trips_features.csv"
    # )
