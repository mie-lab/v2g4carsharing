import os
import argparse
import pandas as pd
import sys

from v2g4carsharing.simulate.compare_distribution import *
from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_path_sim",
        type=str,
        default=os.path.join("outputs", "simulated_car_sharing", "rf_sim"),
        help="path to simulated data",
    )
    parser.add_argument(
        "-d", "--real_data_path", type=str, default="data", help="path to Mobility car sharing data",
    )
    parser.add_argument(
        "-m", "--mobis_data_path", type=str, default=os.path.join("..", "data", "mobis"), help="path to MOBIS data",
    )
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()

    # define paths
    in_path_sim = os.path.join(args.in_path_sim, "sim_reservations.csv")
    in_path_real = args.real_data_path
    # save the distribution comparison in the same directory
    out_path = args.in_path_sim

    os.makedirs(out_path, exist_ok=True)

    f = open(os.path.join(out_path, "evaluation.txt"), "w")
    sys.stdout = f

    # real reservations for the whole time period
    res_real = get_real_daily(in_path_real)
    # simulated reservations
    res_sim = pd.read_csv(in_path_sim, index_col="reservation_no")

    # histogram of number of reservations and unique users
    compare_nr_reservations(res_real, res_sim, out_path=out_path)
    compare_user_dist(res_real, res_sim, out_path=out_path)

    # duration, start and end distributions
    for var in ["duration", "reservationfrom", "reservationto", "drive_km"]:
        if "test" in out_path and var not in ["reservationfrom"]:
            # start and end times do not make sense without
            continue
        compare_hist_dist(res_real, res_sim, var, out_path=out_path)

    # station distribution for a single real day
    # compare_station_dist_one_day(res_sim, in_path_real, out_path)

    # station z values compared to all other days
    compare_station_dist(res_real, res_sim, out_path=out_path)

    # Mode share comparison
    # load simulated labels
    sim_modes = pd.read_csv(os.path.join(args.in_path_sim, "sim_modes.csv"))
    labels_sim = sim_modes["mode"].values
    # Load mobis labels
    mobis_data = pd.read_csv(os.path.join(args.mobis_data_path, "trips_features.csv"))
    mobis_data = mobis_data[[col for col in mobis_data.columns if col in np.unique(labels_sim)]]
    labels_mobis = np.array(mobis_data.columns)[np.argmax(np.array(mobis_data), axis=1)]
    carsharing_share_mobis = (np.sum(labels_mobis == "Mode::CarsharingMobility")) / len(labels_mobis)
    carsharing_share_sim = (np.sum(labels_sim == "Mode::CarsharingMobility")) / len(labels_sim)
    print("Ratio  of sim mode share vs mobis mode share (should be 1)", carsharing_share_sim / carsharing_share_mobis)
    mode_share_plot(labels_mobis, labels_sim, out_path=out_path)

    f.close()
