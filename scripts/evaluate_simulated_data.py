import os
import argparse
import pandas as pd

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

    # real reservations for the whole time period
    res_real = get_real_daily(in_path_real)
    # simulated reservations
    res_sim = pd.read_csv(in_path_sim, index_col="reservation_no")

    # user activity
    compare_user_dist(res_real, res_sim)

    # duration, start and end distributions
    res_real["duration"] = (res_real["reservationto"] - res_real["reservationfrom"]) / 60 / 60
    compare_hist_dist(res_real, res_sim, "duration", out_path=out_path)
    compare_hist_dist(res_real, res_sim, "reservationfrom", out_path=out_path)
    compare_hist_dist(res_real, res_sim, "reservationto", out_path=out_path)

    # driven distances
    compare_hist_dist(res_real, res_sim, "drive_km", out_path=out_path)

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
    mode_share_plot(labels_mobis, labels_sim, out_path=out_path)

    # Further possible comparisons:
    # - distance driven by user
    # - borrowed duration by station / user
