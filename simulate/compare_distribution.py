import numpy as np
import pickle
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns


def compare_user_dist(res_real, res_sim):
    print("real", len(res_real["person_no"].unique()) / len(res_real))
    print("sim", len(res_sim["person_no"].unique()) / len(res_sim))


def compare_hist_dist(res_real, res_sim, col_name, out_path=None):
    print("Distribution of ", col_name)
    names = ["real", "sim"]
    plt.figure(figsize=(10, 4))
    for i, res in enumerate([res_real, res_sim]):
        plt.subplot(1, 2, i + 1)
        if col_name == "duration":
            bins = np.arange(24)
        elif col_name == "drive_km":
            bins = np.arange(0, 50, 1)
        elif col_name == "start_station_no":
            bins = np.arange(0, 15, 1)
        else:
            bins = np.arange(0, 85000, 5000)
        plt.hist(res[col_name], bins=bins)
        plt.title(names[i])
        if col_name == "duration":
            plt.xlim(0, 24)
    if out_path is not None:
        plt.savefig(os.path.join(out_path, col_name + "_dist.png"))
    else:
        plt.show()
    print("Wasserstein:", scipy.stats.wasserstein_distance(res_real[col_name], res_sim[col_name]))


def get_real_day(in_path, day="2020-01-20"):
    # load data
    reservation = pd.read_csv(os.path.join(in_path, "reservation.csv"), index_col="reservation_no")
    reservation["reservationfrom"] = pd.to_datetime(reservation["reservationfrom"])
    reservation["reservationto"] = pd.to_datetime(reservation["reservationto"])

    # define day bounds
    RANDOM_DATE_start = pd.to_datetime(day + " 00:00:00")
    RANDOM_DATE_end = pd.to_datetime(day + " 23:59:59")  # pd.to_datetime("2020-01-21 05:00:00")
    print("reservations between:", RANDOM_DATE_start, RANDOM_DATE_end)

    bef_day_ends = reservation["reservationto"] <= RANDOM_DATE_end
    after_day_starts = reservation["reservationfrom"] >= RANDOM_DATE_start
    res_within_day = reservation[bef_day_ends & after_day_starts]

    # # Reservations containing the day --> to check how many multi-day bookings we loose
    # bef_day = reservation["reservationfrom"] <= RANDOM_DATE_end
    # after_day = reservation["reservationto"] >= RANDOM_DATE_start
    # res_containing_day = reservation[bef_day & after_day]
    # # check distribution of trip duration --> how many are we missing out?
    # res_containing_day["period"] = (res_containing_day["reservationto"] - res_containing_day["reservationfrom"])
    # res_containing_day["duration"] = (res_containing_day["period"].dt.days * 24 * 60 * 60 +
    #                                   res_containing_day["period"].dt.seconds / 60 / 60)
    # for quant in np.arange(0.8, 1.0, 0.01):
    #     print(f"Number of hours for {round(quant, 2)}-quantile: {np.quantile(dur_vals, quant)}")

    # change encoding of res_from and res_to
    res_within_day["reservationfrom"] = (res_within_day["reservationfrom"] - RANDOM_DATE_start).dt.seconds
    res_within_day["reservationto"] = (res_within_day["reservationto"] - RANDOM_DATE_start).dt.seconds
    return res_within_day


if __name__ == "__main__":
    in_path_sim = os.path.join("outputs", "simulated_car_sharing")
    in_path_real = os.path.join("..", "v2g4carsharing", "data")
    sim_name = "simple"
    out_path = os.path.join("outputs", "simulated_car_sharing", "distribution_comp_" + sim_name)
    os.makedirs(out_path, exist_ok=True)

    res_real = get_real_day(in_path=in_path_real)
    res_sim = pd.read_csv(os.path.join(in_path_sim, sim_name + ".csv"), index_col="reservation_no")

    # user activity
    compare_user_dist(res_real, res_sim)

    # duration, start and end distributions
    res_real["duration"] = (res_real["reservationto"] - res_real["reservationfrom"]) / 60 / 60
    compare_hist_dist(res_real, res_sim, "duration", out_path=out_path)
    compare_hist_dist(res_real, res_sim, "reservationfrom", out_path=out_path)
    compare_hist_dist(res_real, res_sim, "reservationto", out_path=out_path)

    # driven distances
    compare_hist_dist(res_real, res_sim, "drive_km", out_path=out_path)

    # distribution over stations
    stations_sim_reservation = res_sim.groupby("start_station_no").agg({"start_station_no": "count"})
    stations_real_reservation = res_real.groupby("start_station_no").agg({"start_station_no": "count"})
    station_comp = (
        stations_sim_reservation.merge(stations_real_reservation, how="outer", left_index=True, right_index=True)
        .fillna(0)
        .rename(columns={"start_station_no_x": "sim_bookings", "start_station_no_y": "real_bookings"})
    )
    compare_hist_dist(stations_real_reservation, stations_sim_reservation, "start_station_no", out_path=out_path)

