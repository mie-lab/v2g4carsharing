import numpy as np
import pickle
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import scipy
import seaborn as sns


def compare_user_dist(res_real, res_sim):
    # same for user distribution
    uni_person_per_day = (
        res_real.reset_index().groupby("start_date").agg({"reservation_no": "count", "person_no": "nunique"})
    )
    uni_person_per_day["ratio_unique"] = uni_person_per_day["person_no"] / uni_person_per_day["reservation_no"]
    mean, std = uni_person_per_day["ratio_unique"].mean(), uni_person_per_day["ratio_unique"].std()
    print(f"real mean: {round(mean, 2)}, std: {round(std, 2)}")
    # print("real", len(res_real["person_no"].unique()) / len(res_real))
    sim_uni_user_ratio = len(res_sim["person_no"].unique()) / len(res_sim)
    print("sim", sim_uni_user_ratio)
    print("z value for user dist", (sim_uni_user_ratio - mean) / std)


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
        elif "station" in col_name:
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


def compare_station_dist_one_day(res_sim, in_path_real, out_path):
    # load real data for one day
    res_real = get_real_day(in_path_real)
    # distribution over stations
    stations_sim_reservation = (
        res_sim.groupby("start_station_no")
        .agg({"start_station_no": "count"})
        .rename(columns={"start_station_no": "station_one_day"})
    )
    stations_real_reservation = (
        res_real.groupby("start_station_no")
        .agg({"start_station_no": "count"})
        .rename(columns={"start_station_no": "station_one_day"})
    )
    # station_comp = (
    #     stations_sim_reservation.merge(stations_real_reservation, how="outer", left_index=True, right_index=True)
    #     .fillna(0)
    #     .rename(columns={"start_station_no_x": "sim_bookings", "start_station_no_y": "real_bookings"})
    # )
    compare_hist_dist(stations_real_reservation, stations_sim_reservation, "station_one_day", out_path=out_path)


def compare_station_dist(res_real, res_sim, out_path=None):
    res_per_day_per_station = (
        res_real.groupby(["start_date", "start_station_no"])
        .agg({"start_station_no": "count"})
        .rename(columns={"start_station_no": "res_count"})
        .reset_index()
    )
    mean_std_res_count = res_per_day_per_station.groupby("start_station_no").agg({"res_count": ["mean", "std"]})
    stations_sim_reservation = (
        res_sim.groupby("start_station_no")
        .agg({"start_station_no": "count"})
        .rename(columns={"start_station_no": "sim_res_count"})
    )
    stations_compare = stations_sim_reservation.merge(mean_std_res_count, how="left", left_index=True, right_index=True)
    # remove std=0
    null_std = stations_compare[("res_count", "std")] == 0
    print(f"Removing {sum(null_std)} stations because std=0")
    stations_compare = stations_compare[~null_std]
    # compute z score
    stations_compare["z_score"] = (
        stations_compare["sim_res_count"] - stations_compare[("res_count", "mean")]
    ) / stations_compare[("res_count", "std")]
    # print average
    print("Average z score", np.mean(np.abs(stations_compare["z_score"])))

    # plot histogram
    plt.figure()
    plt.hist(stations_compare["z_score"], bins=30)
    if out_path is not None:
        plt.savefig(os.path.join(out_path, "zscore_dist.png"))
    else:
        plt.show()

    # Compare average day with histogram comparison
    stations_real_reservation = stations_compare[[("res_count", "mean")]].rename(
        columns={("res_count", "mean"): "station"}
    )
    stations_sim_reservation = stations_compare[["sim_res_count"]].rename(columns={"sim_res_count": "station"})
    compare_hist_dist(stations_real_reservation, stations_sim_reservation, "station", out_path=out_path)


def load_real_reservation(in_path):
    # load data
    reservation = pd.read_csv(os.path.join(in_path, "reservation.csv"), index_col="reservation_no")
    reservation["reservationfrom"] = pd.to_datetime(reservation["reservationfrom"])
    reservation["reservationto"] = pd.to_datetime(reservation["reservationto"])
    # only return drips
    reservation = reservation[reservation["tripmode"] == "Return (Rückgabe an derselben Station)"]
    # filter out the ones that were no actual drive
    reservation = reservation[(~pd.isna(reservation["drive_km"])) & (reservation["drive_km"] > 0)]
    return reservation


def get_real_day(in_path, day="2020-01-20"):
    reservation = load_real_reservation(in_path)

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


def get_real_daily(in_path):
    # load data
    reservation = load_real_reservation(in_path)
    # filter for the ones on the same day
    on_same_day = reservation["reservationfrom"].dt.date == reservation["reservationto"].dt.date
    res_real = reservation[on_same_day]
    print("Percentage of res on same day vs all res: ", len(res_real) / len(reservation))

    res_real["start_date"] = pd.to_datetime(reservation["reservationfrom"].dt.date)
    # change encoding of res_from and res_to to seconds
    res_real["reservationfrom"] = (res_real["reservationfrom"] - res_real["start_date"]).dt.seconds
    res_real["reservationto"] = (res_real["reservationto"] - res_real["start_date"]).dt.seconds
    return res_real


if __name__ == "__main__":
    in_path_sim = os.path.join("outputs", "simulated_car_sharing")
    in_path_real = os.path.join("..", "v2g4carsharing", "data")
    sim_name = "simple"
    out_path = os.path.join("outputs", "simulated_car_sharing", "distribution_comp_" + sim_name)
    os.makedirs(out_path, exist_ok=True)

    # real reservations for the whole time period
    res_real = get_real_daily(in_path_real)
    # simulated reservations
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

    # station distribution for a single real day
    # compare_station_dist_one_day(res_sim, in_path_real, out_path)

    # station z values compared to all other days
    compare_station_dist(res_real, res_sim, out_path=out_path)

    # Further possible comparisons:
    # - distance driven by user
    # - borrowed duration by station / user

