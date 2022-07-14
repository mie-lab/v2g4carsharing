import numpy as np
import pandas as pd
import geopandas as gpd
import os
from shapely import wkt


def get_trips_per_station(reservation):
    # TODO: here could Dominik's model go:
    # assume input of dominiks model: number of trips per stations for one month
    monthly_trips_per_station = reservation.groupby("start_station_no").count()["reservation_no"]
    return monthly_trips_per_station


def get_station_user_stats(use_df, group_var="nearest_station_no", shortcut="booking"):
    user_stats = pd.DataFrame()
    # Add user gender distribution
    user_stats[shortcut + "_user_total"] = use_df.groupby(group_var).count()["gender"]
    count_m = use_df.groupby(group_var)["gender"].apply(lambda x: np.sum(x == "m"))
    user_stats[shortcut + "_gender_m"] = count_m / user_stats[shortcut + "_user_total"]
    user_stats[shortcut + "_gender_w"] = 1 - user_stats[shortcut + "_gender_m"]
    # Add user age distribution
    for agegroup in sorted(use_df["agegroup_int"].unique()):
        if pd.isna(agegroup):
            continue
        user_stats[shortcut + f"_agegroup_int_{agegroup}"] = use_df.groupby(group_var)["agegroup_int"].apply(
            lambda x: np.sum(x == agegroup) / len(x)
        )
    return user_stats


def get_default_sample_weights(res_raw, population_vars):
    grouped = res_raw.groupby(population_vars).agg({"reservation_no": "count"})
    grouped["default_sample_weight"] = 1 / (grouped["reservation_no"] / grouped["reservation_no"].sum())
    grouped["default_sample_weight"] = grouped["default_sample_weight"] / grouped["default_sample_weight"].sum()
    res_raw = res_raw.merge(grouped["default_sample_weight"], left_on=population_vars, right_index=True)
    return res_raw


def sample_booking_data(
    res_raw, station, monthly_trips_per_station, population_vars=["gender", "agegroup_int"], shortcut="booking"
):
    """for each station, sample the data for one month --> for now, only simulate one month! that's sufficient"""
    station_wise_tables = []
    for station_no in monthly_trips_per_station.index:
        # population dist for one station
        population_dist = station.loc[station_no]

        # set to default in the beginning
        res_raw["sample_weight"] = res_raw["default_sample_weight"]

        # iterate over attributes and set the sample weights according to them
        for attr in population_vars:
            for attr_value in np.unique(res_raw[attr]):
                res_raw.loc[res_raw[attr] == attr_value, "sample_weight"] *= population_dist[
                    f"{shortcut}_{attr}_{attr_value}"
                ]

        if np.nansum(res_raw["sample_weight"].values) == 0:
            print("All sample weights NaN, continue for station", station_no)
            continue

        # sample the reservations
        nr_samples = monthly_trips_per_station.loc[station_no]
        one_month_samples = res_raw.sample(nr_samples, replace=False, weights=res_raw["sample_weight"])

        # set station number
        one_month_samples["start_station_no"] = station_no
        one_month_samples["end_station_no"] = station_no

        # append
        station_wise_tables.append(one_month_samples)

    # concat and compress to one month
    one_month_data = pd.concat(station_wise_tables)
    # scale everything to one month
    for time_attr in ["reservationfrom", "reservationto", "drive_firststart", "drive_lastend"]:
        one_month_data[time_attr] = pd.to_datetime(one_month_data[time_attr])
        one_month_data[time_attr] = one_month_data[time_attr] - one_month_data[time_attr].apply(
            lambda x: pd.DateOffset(months=(x.month - 1) + 12 * (x.year - 2019)) if not pd.isna(x) else x
        )
    return one_month_data


def reorder_overlapping_bookings(one_station_data, nr_veh_at_station, nr_iters=20):
    """Reorder the bookings for one station such that they are feasible"""
    # add ordered IDs
    fake_veh_ids = (
        np.arange(nr_veh_at_station)
        .reshape((1, nr_veh_at_station))
        .repeat(len(one_station_data) // nr_veh_at_station + 1, axis=0)
        .flatten()[: len(one_station_data)]
    )
    one_station_data.loc[:, "veh_id_new"] = fake_veh_ids

    # iteratively improve the result
    # print("Overall:", len(one_station_data))
    for i in range(nr_iters):
        one_station_data = one_station_data.sort_values(["veh_id_new", "reservationfrom", "reservationto"])

        one_station_data["prev_end"] = one_station_data["reservationto"].shift(1)
        one_station_data["prev_veh"] = one_station_data["veh_id_new"].shift(1)

        cond1 = one_station_data["prev_veh"] == one_station_data["veh_id_new"]
        cond2 = one_station_data["prev_end"] > one_station_data["reservationfrom"]
        # print(len(one_station_data[cond1 & cond2]))

        # permute IDs
        #     print(one_station_data[cond1 & cond2]["veh_id_new"])
        #     print(one_station_data[cond1 & cond2][["reservationfrom", "reservationto"]])

        day_shift = np.random.randint(0, 31)
        for time_attr in ["reservationfrom", "reservationto"]:
            one_station_data.loc[cond1 & cond2, time_attr] = one_station_data.loc[cond1 & cond2, time_attr].apply(
                lambda x: x + pd.DateOffset(days=day_shift - x.day)
            )
        # Verion 2: change veh ID
        one_station_data.loc[cond1 & cond2, "veh_id_new"] = np.random.permutation(
            one_station_data[cond1 & cond2]["veh_id_new"].values
        )
    return one_station_data


####### EVALUATE DISTRIBUTION SHIFT
def compare_nr_trips(one_month_data, monthly_trips_per_station):
    # check whether the number of trips per station are equal
    new_monthly_trips_per_station = pd.DataFrame(one_month_data.groupby("start_station_no").count()["reservation_no"])
    both_compared = new_monthly_trips_per_station.merge(
        pd.DataFrame(monthly_trips_per_station), how="left", left_index=True, right_index=True
    )
    print(both_compared.head(20))


def compare_user_stats(one_month_data, res_raw):
    # check if the user stats are similar from sampling
    user_stats_real = get_station_user_stats(res_raw, group_var="start_station_no")
    # fake_data_user = one_month_data.merge(user, how="left", left_on="person_no", right_index=True)
    user_stats_fake = get_station_user_stats(one_month_data, group_var="start_station_no")
    compare_both = user_stats_real.merge(user_stats_fake, how="left", left_index=True, right_index=True)
    compare_cols = [col for col in compare_both.columns if "_x" in col]
    compare_cols = np.array([[c, c[:-2] + "_y"] for c in compare_cols]).flatten()
    print(compare_both[compare_cols].head(30))


if __name__ == "__main__":
    base_path = "../v2g4carsharing/data"

    # one month period
    time_start = "2019-01-01"
    time_end = "2019-02-01"
    population_vars = ["gender", "agegroup_int"]

    # READ DATA
    # read reservation
    res_raw = pd.read_csv(os.path.join(base_path, "reservation.csv"))
    # read user
    user_raw = pd.read_csv(os.path.join(base_path, "user_enriched.csv"), index_col="person_no")
    user_raw["geom"] = user_raw["geom"].apply(wkt.loads)
    user_raw = gpd.GeoDataFrame(user_raw, geometry="geom")
    # read station
    station = pd.read_csv(os.path.join(base_path, "station.csv"), index_col="station_no")
    station["geom"] = station["geom"].apply(wkt.loads)
    station = gpd.GeoDataFrame(station, geometry="geom")

    # restrict to time period
    reservation = res_raw[(res_raw["reservationfrom"] >= time_start) & (res_raw["reservationto"] <= time_end)]
    user = user_raw[user_raw["person_no"].isin(reservation["person_no"].unique())]
    station = station[station.index.isin(reservation["start_station_no"].unique())]

    # only keep relevant columns of raw data
    user_raw = user_raw[population_vars]
    res_raw = res_raw.merge(user_raw, how="left", right_index=True, left_on="person_no")
    res_raw = res_raw[~pd.isna(res_raw["gender"])]
    res_raw = res_raw.drop(
        [
            "reservationtype",
            "reservationstate",
            "tripmode",
            "syscreatedate",
            "canceldate",
            "revenue_duration",
            "revenue_distance",
            "revenue_other",
        ],
        axis=1,
    )

    # GET SOCIODEMOGRAPHICS PER STATION
    # res_w_user = reservation.merge(user, how="left", left_on="person_no", right_index=True)
    # user_stats = get_station_user_stats(res_w_user, group_var="start_station_no")
    user_stats = get_station_user_stats(user, group_var="nearest_station_no")
    station = station.merge(user_stats, how="left", left_index=True, right_index=True)

    # SAMPLE FROM ORIG DATA
    # get default sample weight dependent on the overall distribution of age and gender
    res_raw = get_default_sample_weights(res_raw, population_vars)
    # get monthly trip per station (using the res data for one month)
    monthly_trips_per_station = get_trips_per_station(reservation)
    # sample
    sampled_data = sample_booking_data(res_raw, station, monthly_trips_per_station, population_vars)
    # reorder (because there are many overlapping bookings now)
    realistic_sampled_data = []
    for station_no, station_df in sampled_data.groupby("start_station_no"):
        # TODO!! the next line is only a bad approximation
        nr_veh_at_station = len(res_raw[res_raw["start_station_no"] == station_no]["vehicle_no"].unique())
        new_month_df = reorder_overlapping_bookings(station_df, nr_veh_at_station)
        realistic_sampled_data.append(new_month_df)
    realistic_sampled_data = pd.concat(realistic_sampled_data)
