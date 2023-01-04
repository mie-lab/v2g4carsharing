import os
import time
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from v2g4carsharing.simulate.car_sharing_patterns import load_stations
from v2g4carsharing.import_data.import_utils import write_geodataframe


import warnings


def station_scenario(
    station_scenario="all", vehicle_scenario=5000, in_path="data/", sim_date="2020-01-01 00:00:00",
):
    # current scenario: only 1500 stations, as claimed on the website
    # all_stations scenario: 1916 stations (maximum that appear in the reservations)
    # new stations scenario: newXX_stations places XX further stations somehwere
    # "all_stations" or "current" or "30_new_stations"

    v2base = pd.read_csv(os.path.join(in_path, "vehicle_to_base.csv"))
    station = load_stations(in_path)
    print(len(station))
    station = station[station["in_reservation"]]

    # get the vehicle distribution over stations for one day
    v2base_at_sim_date = v2base[(v2base["bizbeg"] <= sim_date) & (v2base["bizend"] > sim_date)]
    v2base_at_sim_date = (
        v2base_at_sim_date.groupby("station_no")
        .agg({"vehicle_no": list})
        .rename(columns={"vehicle_no": "vehicle_list"})
    )

    # Merge with the station geometry such that all stations appear (the ones in the reservations)
    # Note: some have pd.NA in the vehicle list if no vehicle was there
    # already the final table, however, some entries in vehicle_list are NaN
    station_veh_list = station[["geom"]].merge(v2base_at_sim_date, how="left", left_index=True, right_index=True)
    print(type(station_veh_list))

    # remove stations with invalid geometry
    print(len(station_veh_list))
    station_veh_list = station_veh_list[station_veh_list["geom"].x != 0]
    station_veh_list = station_veh_list[station_veh_list["geom"].y >= 0]
    print("after removing some with invalid geometry:", len(station_veh_list))

    # if we want a realistic current situation, we drop the stations that have no vehicles assigned
    if station_scenario == "current":
        station_veh_list = station_veh_list.dropna()
        print("After reducing to the current stations", len(station_veh_list))
    # place new stations if necessary
    elif station_scenario.startswith("new"):
        nr_new_stations = int(station_scenario[3:])
        current_station_locations = np.swapaxes(
            np.array([station_veh_list["geom"].x, station_veh_list["geom"].x]), 1, 0
        )
        new_stations = place_new_stations(nr_new_stations, current_station_locations, subsample_population=500000)
        start_ind = np.max(station_veh_list.index)
        new_stations["station_no"] = np.arange(start_ind, start_ind + len(new_stations), 1)
        new_stations["vehicle_list"] = pd.NA
        new_stations = new_stations.rename(columns={"geometry": "geom"}).set_index("station_no")
        station_veh_list = pd.concat((station_veh_list, new_stations))

    print(
        "Number of stations:",
        len(station_veh_list),
        ", davon stations without vehicle:",
        pd.isna(station_veh_list["vehicle_list"]).sum(),
    )

    # ======== Assign vehicles ==========
    def get_veh_list_from_df(veh_list_values):
        nested_lists = [l for l in veh_list_values if type(l) == list]
        all_occuring_vehicles = [e for elem in nested_lists for e in elem]
        return all_occuring_vehicles

    uni_veh = get_veh_list_from_df(station_veh_list["vehicle_list"].values)

    station_veh_list["nr_veh"] = station_veh_list["vehicle_list"].apply(lambda x: len(x) if type(x) == list else 1)

    if vehicle_scenario <= station_veh_list["nr_veh"].sum():
        print("Keep current nr of vehicles:", station_veh_list["nr_veh"].sum())
        # option 1: we don't need to scale the number of vehicles
        station_veh_list["nr_veh_desired"] = station_veh_list["nr_veh"]
    else:
        # Option 2: scenario has more vehicles than currently in the data, so scale them
        veh_scale_factor = vehicle_scenario / station_veh_list["nr_veh"].sum()

        # Simulate a new distribution of vehicles over station by scaling the current number by veh_scale_factor
        max_trials = 100
        max_dev = 0.005 * vehicle_scenario
        dev, trial_counter = np.inf, 0
        while trial_counter < max_trials and dev > max_dev:
            desired_veh_per_station = (
                (station_veh_list["nr_veh"] * np.random.normal(veh_scale_factor, 0.3, size=len(station_veh_list)))
                .round()
                .astype(int)
            )
            dev = abs(desired_veh_per_station.sum() - vehicle_scenario)
            trial_counter += 1
        if trial_counter >= max_trials:
            warnings.warn("Could not find a suitable vehicle number distribution in feasible time")
        # save the result in a new column
        station_veh_list["nr_veh_desired"] = desired_veh_per_station

    # get possible vehicle IDs to fill the gaps
    possible_veh_ids = pd.read_csv(os.path.join(in_path, "vehicle.csv"))["vehicle_no"]
    possible_veh_ids = possible_veh_ids[~(possible_veh_ids.isin(uni_veh))].values

    # Iterate over stations and update / add the vehicle IDs
    new_id_counter = 0
    vehicle_set = set()
    all_new_lists = []
    for _, row in station_veh_list.iterrows():
        veh_list = row["vehicle_list"]
        #     desired_veh_list = row["nr_veh_desired"]
        # new stations --> no vehicle assigned yet, set to empty list
        if type(veh_list) != list:
            assert pd.isna(veh_list)
            veh_list = []

        # fill list of new vehicle IDs iteratively and use new IDs where necessary
        new_list = []
        for i in range(row["nr_veh_desired"]):
            if i < len(veh_list):
                veh = veh_list[i]
                if veh in vehicle_set:
                    new_list.append(possible_veh_ids[new_id_counter])
                    new_id_counter += 1
                else:
                    new_list.append(veh)
            else:
                # new vehicle ID must be used
                new_list.append(possible_veh_ids[new_id_counter])
                new_id_counter += 1

        # add to the final Series that will be the new list of vehicles
        all_new_lists.append(new_list)
        # add the newly used IDs to the set to ensure uniqueness
        vehicle_set.update(new_list)

    station_veh_list["vehicle_list"] = all_new_lists

    # test again
    uni_vals = get_veh_list_from_df(station_veh_list["vehicle_list"].values)
    assert len(uni_vals) == station_veh_list["nr_veh_desired"].sum()

    print(f"Final scenario has {len(station_veh_list)} stations and {len(uni_vals)} vehicles")

    return station_veh_list[["vehicle_list", "geom"]]


def station_placement_kmeans(X, k, fixed_stations, plot=False):
    x_df = pd.DataFrame(X)
    diff = 1
    cluster = np.zeros(X.shape[0])
    is_not_fixed = np.ones(X.shape[0])
    centroids_not_fixed = x_df.sample(n=k).values
    fixed_centroid_nr = len(fixed_stations)
    while diff:
        centroids = np.concatenate((fixed_stations, centroids_not_fixed), axis=0)
        # for each observation
        tic = time.time()
        for i, row in enumerate(X):
            dists = np.sum((centroids - row) ** 2, axis=1)
            cluster[i] = np.argmin(dists)
            is_not_fixed[i] = cluster[i] >= fixed_centroid_nr
        print("time one iteration of Kmeans:", time.time() - tic)

        fixed_indicator = is_not_fixed.astype(bool)
        new_centroids = x_df[fixed_indicator].groupby(by=cluster[fixed_indicator]).mean().values

        if len(new_centroids) < len(centroids_not_fixed):
            # if some centroids got lost (no population assigned), reinitialize
            init_new = x_df.sample(n = len(centroids_not_fixed) - len(new_centroids)).values
            new_centroids = np.concatenate((new_centroids, init_new), axis=0)
            print("reinitialize stations", len(init_new))

        # if centroids are same then leave
        if np.count_nonzero(centroids_not_fixed - new_centroids) == 0:
            diff = 0
            new_centroids_median = (
                x_df[fixed_indicator].groupby(by=cluster[fixed_indicator]).median().values
            )  # changed to median
            centroids = np.concatenate((fixed_stations, new_centroids_median), axis=0)
        else:
            centroids_not_fixed = new_centroids

        if plot:
            place_new_stations(X, centroids, fixed_stations, save_path=None)

    return centroids, cluster


def plot_new_stations(living_locs, centroids, previous_stations, save_path="../paper_2/new_station_placement.pdf"):
    plt.figure(figsize=(15, 8))
    plt.scatter(living_locs[:, 0], living_locs[:, 1], label="population")
    plt.scatter(centroids[:, 0], centroids[:, 1], label="new stations", c="yellow")
    plt.scatter(previous_stations[:, 0], previous_stations[:, 1], label="original stations")
    plt.axis("off")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def place_new_stations(
    nr_new_stations,
    station_locations,
    population_path="../external_repos/ch-zh-synpop/whole_population/data.statpop.statpop__a67ca3bad3e0f3eb24d038ead1b8d467.p",
    subsample_population=500000,
):
    # load population data
    with open(population_path, "rb") as infile:
        statpop = pickle.load(infile)
    living_locs = np.array(statpop[["home_x", "home_y"]])  # locations of population
    statpop = None

    subsampled_population = living_locs[np.random.permutation(len(living_locs))[:subsample_population]]

    centroids, _ = station_placement_kmeans(subsampled_population, nr_new_stations, station_locations, plot=False)

    plot_new_stations(subsampled_population, centroids, station_locations)

    # assert that the fixed stations remain the same
    assert np.all(centroids[: len(station_locations)] == station_locations)
    # make GDF of new stations
    new_stations_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=centroids[len(station_locations) :, 0], y=centroids[len(station_locations) :, 1])
    )
    return new_stations_gdf


def max_ever_scenario(in_path, out_path=os.path.join("csv", "station_scenario")):
    # Current rationale: maximum capacity of mobility: max no of cars that have been at a station at the same time
    # simply check each month:
    def get_all_dates():
        date_template = "20JJ-MM-02"
        all_dates = []
        for JJ in [19, 20]:
            for MM in range(1, 13):
                if JJ == 2020 and MM > 7:
                    break
                date = date_template.replace("JJ", str(JJ).zfill(2))
                date = date.replace("MM", str(MM).zfill(2))
                all_dates.append(date)
        return all_dates

    def in_date_range(one_station, date):
        beg_before = one_station["bizbeg"] < date
        end_after = one_station["bizend"] > date
        return one_station[beg_before & end_after]

    def get_max_veh(one_station):
        """Find the dates where the maximum was reached and take the list of vehicle IDs on that date"""
        nr_veh = []
        for date in all_dates:
            stations_in_date_range = in_date_range(one_station, date)
            # print(len(test_station[beg_before & end_after]), len(test_station))
            nr_veh.append(len(stations_in_date_range))
        most_veh = np.argmax(nr_veh)
        veh_list = in_date_range(one_station, all_dates[most_veh])["vehicle_no"].unique()
        if len(veh_list) == 0:
            return pd.NA
        return list(veh_list)

    all_dates = get_all_dates()

    # load veh2base and station
    veh2base = pd.read_csv(os.path.join(in_path, "vehicle_to_base.csv"))
    station_df = load_stations(in_path)

    # filter for stations that are in the reservations
    in_reservation = station_df[station_df["in_reservation"]]
    veh2base = veh2base[veh2base["station_no"].isin(in_reservation.index)]

    # get maximum available vehicles per station
    veh_per_station = veh2base.groupby("station_no").apply(get_max_veh).dropna()
    veh_per_station = pd.DataFrame(veh_per_station, columns=["vehicle_list"])

    # merge with geometry
    veh_per_station = veh_per_station.merge(station_df[["geom"]], left_index=True, right_index=True, how="left")

    # ------- Fix duplicates -----
    # compute all occuring vehicle IDs
    nested_lists = [eval(elem) if type(elem) == str else elem for elem in veh_per_station["vehicle_list"].values]
    all_occuring_vehicles = [e for elem in nested_lists for e in elem]
    print("some vehicle IDs appear twice", len(np.unique(all_occuring_vehicles)), len(all_occuring_vehicles))

    # get possible IDs for replacement
    all_vehicle_ids = pd.read_csv(os.path.join(in_path, "vehicle.csv"))["vehicle_no"]
    possible_veh_ids = []
    for veh_no in all_vehicle_ids:
        if veh_no not in all_occuring_vehicles:
            possible_veh_ids.append(veh_no)
        if len(possible_veh_ids) > len(veh_per_station) * 0.8:
            break

    new_id_counter = 0
    vehicle_set = set()
    all_new_lists = []
    for _, row in veh_per_station.iterrows():
        veh_list = row["vehicle_list"]
        new_list = []
        for veh in veh_list:
            if veh in vehicle_set:
                new_list.append(possible_veh_ids[new_id_counter])
                new_id_counter += 1
            else:
                new_list.append(veh)
        all_new_lists.append(new_list)
        vehicle_set.update(veh_list)

    veh_per_station["vehicle_list"] = all_new_lists

    # check it again
    nested_lists = [eval(elem) if type(elem) == str else elem for elem in veh_per_station["vehicle_list"].values]
    all_occuring_vehicles = [e for elem in nested_lists for e in elem]
    assert len(np.unique(all_occuring_vehicles)) == len(all_occuring_vehicles), "still some appearing twice"

    # ------- Save -----

    write_geodataframe(veh_per_station, os.path.join(out_path, "same_stations.csv"))


def fix_duplicate_vehicles_in_existing_scenario(current_folder):
    """This function is not used anywhere, it's just for myself to backup"""

    current_folder = "xgb_2019_sim_prevmode"
    sim_reservations = pd.read_csv(
        f"../../v2g4carsharing/outputs/simulated_car_sharing/{current_folder}/sim_reservations.csv"
    )

    # code to correct datetime
    # date_simulation_2019 = "2019-01-01 00:00:00"
    # sim_reservations = sim_reservations.rename(
    #     columns={"reservationfrom": "reservationfrom_sec", "reservationto": "reservationto_sec"}
    # )
    # sim_reservations["reservationfrom"] = pd.to_datetime(date_simulation_2019) + pd.to_timedelta(
    #     sim_reservations["reservationfrom_sec"], unit="S"
    # )
    # sim_reservations["reservationto"] = pd.to_datetime(date_simulation_2019) + pd.to_timedelta(
    #     sim_reservations["reservationto_sec"], unit="S"
    # )

    print(len(sim_reservations))
    veh_sim_df = sim_reservations[["vehicle_no", "start_station_no"]].drop_duplicates().sort_values("vehicle_no")
    veh_sim_df["new_id"] = pd.NA
    counter = 1
    no_equals = 1
    new_ids_counter = 0
    while no_equals > 0:
        veh_sim_df[f"prev_vehicle_{counter}"] = veh_sim_df["vehicle_no"].shift(counter)
        cond = veh_sim_df[f"prev_vehicle_{counter}"] == veh_sim_df["vehicle_no"]
        veh_sim_df.loc[cond, "new_id"] = possible_veh_ids[new_ids_counter : new_ids_counter + sum(cond)]
        new_ids_counter += sum(cond)
        no_equals = sum(cond)
        print(no_equals)
        counter += 1
    # fill the rest of the new ids
    veh_sim_df["new_id"].fillna(veh_sim_df["vehicle_no"], inplace=True)
    # merge with original data
    sim_reservations_corrected = sim_reservations.merge(
        veh_sim_df[["vehicle_no", "start_station_no", "new_id"]],
        how="left",
        left_on=["vehicle_no", "start_station_no"],
        right_on=["vehicle_no", "start_station_no"],
    )
    sim_reservations_corrected = sim_reservations_corrected.drop("vehicle_no", axis=1).rename(
        columns={"new_id": "vehicle_no"}
    )
    # test result
    test = sim_reservations_corrected.groupby("vehicle_no").agg({"start_station_no": "nunique"})
    assert test.max()[0] == 1
    print(len(sim_reservations))
    sim_reservations.to_csv(
        f"../../v2g4carsharing/outputs/simulated_car_sharing/{current_folder}/sim_reservations.csv", index=False
    )
