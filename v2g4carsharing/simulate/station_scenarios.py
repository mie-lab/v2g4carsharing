import os
import pandas as pd
import numpy as np

from v2g4carsharing.simulate.car_sharing_patterns import load_stations
from v2g4carsharing.import_data.import_utils import write_geodataframe


import warnings


def station_scenario(
    station_scenario="all_stations", vehicle_scenario=5000, in_path="data/", sim_date="2020-01-01 00:00:00",
):
    # current scenario: only 1500 stations, as claimed on the website
    # all_stations scenario: 1916 stations (maximum that appear in the reservations)
    # new stations scenario: XX_new_stations places XX further stations somehwere
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
    # .str.contains("(0.00", regex=False)] # old versio
    print("after removing some with invalid geometry:", len(station_veh_list))

    # if we want a realistic current situation, we drop the vehicles that have NaNs
    if station_scenario == "current_scenario":
        station_veh_list = station_veh_list.dropna()

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
    assert len(uni_vals) == desired_veh_per_station.sum()

    print(f"Final scenario has {len(station_veh_list)} stations and {len(uni_vals)} vehicles")

    return station_veh_list[["vehicle_list", "geom"]]


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
