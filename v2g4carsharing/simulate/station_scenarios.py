import os
import pandas as pd
import numpy as np

from v2g4carsharing.simulate.car_sharing_patterns import load_stations
from v2g4carsharing.import_data.import_utils import write_geodataframe


def simple_vehicle_station_scenario(in_path, out_path=os.path.join("csv", "station_scenario")):
    # Current rationale: maximum capacity of mobility: max no of cars that have been at a station at the same time
    # TODO: this should be moved into new file. it is only here for backup
    # TODO: CAREFUL: the vehicle IDs can appear in the vehicle_list for multiple stations! tow vehicles can therefore
    # be in use at the same time. this must be prevented in future scenarios.
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
    veh_per_station = station_df[["geom"]].merge(veh_per_station, left_index=True, right_index=True, how="right")

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
