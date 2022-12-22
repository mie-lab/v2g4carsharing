import os
import numpy as np
import pandas as pd

from utils import (ts_to_index, BASE_DATE, FINAL_DATE)
FINAL_DATE_STR = FINAL_DATE.strftime("%Y-%m-%d %H:%M:%S.%f")


def get_veh_per_station(station_matrix):
    """
    Get a dataframe with the number of vehicles per station at any time
    """

    def get_num_by_station(one_column):
        actual_stations = one_column[one_column > 0].values
        stations_in_column, vehicle_count = np.unique(
            actual_stations, return_counts=True
        )
        station_counts = pd.Series(vehicle_count, index=stations_in_column)
        return station_counts

    station_veh_count = station_matrix.apply(lambda x: get_num_by_station(x))
    return station_veh_count


def compute_avail_use_vehicle(
    vehicle_df,
    time_granularity=0.5,
    overall_slots=577,
    start_time_col="reservationfrom",
    end_time_col="reservationto"
):
    """
    Compute availability or usage for one vehicle
    Creating a matrix of discrete timesteps with granularity 'time_granularity'
    and therefore 'overall_slots' columns
    """
    veh_total_available_hours = np.zeros(overall_slots + 1)

    for i, row in vehicle_df.iterrows():
        start_time = max([pd.to_datetime(row[start_time_col]), BASE_DATE])

        end_time = FINAL_DATE if row[end_time_col
                                     ] > FINAL_DATE_STR else pd.to_datetime(
                                         row[end_time_col]
                                     )
        start_ind = ts_to_index(start_time, time_granularity=time_granularity)
        end_ind = ts_to_index(end_time, time_granularity=time_granularity)
        #         print(start_time, " to ", end_time)
        #         print(start_ind, end_ind)
        veh_total_available_hours[start_ind:end_ind] += 1

        # compute total hours
        total_hours_since_start = (start_time -
                                   BASE_DATE).total_seconds() / (60 * 60)
        subtract_from_start_ind = (
            total_hours_since_start % time_granularity
        ) / time_granularity
        veh_total_available_hours[start_ind] -= subtract_from_start_ind

        total_hours_since_start = (end_time -
                                   BASE_DATE).total_seconds() / (60 * 60)
        add_to_end_ind = (
            total_hours_since_start % time_granularity
        ) / time_granularity
        veh_total_available_hours[end_ind] += add_to_end_ind
    return veh_total_available_hours


def compute_availability(data_path, time_granularity=24):
    vehicle = pd.read_csv(
        os.path.join(data_path, "vehicle.csv"), index_col="vehicle_no"
    )
    v2b = pd.read_csv(os.path.join(data_path, "vehicle_to_base.csv"))
    overall_slots = ts_to_index(FINAL_DATE, time_granularity=time_granularity)

    # 1) get availability
    total_available_hours_ice = np.zeros(overall_slots + 1)
    total_available_hours_ev = np.zeros(overall_slots + 1)

    for veh_id, vehicle_df in v2b.groupby("vehicle_no"):
        veh_cat = vehicle.loc[veh_id]["energytypegroup"]

        # collect vehicle-wise availability hours
        veh_total_available_hours = compute_avail_use_vehicle(
            vehicle_df,
            time_granularity=time_granularity,
            overall_slots=overall_slots,
            start_time_col="bizbeg",
            end_time_col="bizend"
        )
        if veh_cat == "Electro":
            total_available_hours_ev += veh_total_available_hours
        else:
            total_available_hours_ice += veh_total_available_hours
    print("Done availability")
    return total_available_hours_ice, total_available_hours_ev


def compute_usage(data_path, time_granularity=24):
    reservation = pd.read_csv(
        os.path.join(data_path, "reservation.csv"), index_col="reservation_no"
    )
    overall_slots = ts_to_index(FINAL_DATE, time_granularity=time_granularity)

    total_used_hours_ev = np.zeros(overall_slots + 1)
    total_used_hours_ice = np.zeros(overall_slots + 1)

    print("overall vehicles", len(reservation["vehicle_no"].unique()))
    veh_counter = 0
    for _, vehicle_df in reservation.groupby("vehicle_no"):
        veh_counter += 1
        veh_cat = vehicle_df["energytypegroup"].unique()[0]

        veh_total_used_hours = compute_avail_use_vehicle(
            vehicle_df,
            time_granularity=time_granularity,
            overall_slots=overall_slots
        )
        total_used_hours = np.clip(veh_total_used_hours, 0, 1)
        if veh_cat == "Electro":
            total_used_hours_ev += total_used_hours
        else:
            total_used_hours_ice += total_used_hours
    print("Done usage")
    return (total_used_hours_ice, total_used_hours_ev)


def prepare_scenario_1(base_path="."):
    # Load basics
    vehicle = pd.read_csv(os.path.join(base_path, "data", "vehicle.csv"), index_col="vehicle_no")
    # Load scenario
    scenario = pd.read_csv(
        os.path.join(base_path, "csv", "scenario_1_models.csv")
    )
    # # V1: Only take the vehicles that appear in the reservations
    # reservation = pd.read_csv(
    #     os.path.join(base_path, "data", "reservation.csv")
    # )
    # # merge vehicle information
    # res_veh = reservation.merge(
    #     vehicle, how="left", left_on="vehicle_no", right_on="vehicle_no"
    # )
    # veh_cat = res_veh.groupby("vehicle_no").agg(
    #     {
    #         "vehicle_category": "first",
    #         "energytypegroup_x": "first"
    #     }
    # )
    # # filter for non EVs
    # veh_cat_ev = vehicle[vehicle["energytypegroup"] != "Electro"].drop(
    #     ["brand_name", "model_name", "energytype"], axis=1
    # )
    veh_cat_ev = vehicle.drop(
        ["brand_name", "model_name", "energytype"], axis=1
    )
    # merge with scneario
    new_model_by_veh = veh_cat_ev.reset_index().merge(
        scenario, left_on="vehicle_category", right_on="vehicle_category"
    )
    new_model_by_veh.set_index("vehicle_no").drop(
        "vehicle_category", axis=1
    ).to_csv(os.path.join(base_path, "csv", "scenario_1.csv"))


if __name__ == "__main__":
    prepare_scenario_1()