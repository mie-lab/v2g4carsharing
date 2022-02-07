import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
import time
import json

from data_io import save_matrix, load_ev_data
from utils import (
    convert_to_datetime, convert_to_timestamp, diff_in_hours, ts_to_index,
    index_to_ts, BASE_DATE
)
from preprocessing import clean_reservations


def create_matrices(ev_reservation, ev_models, time_granularity):
    """
    Main function to produce station and SOC matrices
    Takes a list of reservations as input, as well as ev specifications
    Outputs discrete charging times and states at granularity time_granularity
    """
    # kw - TODO: take into account that several cars may charge at that station
    available_charging_power = 11
    overall_slots = ts_to_index(
        "2020-07-31 23:59:59.999", time_granularity=time_granularity
    )
    num_vehicles = len(ev_reservation["vehicle_no"].unique())
    # mapping of vehicles to the index
    index_to_vehicle = ev_reservation["vehicle_no"].unique()
    vehicle_to_index = {veh: i for i, veh in enumerate(index_to_vehicle)}

    # ------- Output arrays
    station_matrix = np.zeros((overall_slots + 1, num_vehicles))
    # second matrix: arrival and departure SOC:
    soc_matrix = np.zeros((overall_slots + 1, num_vehicles))
    # auxiliary matrix for testing:
    reservation_matrix = np.zeros((overall_slots + 1, num_vehicles))

    # shape of output
    print("Output matrix has shape", station_matrix.shape)

    # ------- Auxiliary arrays
    # track timepoint of following reservation (the index)
    veh_next_booking_time = np.empty(num_vehicles, dtype="<U23")
    # by default always the last slot
    veh_next_booking_time[:] = "2020-07-31 23:59:59.999"
    for_testing = np.zeros(num_vehicles, dtype=int)
    # track SOC necessary for following reservation (in kwh)
    veh_soc_next_booking = np.zeros(num_vehicles)
    start_station_next_booking = np.zeros(num_vehicles) - 1

    # iterate over reservations in reversed order
    ev_reservation.sort_values(by="start_time", ascending=False, inplace=True)

    for res_no, row in ev_reservation.iterrows():
        vehicle_no = row["vehicle_no"]
        start_time = row["start_time"]
        end_time = row["end_time"]
        start_station = row["start_station_no"]
        end_station = row["end_station_no"]

        if start_time >= end_time or res_no == 25657616 or res_no == 24615362:
            # two special caseses with several overlapping bookings
            # cannot be filtered out nicely beforehand
            print(
                res_no,
                "Skip because of other overlapping reservations, and 0 km"
            )
            assert start_station == end_station
            if row["drive_km"] > 0:
                print(row)
            continue

        # Get indices
        veh_index = vehicle_to_index[vehicle_no]
        # get start index of the next booking
        next_booking_time = veh_next_booking_time[veh_index]
        next_booking_index = ts_to_index(
            next_booking_time, time_granularity=time_granularity
        )
        # get index of start and end end of current booking
        end_booking_index = ts_to_index(
            end_time, time_granularity=time_granularity
        )
        start_booking_index = ts_to_index(
            start_time, time_granularity=time_granularity
        )
        if start_booking_index == end_booking_index:
            print(res_no, "Skip because starts and ends in the same time slot")
            assert start_station == end_station
            if row["drive_km"] > 0:
                print(ev_reservation.loc[res_no])
            continue

        # update the next-booking entry for this vehicle
        veh_next_booking_time[veh_index] = start_time

        next_station = start_station_next_booking[veh_index]
        if end_station != next_station and next_station != -1:
            print(res_no, "Station mismatch")

        start_station_next_booking[veh_index] = start_station

        # =================== Matrix 1 : where's the car =====================
        # check: sometimes there are duplicates / overlapping bookings
        if end_booking_index > next_booking_index:
            # if end_booking_index < overall_slots:
            #     print(
            #         res_no, "WARNING: overlapping booking vehicle - continue"
            #     )
            #     print("current reservation")
            #     print(row)
            #     print("next reservation")
            #     print(ev_reservation.loc[for_testing[veh_index]])
            #     continue
            # else:
            assert (
                end_booking_index >= overall_slots
            ), "end booking > start of next booking!"
            end_booking_index = overall_slots + 1

        # fill stations until next booking
        station_matrix[end_booking_index:next_booking_index,
                       veh_index] = end_station

        # =================== Matrix 2 : what's the SOC =====================
        # soc needed for next booking
        needed_soc_next_booking = veh_soc_next_booking[veh_index]  # in kwh
        # get car capacity etc
        brand_model = (row["brand_name"], row["model_name"])
        assert brand_model in ev_models.index, f"model not known {brand_model}"
        battery_power = ev_models.loc[brand_model]["charge_power"]
        battery_capacity = ev_models.loc[brand_model]["battery_capacity"]
        assert (
            not pd.isna(ev_models.loc[brand_model]["range"])
        ), "range is NaN - edit ev_models"
        assert (
            not pd.isna(ev_models.loc[brand_model]["battery_capacity"])
        ), "capacity is NaN - edit ev_models"
        consumption_per_km = battery_capacity / ev_models.loc[brand_model][
            "range"]
        charging_power = min([available_charging_power, battery_power])
        # 1) compute how much needs to be left at arrival, such that the next
        # booking is feasible
        if end_booking_index < overall_slots:
            # time between end of current booking and start of next booking
            # (in hours)
            time_inbetween = diff_in_hours(next_booking_time, end_time)
            if time_inbetween < 0:
                print("current reservation")
                print(row)
                print("next reservation")
                print(ev_reservation.loc[for_testing[veh_index]])
            # The required SOC at arrival is given by the next booking and the
            # charging speed (but 0 if there is a full charging period inbetw.)
            needed_soc_arrival = max(
                [needed_soc_next_booking - charging_power * time_inbetween, 0]
            )
        else:
            needed_soc_arrival = 0

        for_testing[veh_index] = res_no

        # 2) Compute with drive_km what SOC is necessary at departure
        needed_soc_trip = consumption_per_km * row["drive_km"]  # usage in kwh
        needed_soc_departure = needed_soc_trip + needed_soc_arrival

        soc_matrix[
            start_booking_index,
            veh_index] = min(needed_soc_departure / battery_capacity, 1)
        if (
            end_booking_index <= overall_slots
            and end_booking_index > start_booking_index
        ):
            soc_matrix[
                end_booking_index - 1,
                veh_index] = min(needed_soc_arrival / battery_capacity, 1)
        # update next-booking soc
        veh_soc_next_booking[veh_index] = needed_soc_departure

        # ========== Add reservation to reservation matrix
        reservation_matrix[start_booking_index:end_booking_index,
                           veh_index] = res_no

    # postprocessing: add -1 for the start because we don't know whether the
    # car was already part of the fleet at that point
    for veh in range(station_matrix.shape[1]):
        first_start_ind = ts_to_index(
            veh_next_booking_time[veh], time_granularity=time_granularity
        )
        station_matrix[:first_start_ind, veh] = -1

    return station_matrix, soc_matrix, reservation_matrix, index_to_vehicle


if __name__ == "__main__":
    out_path = "outputs"
    time_granularity = 0.5  # in reference to one hour, e.g. 0.5 = half an hour

    # Load data
    ev_reservation, ev_models = load_ev_data("postgis")

    # Run
    (station_matrix, soc_matrix, reservation_matrix, index_to_vehicle
     ) = create_matrices(ev_reservation, ev_models, time_granularity)

    # save in csv files
    columns = [
        index_to_ts(
            index, time_granularity=time_granularity, base_date=BASE_DATE
        ) for index in range(len(soc_matrix))
    ]
    for matrix, name in zip(
        [station_matrix, soc_matrix, reservation_matrix],
        ["station_matrix", "soc_matrix", "reservation_matrix"]
    ):
        save_matrix(
            matrix,
            columns=columns,
            index=index_to_vehicle,
            name=name,
            out_path=out_path
        )
