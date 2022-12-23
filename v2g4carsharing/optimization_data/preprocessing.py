from typing import Type
import pandas as pd
import numpy as np
import datetime
import time


def get_corrected_end(row):
    # only the reservation ending can be changed in this setting
    if pd.isna(row["next_reservationfrom"]):
        return row["reservationto"]
    # check if overlapping at all and if the next drive was even a drive
    if (row["reservationto"] > row["next_reservationfrom"]
        ) or (row["reservationto"] > row["next_drive_firststart"]):
        return row["next_drive_firststart"]
    return row["reservationto"]


def get_corrected_start(row):
    # only the reservation ending can be changed in this setting
    # check if overlapping at all
    if not pd.isna(row["prev_end_time"]
                   ) and row["prev_end_time"] != row["prev_reservationto"]:
        return row["prev_end_time"]
    # here we could check if drive_firststart is before reservationfrom, but
    # then station mismatch because of 0-minute-relocations
    return row["reservationfrom"]

bad_counter = 0
def get_corrected_per_veh(df):
    # sort by time
    df.sort_values("drive_firststart", inplace=True)
    # Add prev and next booking
    df["next_reservation_no"] = df["reservation_no"].shift(-1)
    df["next_reservationfrom"] = df["reservationfrom"].shift(-1)
    df["next_drive_firststart"] = df["drive_firststart"].shift(-1)
    # get previous reservation
    df["prev_reservation_no"] = df["reservation_no"].shift(1)
    df["prev_reservationto"] = df["reservationto"].shift(1)

    if len(df[df["next_reservationfrom"] < df["reservationto"]]) > 0 or len(df[df["reservationfrom"] < df["prev_reservationto"]]) > 0:
        global bad_counter 
        bad_counter += 1
        print(df["reservationfrom"].values[1], df["vehicle_no"].unique())

    # first change the end time
    df["end_time"] = df.apply(get_corrected_end, axis=1)
    # then, we have to change the start time according to the end time before!
    df["prev_end_time"] = df["end_time"].shift(1)
    # compute start time
    df["start_time"] = df.apply(get_corrected_start, axis=1)

    return df


# test
# ts_to_index("2019-05-02 15:21:14.000", time_granularity=time_granularity)
def clean_reservations(ev_reservation):
    """
    Reservations have overlaps etc.
    Cleaned up here, TODO: move to preprocessing script
    """
    # assert no free floating
    if "tripmode" in ev_reservation.columns:
        assert all(
            ev_reservation["tripmode"] !=
            "FreeFloating (Rückgabe an einem beliebigen Ort)"
        )
    # remove the ones with drivekm =0
    print("Number of ev reservaions before cleaning", len(ev_reservation))
    ev_reservation.reset_index(inplace=True)
    ev_reservation = ev_reservation[ev_reservation["drive_km"] > 0]
    print("length after drive_km>0", len(ev_reservation))

    # get new start and end time
    ev_preprocessed = ev_reservation.groupby("vehicle_no"
                                             ).apply(get_corrected_per_veh)
    print("length after cleaning", len(ev_preprocessed))
    print("Problems with end time after start of next", bad_counter, "out of", len(ev_preprocessed))

    # remove the ones that are wrong after cleaning
    ev_preprocessed = ev_preprocessed[
        ~(ev_preprocessed["start_time"] > ev_preprocessed["end_time"])]
    print("length after removal of start > end", len(ev_preprocessed))
    ev_preprocessed = ev_preprocessed[
        ~(ev_preprocessed["start_time"] < ev_preprocessed["prev_end_time"])]
    print("length after removal of start < prev end", len(ev_preprocessed))

    # clean up
    ev_preprocessed = ev_preprocessed.rename(
        columns={
            "vehicle_no": "vehicle_no_orig"
        }
    ).reset_index()
    assert all(
        ev_preprocessed["vehicle_no"] == ev_preprocessed["vehicle_no_orig"]
    )
    ev_preprocessed = ev_preprocessed.drop(
        columns=["vehicle_no_orig", "level_1"]
    ).set_index("reservation_no")

    # drop columns and duplicates
    ev_preprocessed.drop(
        columns=[
            "prev_reservationto",
            "next_drive_firststart",
            "next_reservationfrom",
        ],
        inplace=True
    )
    ev_preprocessed.drop_duplicates(
        subset=["vehicle_no", "person_no", "start_time", "end_time"],
        inplace=True
    )
    print("length after removing duplicates", len(ev_preprocessed))
    # convert start and end time into datetimes
    ev_preprocessed["start_time"] = pd.to_datetime(
        ev_preprocessed["start_time"]
    )
    ev_preprocessed["end_time"] = pd.to_datetime(ev_preprocessed["end_time"])
    return ev_preprocessed
