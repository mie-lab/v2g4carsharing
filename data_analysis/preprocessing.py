import pandas as pd
import numpy as np
import datetime
import time


def get_corrected_end(row):
    # only the reservation ending can be changed in this setting
    if pd.isna(row["next_reservationfrom"]):
        return row["reservationto"]
    # check if overlapping at all and if the next drive was even a drive
    if (
        row["reservationto"] > row["next_reservationfrom"]
        and row["next_drive_km"] > 0
    ):
        return row["next_drive_firststart"]
    elif (
        row["reservationto"] > row["next_reservationfrom"]
        and row["next_drive_km"] == 0
    ):
        return row["drive_lastend"]
    else:
        return row["reservationto"]


def get_corrected_start(row):
    # only the reservation ending can be changed in this setting
    # check if overlapping at all
    if not pd.isna(row["prev_end_time"]
                   ) and row["prev_end_time"] != row["prev_reservationto"]:
        return row["prev_end_time"]
    elif (
        not pd.isna(row["prev_reservationto"])
        and row["reservationfrom"] < row["prev_reservationto"]
        and row["drive_km"] == 0
    ):
        return row["prev_reservationto"]

    return row["reservationfrom"]


def get_corrected_per_veh(df):
    # sort by time
    df.sort_values("reservationfrom", inplace=True)
    # Add prev and next booking
    df["next_reservation_no"] = df["reservation_no"].shift(-1)
    df["next_reservationfrom"] = df["reservationfrom"].shift(-1)
    df["next_drive_km"] = df["drive_km"].shift(-1)
    df["next_drive_firststart"] = df["drive_firststart"].shift(-1)
    # get previous reservation
    df["prev_reservation_no"] = df["reservation_no"].shift(1)
    df["prev_reservationto"] = df["reservationto"].shift(1)
    df["prev_drive_km"] = df["drive_km"].shift(1)
    df["prev_drive_lastend"] = df["drive_lastend"].shift(1)

    # first change the end time
    df["end_time"] = df.apply(get_corrected_end, axis=1)
    # then, we have to change the start time according to the end time before!
    df["prev_end_time"] = df["end_time"].shift(1)
    df["start_time"] = df.apply(get_corrected_start, axis=1)

    # last preprocessing step: sort again and remove the 0km ones that cause
    # trouble
    df.sort_values("start_time", inplace=True)
    df["next_start_time"] = df["start_time"].shift(-1)
    df["prev_end_time"] = df["end_time"].shift(1)
    df = df[(df["next_start_time"] >= df["end_time"]) | (df["drive_km"] > 0)]
    df = df[(df["prev_end_time"] <= df["start_time"]) | (df["drive_km"] > 0)]

    return df


# test
# ts_to_index("2019-05-02 15:21:14.000", time_granularity=time_granularity)
def clean_reservations(ev_reservation):
    """
    Reservations have overlaps etc.
    Cleaned up here, TODO: move to preprocessing script
    """
    print("length before ff removal", len(ev_reservation))
    wo_ff = ev_reservation[ev_reservation["tripmode"] !=
                           "FreeFloating (Rückgabe an einem beliebigen Ort)"
                           ].reset_index()
    print("length before cleaning", len(wo_ff))
    wo_ff_new = wo_ff.groupby("vehicle_no").apply(get_corrected_per_veh)
    print("length after cleaning", len(wo_ff_new))
    wo_ff_new = wo_ff_new.rename(columns={
        "vehicle_no": "vehicle_no_orig"
    }).reset_index()
    assert all(wo_ff_new["vehicle_no"] == wo_ff_new["vehicle_no_orig"])
    wo_ff_new = wo_ff_new.drop(columns=["vehicle_no_orig", "level_1"]
                               ).set_index("reservation_no")

    # drop duplicates
    wo_ff_new.drop(
        columns=[
            "prev_reservationto",
            "prev_drive_km",
            "prev_drive_lastend",
            "next_drive_firststart",
            "next_drive_km",
            "next_reservationfrom",
        ],
        inplace=True
    )
    wo_ff_new.drop_duplicates(
        subset=["vehicle_no", "person_no", "start_time", "end_time"],
        inplace=True
    )
    print("length after removing duplicates", len(wo_ff_new))
    # clean df
    return wo_ff_new
