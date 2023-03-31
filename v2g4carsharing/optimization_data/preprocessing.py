from typing import Type
import pandas as pd
import numpy as np
import datetime
import time
import os


def get_corrected_end(row):
    # only the reservation ending can be changed in this setting
    if pd.isna(row["next_reservationfrom"]):
        return row["reservationto"]
    # check if overlapping at all and if the next drive was even a drive
    if (row["reservationto"] > row["next_reservationfrom"]) or (
        row["reservationto"] > row["next_drive_firststart"]
    ):
        return row["next_drive_firststart"]
    return row["reservationto"]


def get_corrected_start(row):
    # only the reservation ending can be changed in this setting
    # check if overlapping at all
    if (
        not pd.isna(row["prev_end_time"])
        and row["prev_end_time"] != row["prev_reservationto"]
    ):
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

    if (
        len(df[df["next_reservationfrom"] < df["reservationto"]]) > 0
        or len(df[df["reservationfrom"] < df["prev_reservationto"]]) > 0
    ):
        global bad_counter
        bad_counter += 1
        # print(df["reservationfrom"].values[1], df["vehicle_no"].unique())

    # first change the end time
    df["end_time"] = df.apply(get_corrected_end, axis=1)
    # then, we have to change the start time according to the end time before!
    df["prev_end_time"] = df["end_time"].shift(1)
    # compute start time
    df["start_time"] = df.apply(get_corrected_start, axis=1)

    return df


# test
# ts_to_index("2019-05-02 15:21:14.000", time_granularity=time_granularity)
def deprecated_clean_reservations(ev_reservation):
    """
    Reservations have overlaps etc.
    Cleaned up here, TODO: move to preprocessing script
    """
    # assert no free floating
    if "tripmode" in ev_reservation.columns:
        assert all(
            ev_reservation["tripmode"]
            != "FreeFloating (Rückgabe an einem beliebigen Ort)"
        )
    # remove the ones with drivekm =0
    print("Number of ev reservaions before cleaning", len(ev_reservation))
    ev_reservation.reset_index(inplace=True)
    ev_reservation = ev_reservation[ev_reservation["drive_km"] > 0]
    print("length after drive_km>0", len(ev_reservation))

    # get new start and end time
    ev_preprocessed = ev_reservation.groupby("vehicle_no").apply(
        get_corrected_per_veh
    )
    print("length after cleaning", len(ev_preprocessed))
    print(
        "Problems with end time after start of next",
        bad_counter,
        "out of",
        len(ev_preprocessed),
    )

    # remove the ones that are wrong after cleaning
    ev_preprocessed = ev_preprocessed[
        ~(ev_preprocessed["start_time"] > ev_preprocessed["end_time"])
    ]
    print("length after removal of start > end", len(ev_preprocessed))
    ev_preprocessed = ev_preprocessed[
        ~(ev_preprocessed["start_time"] < ev_preprocessed["prev_end_time"])
    ]
    print("length after removal of start < prev end", len(ev_preprocessed))

    # clean up
    ev_preprocessed = ev_preprocessed.rename(
        columns={"vehicle_no": "vehicle_no_orig"}
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
        inplace=True,
    )
    ev_preprocessed.drop_duplicates(
        subset=["vehicle_no", "person_no", "start_time", "end_time"],
        inplace=True,
    )
    print("length after removing duplicates", len(ev_preprocessed))
    # convert start and end time into datetimes
    ev_preprocessed["start_time"] = pd.to_datetime(
        ev_preprocessed["start_time"]
    )
    ev_preprocessed["end_time"] = pd.to_datetime(ev_preprocessed["end_time"])
    return ev_preprocessed


def get_problematic_rows(wo_nans):
    """Get indices where two bookings of the same vehicle overlap"""
    # sort by vehicle and start time
    wo_nans.sort_values(["vehicle_no", "start_time"], inplace=True)
    # shift vehicles and end times
    wo_nans["prev_vehicle_no"] = wo_nans["vehicle_no"].shift(1)
    wo_nans["prev_end_time"] = wo_nans["end_time"].shift(1)
    problematic_rows = wo_nans[
        (wo_nans["prev_vehicle_no"] == wo_nans["vehicle_no"])
        & (wo_nans["prev_end_time"] > wo_nans["start_time"])
    ].index
    return problematic_rows


def reservation_start_end_time(res, only_drive=False):
    # convert columns to datetime
    for col in [
        "reservationfrom",
        "reservationto",
        "drive_firststart",
        "drive_lastend",
    ]:
        res[col] = pd.to_datetime(res[col])

    # remove nans and for now also the ones that have zero drive_km
    wo_nans = res[
        ~pd.isna(res["drive_lastend"]) & (res["drive_km"] > 0)
    ].reset_index()

    # initial start and end time: reservation from and to
    if only_drive:
        wo_nans["start_time"] = wo_nans["drive_firststart"]
    else:
        wo_nans["start_time"] = wo_nans[
            ["reservationfrom", "drive_firststart"]
        ].min(axis=1)
    wo_nans["end_time"] = wo_nans["drive_lastend"]

    # get the problematic cases
    for i in range(5):
        problematic = get_problematic_rows(wo_nans)
        print("Number of current problematic cases", len(problematic))
        if len(problematic) == 0:
            break
        # fix the problematic ones --> set them to the previous drive_lastend
        wo_nans.loc[problematic, "start_time"] = wo_nans.loc[
            problematic, "prev_end_time"
        ]
        # check that they are fixed
    new_problematic = get_problematic_rows(wo_nans)
    print("Number of problems after fix:", len(new_problematic))

    # remove cases where our modification led to the case where end <= start
    print(
        "removing cases where start_time == end time",
        sum(wo_nans["start_time"] == wo_nans["end_time"]),
    )
    print(
        "removing cases where start_time > end time",
        sum(wo_nans["start_time"] > wo_nans["end_time"]),
    )
    wo_nans = wo_nans[wo_nans["start_time"] < wo_nans["end_time"]]

    if only_drive:
        #         # remove drives that are less than 3 minute long
        #         less_than_three_min = (wo_nans["end_time"] - wo_nans["start_time"]).dt.total_seconds() < 3 * 60
        #         wo_nans = wo_nans[~less_than_three_min]
        #         print("removing drives that are less than 3 min long", sum(less_than_three_min))
        return wo_nans.drop(
            ["prev_vehicle_no", "prev_end_time", "index"],
            axis=1,
            errors="ignore",
        )

    # Finally: add the ones with drive_km=0
    zero_drive_km = res[res["drive_km"] == 0]
    zero_drive_km["start_time"] = zero_drive_km["reservationfrom"]
    zero_drive_km["end_time"] = zero_drive_km["reservationto"]
    # concat them and initialize that none of them is removed
    with_drive_km = pd.concat([wo_nans, zero_drive_km])
    with_drive_km["remove"] = 0
    # Iterative remove rows (only the 0-km ones) until there are no problems
    for i in range(20):
        with_drive_km.sort_values(["vehicle_no", "start_time"], inplace=True)
        with_drive_km["prev_end_time"] = with_drive_km["end_time"].shift(1)
        with_drive_km["prev_vehicle_no"] = with_drive_km["vehicle_no"].shift(1)
        with_drive_km["next_start_time"] = with_drive_km["start_time"].shift(-1)
        with_drive_km["next_vehicle_no"] = with_drive_km["vehicle_no"].shift(-1)
        # remove all with drive_km == 0 that have overlaps
        with_drive_km.loc[
            (
                # overlaps at start time and 0km
                (with_drive_km["prev_end_time"] > with_drive_km["start_time"])
                & (with_drive_km["drive_km"] == 0)
                & (
                    with_drive_km["prev_vehicle_no"]
                    == with_drive_km["vehicle_no"]
                )
            )
            | (
                # overlaps at end time and 0km
                (with_drive_km["next_start_time"] < with_drive_km["end_time"])
                & (with_drive_km["drive_km"] == 0)
                & (
                    with_drive_km["next_vehicle_no"]
                    == with_drive_km["vehicle_no"]
                )
            ),
            "remove",
        ] = 1
        # remove rows
        print(
            "Deleting xx rows with drive_km=0 because of overlaps",
            sum(with_drive_km["remove"]),
        )
        with_drive_km = with_drive_km[with_drive_km["remove"] == 0]

        new_problematic = get_problematic_rows(with_drive_km)
        print(len(new_problematic))
        if len(new_problematic) == 0:
            break

    new_problematic = get_problematic_rows(with_drive_km)
    print(
        "Number of problems after addding the ones with drive_km=0:",
        len(new_problematic),
    )

    print(
        "Kept ones out of all zero-km ones:",
        sum(with_drive_km["drive_km"] == 0) / len(zero_drive_km),
    )

    clean_res = with_drive_km.drop(
        [
            "index",
            "next_start_time",
            "prev_end_time",
            "prev_vehicle_no",
            "next_vehicle_no",
            "remove",
        ],
        axis=1,
    )
    clean_res["duration_planned_min"] = (
        clean_res["reservationto"] - clean_res["reservationfrom"]
    ).dt.total_seconds() / 60
    clean_res["duration_actual_min"] = (
        clean_res["end_time"] - clean_res["start_time"]
    ).dt.total_seconds() / 60
    return clean_res


def clean_reservations(res, relocations):
    # clean up the reservations
    clean_res = reservation_start_end_time(res, only_drive=True)
    # clean relocations
    relocations.reset_index(inplace=True)
    relocations["reservation_no"] = relocations["relocation_no"] + 30000000
    relocations["start_time"] = pd.to_datetime(relocations["start_time"])
    relocations["end_time"] = pd.to_datetime(relocations["end_time"])

    # concat reservations and relocations
    together = (
        pd.concat([clean_res, relocations])
        .drop(["relocation_no"], axis=1)
        .set_index("reservation_no")
    )

    # iteratively remove problems with relocations that happen just in the
    # middle of a resevration
    for i in range(5):
        together.sort_values(["vehicle_no", "start_time"], inplace=True)
        together["next_start_station_no"] = together["start_station_no"].shift(
            -1
        )
        together["next_end_station_no"] = together["end_station_no"].shift(-1)
        together["next_start_time"] = together["start_time"].shift(-1)
        together["next_vehicle_no"] = together["vehicle_no"].shift(-1)

        problem = (together["next_vehicle_no"] == together["vehicle_no"]) & (
            together["end_time"] > together["next_start_time"]
        )
        print(sum(problem))
        if sum(problem) == 0:
            break
        # three cases:
        # 1) start and end station no are both the same --> no need ot do anything
        # 2) both are different --> we don't do anything, no clue what's going on
        # 3) start is the same, but end is different -> replace end with next_end
        cond = problem & (
            together["start_station_no"] == together["next_start_station_no"]
        )
        together.loc[cond, "end_station_no"] = together.loc[
            cond, "next_end_station_no"
        ]
        # 4) end is the same, but start is different -> replace start with next_start
        cond = problem & (
            together["end_station_no"] == together["next_end_station_no"]
        )
        together.loc[cond, "start_station_no"] = together.loc[
            cond, "next_start_station_no"
        ]

        # remove the problems
        together["prev_vehicle_no"] = together["vehicle_no"].shift(1)
        together["prev_end_time"] = together["end_time"].shift(1)

        problem_prev = (
            together["prev_vehicle_no"] == together["vehicle_no"]
        ) & (together["prev_end_time"] > together["start_time"])
        together = together[~problem_prev]
        print("Removed relocations that overlap", sum(problem_prev))

    return together


if __name__ == "__main__":
    # CREATE CLEAN RESERVATIONS
    res = pd.read_csv(os.path.join("data", "reservation.csv"))
    clean_res = reservation_start_end_time(res)
    problem = get_problematic_rows(clean_res)
    assert len(problem) == 0
    clean_res.drop(["prev_vehicle_no", "prev_end_time"], axis=1).set_index(
        "reservation_no"
    ).to_csv(os.path.join("data", "clean_reservation.csv"))
