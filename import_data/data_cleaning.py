import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import json

from import_utils import (
    lon_lat_to_geom, write_geodataframe, read_geodataframe,
    convert_to_timestamp
)
from preprocess_relocations import v2b_to_relocations


def preprocess_user(source_path, plot_path=None):
    data_user = pd.read_csv(
        os.path.join(source_path, "20211213_ethz_person.tsv"), sep="\t"
    )
    print("Number users", len(data_user))
    # add geometry
    geo_frame = lon_lat_to_geom(data_user)
    # add agegroup as an integer
    age_group = data_user["AGEGROUP"].values
    age_group = age_group[~pd.isna(age_group)]
    sorted_groups = np.sort(np.unique(age_group))
    map_dict = {g: i for i, g in enumerate(sorted_groups)}
    geo_frame["AGEGROUP_int"] = geo_frame["AGEGROUP"].apply(
        lambda x: map_dict.get(x, pd.NA)
    )
    # set index
    assert len(np.unique(geo_frame["PERSON_NO"])) == len(data_user)
    geo_frame.set_index("PERSON_NO", inplace=True)

    # plotting
    if plot_path is not None:
        for col in ["AGEGROUP", "GENDER", "LANGUAGE", "ABOGROUP"]:
            sns.countplot(data=data_user, x=col)
            if len(np.unique(data_user[col].dropna())) > 4:
                plt.xticks(rotation=90)
            plt.savefig(os.path.join(plot_path, "user_" + col + ".png"))

    return geo_frame


def preprocess_vehicle(source_path, plot_path=None):
    data_cars = pd.read_csv(
        os.path.join(source_path, "20211213_ethz_vehicle.tsv"), sep="\t"
    )
    data_cars_cleaned = data_cars.drop_duplicates().set_index("VEHICLE_NO")
    print(
        "vehicles in original file", len(data_cars), "unique",
        len(data_cars_cleaned), len(np.unique(data_cars["VEHICLE_NO"]))
    )

    if plot_path is not None:
        for col in [
            "VEHICLE_CATEGORY", "BRAND_NAME", "ENERGYTYPEGROUP", "ENERGYTYPE"
        ]:
            sns.countplot(data=data_cars_cleaned, x=col)
            if len(np.unique(data_cars_cleaned[col].dropna())) > 4:
                plt.xticks(rotation=90)
            plt.savefig(os.path.join(plot_path, "vehicle_" + col + ".png"))
    return data_cars_cleaned


def preprocess_reservation(source_path):
    booking_path = os.path.join(source_path, "20211213_ethz_reservation")
    all_bookings = []
    for booking_csv in sorted(os.listdir(booking_path)):
        next_csv = pd.read_csv(
            os.path.join(booking_path, booking_csv), sep="\t"
        )
        all_bookings.append(next_csv)
    all_bookings = pd.concat(all_bookings)
    data_booking = all_bookings.drop_duplicates().set_index("RESERVATION_NO")
    data_booking.drop(
        columns=[
            "VEHICLE_CATEGORY", "ENERGYTYPE", 'BASESTART_NAME',
            'BASESTART_LAT', 'BASESTART_LON', 'BASEEND_NAME', 'BASEEND_LAT',
            'BASEEND_LON'
        ],
        inplace=True
    )
    return data_booking


def preprocess_station(source_path):
    data_station = pd.read_excel(
        os.path.join(source_path, "20220204_eth_base.xlsx")
    )
    assert len(data_station) == len(data_station["BASE_NO"].unique())
    data_station.rename(
        columns={
            "BASE_NO": "STATION_NO",
            "BASE_NAME": "NAME"
        }, inplace=True
    )
    data_station.set_index("STATION_NO", inplace=True)
    data_station = lon_lat_to_geom(data_station)
    return data_station


def preprocess_v2b(source_path):
    path = os.path.join(source_path, "20220204_eth_vehicle_to_base.xlsx")
    v2b = pd.read_excel(path)
    BASE_DATE = pd.to_datetime('2019-01-01 00:00:00.000')
    FINAL_DATE = pd.to_datetime("2020-07-31 23:59:59.999")
    v2b_in_range = v2b[(v2b["BIZBEG"] <= FINAL_DATE)
                       & (v2b["BIZEND"] > BASE_DATE)]
    v2b_in_range.index.name = "v2b_no"
    return v2b_in_range


def split_reservation(out_path):
    data_booking = pd.read_csv(
        os.path.join(out_path, "all_reservation.csv"),
        index_col="reservation_no"
    )
    # 1) service reservations
    service_reservation = data_booking[
        data_booking["reservationtype"] != "Normal"]
    service_reservation.to_csv(
        os.path.join(out_path, "service_reservation.csv"),
        index="reservation_no"
    )
    # reduce booking data to the rest
    data_booking = data_booking[data_booking["reservationtype"] == "Normal"]

    # 2) Cancelled bookings
    # There are rows where there is a canceldate, but reservationstate is
    # not "annulliert" and everything else looks normal
    # only 44 rows, so we just ignore that and delete it from the data
    cond_cancelled = ~pd.isna(data_booking["canceldate"]) | (
        data_booking["reservationstate"] == "sofortige Rückgabe"
    ) | (data_booking["reservationstate"] == "annulliert")
    canceled_bookings = data_booking[cond_cancelled]
    canceled_bookings.to_csv(
        os.path.join(out_path, "cancelled_reservation.csv"),
        index="reservation_no"
    )
    # reduce to rest
    data_booking = data_booking[~cond_cancelled]

    # 3) TODO: outliers ( only bookings that are too long etc)
    # --> currently not filtered out in reservation.csv
    # Open questions: bookings that are too short? free floating?
    # Outliers are the ones that start much earlier
    # TODO: outliers are the ones with more than 7 days of booking (168h)
    #  - delete or not??
    # why not: because they might be relevant
    cond_outlier = data_booking["reservationfrom"] < "2019"
    # data_booking["duration_hours"] > 168
    outlier_bookings = data_booking[cond_outlier]
    outlier_bookings.to_csv(
        os.path.join(out_path, "outlier_reservation.csv"),
        index="reservation_no"
    )
    # reduce to rest
    data_booking = data_booking[~cond_outlier]

    # 4) Remove free floating
    cond_ff = (
        data_booking["tripmode"] ==
        "FreeFloating (Rückgabe an einem beliebigen Ort)"
    )
    free_floating = data_booking[cond_ff]
    free_floating.to_csv(
        os.path.join(out_path, "free_floating_reservation.csv"),
        index="reservation_no"
    )
    # reduce to rest
    data_booking = data_booking[~cond_ff]

    # 5) save the leftover part
    data_booking.to_csv(
        os.path.join(out_path, "reservation.csv"), index="reservation_no"
    )


if __name__ == "__main__":
    # folder
    source_path = "../data/V2G4Carsharing"
    out_path = "data"
    os.makedirs(out_path)

    # load and preprocess all
    data_user = preprocess_user(source_path)
    data_vehicle = preprocess_vehicle(source_path)
    data_station = preprocess_station(source_path)
    data_reservation = preprocess_reservation(source_path)
    data_v2b = preprocess_v2b(source_path)

    data = [data_user, data_vehicle, data_station, data_reservation, data_v2b]
    save_name = [
        "user", "vehicle", "station", "all_reservation", "vehicle_to_base"
    ]

    # Rename columns to lower case and save
    for df, df_name in zip(data, save_name):
        index_name = df.index.name
        print(df_name, index_name, type(df))

        # modify columns
        new_names = {name: name.lower() for name in df.reset_index().columns}
        if df_name == "all_reservation":
            new_names["BASESTART_NO"] = "start_station_no"
            new_names["BASEEND_NO"] = "end_station_no"
        if df_name == "vehicle_to_base":
            new_names["BASE_NO"] = "station_no"

        df = df.reset_index().rename(columns=new_names
                                     ).set_index(index_name.lower())

        # write
        if df_name in ["user", "station"]:
            write_geodataframe(df, os.path.join(out_path, f"{df_name}.csv"))
        else:
            df.to_csv(
                os.path.join(out_path, f"{df_name}.csv"),
                index=index_name.lower()
            )
        print("Written file")

    # split whole table of reservations into service reservation, cancelled etc
    split_reservation(out_path)

    # convert v2b into relocations
    data_v2b = pd.read_csv(
        os.path.join(out_path, "vehicle_to_base.csv"), index_col="v2b_no"
    )
    relocations = v2b_to_relocations(data_v2b)
    relocations.index.name = "relocation_no"
    relocations.to_csv(
        os.path.join(out_path, "relocation.csv"), index="relocation_no"
    )
