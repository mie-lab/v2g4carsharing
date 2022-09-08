import os
import pandas as pd
import argparse

from v2g4carsharing.import_data.data_cleaning import *
from v2g4carsharing.import_data.to_postgis import write_to_postgis


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default="../data/V2G4Carsharing", help="path to raw data")
    parser.add_argument("-o", "--out_path", type=str, default="data", help="path to save output")
    parser.add_argument("-p", "--postgis_json_path", type=str, default=None, help="path to postgis access json file")
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()
    # set folders
    source_path = args.in_path
    out_path = args.out_path
    postgis_json_path = args.postgis_json_path  # Set to true for transfering from the data path to postgis database
    os.makedirs(out_path, exist_ok=True)

    # load and preprocess all
    data_user = preprocess_user(source_path)
    data_vehicle = preprocess_vehicle(source_path)
    data_station = preprocess_station(source_path)  # , path_for_duplicates=out_path)
    data_reservation = preprocess_reservation(source_path)
    data_v2b = preprocess_v2b(source_path)

    data = [data_user, data_vehicle, data_station, data_reservation, data_v2b]
    save_name = ["user", "vehicle", "station", "all_reservation", "vehicle_to_base"]

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
        if df_name == "station":
            # Add the variable whether it's in the reservations
            # valid stations: no free floating and no service reservations
            valid_stations = (
                data_reservation[
                    (data_reservation["TRIPMODE"] != "FreeFloating (Rückgabe an einem beliebigen Ort)")
                    & (data_reservation["RESERVATIONTYPE"] == "Normal")
                ]
            )["BASESTART_NO"].unique()
            # note: some stations are still "internaluse", but this is B2B and CoronaMiniMiete, so we include it
            df["in_reservation"] = df.index.isin(valid_stations)
            print("In reservations", sum(df["in_reservation"]))

        df = df.reset_index().rename(columns=new_names).set_index(index_name.lower())

        # write
        if df_name in ["user", "station"]:
            write_geodataframe(df, os.path.join(out_path, f"{df_name}.csv"))
        else:
            df.to_csv(os.path.join(out_path, f"{df_name}.csv"), index=index_name.lower())
        print("Written file")

    # split whole table of reservations into service reservation, cancelled etc
    split_reservation(out_path)

    # convert v2b into relocations
    data_v2b = pd.read_csv(os.path.join(out_path, "vehicle_to_base.csv"), index_col="v2b_no")
    relocations = v2b_to_relocations(data_v2b)
    relocations.index.name = "relocation_no"
    relocations.to_csv(os.path.join(out_path, "relocation.csv"), index="relocation_no")

    if postgis_json_path is not None:
        write_to_postgis(out_path, postgis_json_path)
