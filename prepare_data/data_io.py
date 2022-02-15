import json
import os
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from preprocessing import clean_reservations


def load_data_postgis(path_credentials):
    with open(path_credentials, "r") as infile:
        db_credentials = json.load(infile)

    def get_con():
        return psycopg2.connect(**db_credentials)

    engine = create_engine('postgresql+psycopg2://', creator=get_con)
    vehicle = pd.read_sql(
        "SELECT * FROM mobility.vehicle", engine, index_col="vehicle_no"
    )
    # EV models
    ev_models = pd.read_sql("SELECT * FROM mobility.ev_models",
                            engine).set_index(["brand_name", "model_name"])

    # Service reservations
    # Note: empty atm! no EV one-way service reservations exist!
    sql_ev = (
        "SELECT * FROM mobility.service_reservation WHERE energytypegroup\
     = 'Electro' AND canceldate is null AND start_station_no != end_station_no"
    )
    services = pd.read_sql(sql_ev, engine, index_col="reservation_no")
    # Reservations
    sql_ev = (
        "SELECT * FROM mobility.reservations WHERE energytypegroup = 'Electro'"
    )
    ev_reservation = pd.read_sql(sql_ev, engine, index_col="reservation_no")

    # one_way_services = services[
    #     pd.isna(services["canceldate"])
    #     & (services["start_station_no"] != services["end_station_no"])
    #     & (services["energytypegroup"] == "Electro")]
    ev_reservation = pd.concat((ev_reservation, services))

    # merge with vehicle and ev models
    vehicle_w_model = vehicle.merge(
        ev_models, left_on=("brand_name", "model_name"), right_index=True
    )
    ev_reservation = ev_reservation.merge(
        vehicle_w_model, how="left", left_on="vehicle_no", right_index=True
    )
    # get vehicle to base
    sql_v2b = "SELECT * FROM mobility.relocation"
    relocation = pd.read_sql(sql_v2b, engine, index_col="relocation_no")
    return ev_reservation, relocation


def load_data_csv(path_clean_data):
    reservation = pd.read_csv(
        os.path.join(path_clean_data, "reservation.csv"),
        index_col="reservation_no"
    )
    vehicle = pd.read_csv(
        os.path.join(path_clean_data, "vehicle.csv"), index_col="vehicle_no"
    )
    ev_models = pd.read_csv(os.path.join("csv", "ev_models.csv")
                            ).set_index(["brand_name", "model_name"])
    # restrict to EVs for now
    ev_reservation = reservation[reservation["energytypegroup"] == "Electro"]

    # add Service reservations
    services = pd.read_csv(
        os.path.join(path_clean_data, "service_reservation.csv"),
        index_col="reservation_no"
    )
    # Note: empty atm! no EV one-way service reservations exist!
    one_way_services = services[
        pd.isna(services["canceldate"])
        & (services["start_station_no"] != services["end_station_no"])
        & (services["energytypegroup"] == "Electro")]
    ev_reservation = pd.concat((ev_reservation, one_way_services))

    # merge with vehicle and ev models
    vehicle_w_model = vehicle.merge(
        ev_models, left_on=("brand_name", "model_name"), right_index=True
    )
    ev_reservation = ev_reservation.merge(
        vehicle_w_model, how="left", left_on="vehicle_no", right_index=True
    )
    # get vehicle to base
    relocation = pd.read_csv(
        os.path.join(path_clean_data, "relocation.csv"),
        index_col="relocation_no"
    )
    return ev_reservation, relocation


def load_ev_data(
    path_clean_data="postgis", path_credentials="../../../goeco_login.json"
):
    if os.path.exists(path_clean_data):
        ev_reservation, relocation = load_data_csv(path_clean_data)
    else:
        ev_reservation, relocation = load_data_postgis(
            path_credentials=path_credentials
        )

    # preprocess bookings
    ev_reservation = clean_reservations(ev_reservation)

    # preprocess relocations - restrict v2b to the relevant vehicles
    unique_veh_ids = ev_reservation["vehicle_no"].unique()
    relocation = relocation[relocation["vehicle_no"].isin(unique_veh_ids)]
    print("number of relocations", len(relocation))
    # append to reservations
    ev_reservation = pd.concat((ev_reservation, relocation))

    return ev_reservation


def save_matrix(
    matrix, index=None, columns=None, name="matrix", out_path="outputs"
):
    station_df = pd.DataFrame(
        np.swapaxes(matrix, 1, 0), index=index, columns=columns
    )
    station_df.index.name = "vehicle_no"
    station_df.to_csv(os.path.join(out_path, f"{name}.csv"))
