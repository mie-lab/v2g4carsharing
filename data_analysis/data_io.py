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

    return ev_reservation, vehicle, ev_models


def load_data_csv(path_clean_data):
    reservation = pd.read_csv(
        os.path.join(path_clean_data, "reservation.csv"),
        index_col="reservation_no"
    )
    vehicle = pd.read_csv(
        os.path.join(path_clean_data, "vehicle.csv"), index_col="vehicle_no"
    )
    ev_models = pd.read_csv(os.path.join(path_clean_data, "ev_models.csv")
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

    return ev_reservation, vehicle, ev_models


def load_ev_data(
    path_clean_data="postgis", path_credentials="../../../goeco_login.json"
):
    if os.path.exists(path_clean_data):
        ev_reservation, vehicle, ev_models = load_data_csv(path_clean_data)
    else:
        ev_reservation, vehicle, ev_models = load_data_postgis(
            path_credentials=path_credentials
        )

    ev_reservation = ev_reservation.merge(
        vehicle, how="left", left_on="vehicle_no", right_index=True
    )
    ev_reservation = clean_reservations(ev_reservation)
    return ev_reservation, ev_models


def save_matrix(
    matrix, index=None, columns=None, name="matrix", out_path="outputs"
):
    station_df = pd.DataFrame(
        np.swapaxes(matrix, 1, 0), index=index, columns=columns
    )
    station_df.index.name = "vehicle_no"
    station_df.to_csv(os.path.join(out_path, f"{name}.csv"))
