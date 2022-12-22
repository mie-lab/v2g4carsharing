import json
import os
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from v2g4carsharing.optimization_data.preprocessing import clean_reservations


def load_data_postgis(path_credentials, filter_ev=False):
    with open(path_credentials, "r") as infile:
        db_credentials = json.load(infile)

    def get_con():
        return psycopg2.connect(**db_credentials)

    engine = create_engine('postgresql+psycopg2://', creator=get_con)

    # Service reservations
    sql_service = (
        "SELECT * FROM mobility.service_reservation WHERE canceldate is\
             null AND start_station_no != end_station_no"
    )
    if filter_ev:
        sql_service += " AND energytypegroup = 'Electro'"
    services = pd.read_sql(sql_service, engine, index_col="reservation_no")

    # Reservations
    sql_ev = ("SELECT * FROM mobility.reservations")
    if filter_ev:
        sql_ev += " WHERE energytypegroup = 'Electro'"
    reservation = pd.read_sql(sql_ev, engine, index_col="reservation_no")

    reservation = pd.concat((reservation, services))

    # get vehicle to base
    sql_v2b = "SELECT * FROM mobility.relocation"
    relocation = pd.read_sql(sql_v2b, engine, index_col="relocation_no")
    return reservation, relocation


def load_data_csv(path_clean_data, filter_ev=False):
    reservation = pd.read_csv(
        os.path.join(path_clean_data, "reservation.csv"),
        index_col="reservation_no"
    )

    # add Service reservations
    services = pd.read_csv(
        os.path.join(path_clean_data, "service_reservation.csv"),
        index_col="reservation_no"
    )
    # get the service reservations that are relocations
    one_way_services = services[
        pd.isna(services["canceldate"])
        & (services["start_station_no"] != services["end_station_no"])]
    # combine reservations and services
    reservation = pd.concat((reservation, one_way_services))

    # restrict to EVs for now
    if filter_ev:
        reservation = reservation[reservation["energytypegroup"] == "Electro"]

    # get vehicle to base
    relocation = pd.read_csv(
        os.path.join(path_clean_data, "relocation.csv"),
        index_col="relocation_no"
    )
    return reservation, relocation


def simulate_filter_evs(reservation, sim_ev_mode="sample_current"):
    """
    Simulate EV models for non-ev vehicles or filter out

    Args:
        reservation (pd.DataFrame): loaded reservations
        sim_ev_mode (str, optional): Mode what to do with non-electric vehices.
            Possible modes:
                `filter`: Filter out all non-electric vehicles
                `sample_current` (default): Sample random models according to
                    current distribution of models
                `scenario_1`: Loads scenario from csv with models by vehice cat

    Returns:
        reservation (pd.DataFrame): reservations together with EV specs
    """

    def get_ev_models(reservation):
        evs_in_fleet = reservation[reservation["energytypegroup"] == "Electro"
                                   ].groupby("vehicle_no").agg(
                                       {
                                           "brand_name": "first",
                                           "model_name": "first"
                                       }
                                   )
        return evs_in_fleet

    ev_models = pd.read_csv(os.path.join("csv", "ev_models.csv")
                            ).set_index(["brand_name", "model_name"])
    if sim_ev_mode == "filter":
        return reservation.merge(
            ev_models, left_on=("brand_name", "model_name"), right_index=True
        )
    elif sim_ev_mode == "sample_current":
        # first: get brand and model for all EVs
        evs_in_fleet = get_ev_models(reservation)
        # get number of occurences for each EV
        ev_model_occurence = evs_in_fleet.groupby(
            ["brand_name", "model_name"]
        ).agg({
            "model_name": "count"
        }).rename(columns={"model_name": "occurence"})
        ev_model_occurence["occurence"] = ev_model_occurence[
            "occurence"] / ev_model_occurence["occurence"].sum()
        ev_model_occurence.reset_index(inplace=True)
        # get IDs of ICE vehicles
        non_ev_vehicles = (
            reservation[reservation["energytypegroup"] != "Electro"]
        )["vehicle_no"].unique()
        # select random EV model for each ICE vehicle
        range_index = np.arange(len(ev_model_occurence))
        rand_models = np.random.choice(
            range_index,
            size=len(non_ev_vehicles),
            p=ev_model_occurence["occurence"]
        )
        df_new = pd.DataFrame()
        df_new["brand_name"] = ev_model_occurence.loc[rand_models]["brand_name"
                                                                   ]
        df_new["model_name"] = ev_model_occurence.loc[rand_models]["model_name"
                                                                   ]
        df_new["vehicle_no"] = non_ev_vehicles
        df_new.set_index("vehicle_no", inplace=True)

        # merge real evs with fake evs
        all_models = pd.concat((df_new, evs_in_fleet))

        reservation = reservation.drop(
            columns=["brand_name", "model_name"]
        ).merge(all_models, left_on="vehicle_no", right_index=True)
        reservation = reservation.merge(
            ev_models, left_on=("brand_name", "model_name"), right_index=True
        )
        return reservation
    elif sim_ev_mode == "scenario_1":
        # load brand and model for all non-EVs in this scenario
        scenario = pd.read_csv(os.path.join("csv", "scenario_1.csv")
                               ).set_index("vehicle_no")
        # second: get brand and model for all EVs
        evs_in_fleet = get_ev_models(reservation)
        # concatenate to yield a list of all brand-models for fake and real EVs
        # Note: it can happen that EV models are in the real EVs and in the fake EVs. We use keep=first to remove the
        # duplicates but to keep the first ones
        all_models = pd.concat((evs_in_fleet, scenario)).reset_index().drop_duplicates(
            subset="vehicle_no", keep="first"
        )
        # replace real models with the simulated ones
        reservation = reservation.drop(
            columns=["brand_name", "model_name"]
        ).merge(all_models, left_on="vehicle_no", right_on="vehicle_no")
        # merge with EV specs
        return reservation.merge(
            ev_models, left_on=("brand_name", "model_name"), right_index=True
        )
    else:
        raise NotImplementedError(
            "mode must be one of filter, sample_current or scenario_1"
        )


def load_ev_data(
    path_clean_data="postgis",
    sim_ev_mode="scenario_1",
    path_credentials="../../../goeco_login.json",
    simulate = False
):
    filter_ev = sim_ev_mode == "filter"  # filter out or simulate non-ev cars?
    if simulate:
        reservation = pd.read_csv(os.path.join(path_clean_data, "sim_reservations.csv"))
        relocation = pd.DataFrame(columns=["vehicle_no"])
        reservation["energytypegroup"] = "Benzin"
        reservation["brand_name"] = "None"
        reservation["model_name"] = "None"
        reservation["drive_firststart"] = reservation["reservationfrom"]
        reservation["drive_lastend"] = reservation["reservationto"]
        reservation["end_station_no"] = reservation["start_station_no"]
        reservation["reservationtype"] = "Simulated"
    else:
        if os.path.exists(path_clean_data):
            reservation, relocation = load_data_csv(
                path_clean_data, filter_ev=filter_ev
            )
        else:
            reservation, relocation = load_data_postgis(
                path_credentials=path_credentials, filter_ev=filter_ev
            )
    # load, merge and filter data
    ev_reservation = simulate_filter_evs(reservation, sim_ev_mode=sim_ev_mode)

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
