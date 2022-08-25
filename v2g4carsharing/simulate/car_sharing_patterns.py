import numpy as np
import geopandas as gpd
import os
import time
import pandas as pd
from shapely import wkt
from ast import literal_eval
import warnings

from v2g4carsharing.import_data.import_utils import to_datetime_bizend

RANDOM_DATE = pd.to_datetime("2020-01-20")


def load_stations(path_car_sharing_data):
    station_df = pd.read_csv(os.path.join(path_car_sharing_data, "station.csv"), index_col="station_no")
    # load geom
    station_df["geom"] = station_df["geom"].apply(wkt.loads)
    station_df = gpd.GeoDataFrame(station_df, geometry="geom", crs="EPSG:4326")
    station_df = station_df.to_crs("EPSG:2056")
    return station_df


def carsharing_availability_one_day(in_path, date=RANDOM_DATE):
    veh2base = pd.read_csv(os.path.join(in_path, "vehicle_to_base.csv"))
    # reduce to the ones that are in the reservation
    station_df = load_stations(in_path)
    station_df = station_df[station_df["in_reservation"]]

    # reduce veh2base to the ones with suitable stations
    # PROBLEM: now we want to work with all stations with in_reservation=True. So we also need vehicle IDs for all of
    # them. So I think we need to start desigining scenarios already. It does not make sense to have a test case right now

    # to datetime
    veh2base["bizbeg"] = pd.to_datetime(veh2base["bizbeg"])
    veh2base["bizend"] = veh2base["bizend"].apply(to_datetime_bizend)

    # filter for time frame
    after_start = veh2base["bizbeg"] < date
    bef_end = veh2base["bizend"] > date
    veh_avail = veh2base[after_start & bef_end].drop(["bizbeg", "bizend"], axis=1)

    # get per vehicle information - merge by station
    veh_avail = veh_avail.merge(station_df, left_on="station_no", right_index=True).drop(["v2b_no"], axis=1)

    # clean: remove internaluse stations and free floating
    veh_avail = veh_avail[(veh_avail["resmode"] == "Stationsbasiert") & pd.isna(veh_avail["internaluse"])]
    assert len(veh_avail["vehicle_no"].unique()) == len(veh_avail)

    station_count = (
        veh_avail.groupby("station_no")
        .agg({"vehicle_no": list, "geom": "first"})
        .rename(columns={"vehicle_no": "vehicle_list"})
    )
    station_count["vehicle_no"] = station_count["vehicle_list"].apply(len)
    # to geodataframe
    station_count = gpd.GeoDataFrame(station_count, geometry="geom", crs="EPSG:2056")

    return station_count


def load_station_scenario(path):
    # TODO: just load the scenario or do we need to do it in several steps, e.g. assign EVs or so
    return pd.read_csv(path, index_col="station_no", converters={'vehicle_list': literal_eval})


def derive_decision_time(acts_gdf_mode):
    # rename distance column if necessary
    if "distance" not in acts_gdf_mode.columns:
        if "feat_distance" not in acts_gdf_mode.columns:
            raise RuntimeError("distance between geometries must be computed")
        acts_gdf_mode.rename(columns={"feat_distane": "distance"}, inplace=True)

    avg_drive_speed = 50  # kmh
    print("max distance between activities", round(acts_gdf_mode["distance"].max()))
    # get appriximate travel time in minutes
    acts_gdf_mode["drive_time"] = 60 * acts_gdf_mode["distance"] / (1000 * avg_drive_speed)
    # compute time for decision making
    acts_gdf_mode["mode_decision_time"] = (
        # in seconds, giving 10min decision time
        acts_gdf_mode["start_time_sec_destination"]
        - acts_gdf_mode["drive_time"] * 60
        - 10 * 60
    )
    # drop geometry for easier processing
    acts_gdf_mode.drop(["geom_origin", "geom_destination"], axis=1, inplace=True, errors="ignore")
    # drop the rows of activities that are repeated
    print("Number of activities", len(acts_gdf_mode))
    acts_gdf_mode = acts_gdf_mode[acts_gdf_mode["distance"] > 0]
    print("Activities after dropping 0-distance ones:", len(acts_gdf_mode))

    # correct wrong decision times (sometimes they are lower than the one of the previous activity,
    # due to rough approximation of vehicle speed)
    cond1, cond2 = np.array([True]), np.array([True])
    while np.sum(cond1 & cond2) > 0:
        acts_gdf_mode["prev_dec_time"] = acts_gdf_mode["mode_decision_time"].shift(1)
        acts_gdf_mode["prev_person"] = acts_gdf_mode["person_id"].shift(1)
        cond1 = acts_gdf_mode["prev_dec_time"] > acts_gdf_mode["mode_decision_time"]
        cond2 = acts_gdf_mode["prev_person"] == acts_gdf_mode["person_id"]
        # print(np.sum(cond1 & cond2))
        # reset decision time to after the previously chosen
        acts_gdf_mode.loc[(cond1 & cond2), "mode_decision_time"] = (
            acts_gdf_mode.loc[(cond1 & cond2), "prev_dec_time"] + 2 * 60
        )  # add five minutes to the previous decision time

    # now all the decision times should be sorted
    assert acts_gdf_mode.equals(acts_gdf_mode.sort_values(["person_id", "mode_decision_time"]))

    return acts_gdf_mode


def assign_mode(acts_gdf_mode, per_station_veh_avail, mode_choice_function):
    # now sort by mode decision time, not by person
    acts_gdf_mode = acts_gdf_mode.sort_values("mode_decision_time")
    # keep track for each person where their shared trips started
    shared_starting_station = {}
    # keep track of the vehicle ID of the currently borrowed car
    shared_vehicle_id = {}

    tic = time.time()
    final_modes, final_veh_ids = [], []
    for i, row in acts_gdf_mode.iterrows():
        # get necessary variables
        person_id = row["person_id"]
        closest_station = row["closest_station_origin"]  # closest station at previous activity for starting
        nr_avail = len(per_station_veh_avail[closest_station])
        distance = row["distance"]

        # check if we already borrowed a car --> need to keep it for return trip
        shared_start = shared_starting_station.get(person_id, None)
        if shared_start:
            shared_vehicle = shared_vehicle_id[person_id]
            final_veh_ids.append(shared_vehicle)  # current veh ID is the shared vehicle
            final_modes.append("Mode::CarsharingMobility")
            # check whether we are back at the start station --> give back the car
            if shared_start == row["closest_station_destination"]:
                # give vehicle back to station
                per_station_veh_avail[shared_start].append(shared_vehicle)
                # clean the dictionary entries
                del shared_starting_station[person_id]
                del shared_vehicle_id[person_id]
                # print(person_id, "gave shared car back at station", shared_start)
            continue

        # otherwise: decide whether to borrow the car
        if nr_avail < 1:
            mode = "Mode::Car"
        else:
            mode = mode_choice_function(row)

        # if shared, set vehicle as borrowed and remember the pick up station (for return)
        if mode == 'Mode::CarsharingMobility':
            veh_id_borrow = per_station_veh_avail[closest_station].pop()
            shared_vehicle_id[person_id] = veh_id_borrow
            shared_starting_station[person_id] = closest_station
            final_veh_ids.append(veh_id_borrow)
            print(person_id, "borrowed car at station", closest_station)

        assert len(per_station_veh_avail[closest_station]) >= 0

        final_modes.append(mode)
        if mode != 'Mode::CarsharingMobility':
            final_veh_ids.append(-1)
    print("time for reservation generation:", time.time() - tic)
    acts_gdf_mode["mode"] = final_modes
    acts_gdf_mode["vehicle_no"] = final_veh_ids
    # sort back
    acts_gdf_mode.sort_values(["person_id", "activity_index"], inplace=True)
    return acts_gdf_mode


def derive_reservations(acts_gdf_mode):
    acts_gdf_mode["index_temp"] = acts_gdf_mode.index.values
    acts_gdf_mode["next_person_id"] = acts_gdf_mode["person_id"].shift(-1).values
    acts_gdf_mode["next_mode"] = acts_gdf_mode["mode"].shift(-1).values
    # # relevant if including cond5 / cond6
    # shared_rides["next_activity_index"] = shared_rides["activity_index"].shift(-1).values
    # shared_rides["next_vehicle_no"] = shared_rides["vehicle_no"].shift(-1).values

    # merge the bookings to subsequent activities:
    cond = pd.Series(data=False, index=acts_gdf_mode.index)
    cond_old = pd.Series(data=True, index=acts_gdf_mode.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        acts_gdf_mode["next_id"] = acts_gdf_mode["index_temp"].shift(-1).values

        # identify rows to merge
        cond0 = acts_gdf_mode["next_person_id"] == acts_gdf_mode["person_id"]
        cond1 = acts_gdf_mode["index_temp"] != acts_gdf_mode["next_id"]  # already merged
        cond2 = acts_gdf_mode["mode"] == 'Mode::CarsharingMobility'
        cond3 = acts_gdf_mode["next_mode"] == 'Mode::CarsharingMobility'
        cond4 = ~pd.isna(acts_gdf_mode["next_id"])
        # cond5 = shared_rides["activity_index"] == shared_rides["next_activity_index"] - 1
        # # we cannot trust activity index because the repeated locations were removed --> cond5 is unsuitable.
        # # therefore, cond 2 and 3 were included instead
        # cond6 = shared_rides["vehicle_no"] == shared_rides["next_vehicle_no"]
        # # TODO: include cond6? Contra: We might give the vehicle back and borrow it an hour later at the same location,
        # # which does not make sense.

        # todo: vehicle IDs match
        cond = cond0 & cond1 & cond2 & cond3 & cond4

        # assign index to next row
        acts_gdf_mode.loc[cond, "index_temp"] = acts_gdf_mode.loc[cond, "next_id"]

        # check whether anything was changed
        cond_diff = cond != cond_old
        cond_old = cond.copy()

    # now after setting the index, reduce to shared
    shared_rides = acts_gdf_mode[acts_gdf_mode["mode"] == 'Mode::CarsharingMobility']

    # aggregate into car sharing bookings instead
    agg_dict = {
        "id": list,
        "person_id": "first",
        "vehicle_no": "first",
        "distance_to_station_destination": "last",
        "mode_decision_time": "first",  # first decision time is the start of the booking
        "start_time_sec_destination": "last",  # last start time is the end of the booking
        "distance": "sum",  # covered distance
        "closest_station_destination": "last",
    }
    sim_reservations = shared_rides.reset_index().groupby(by="index_temp").agg(agg_dict)
    sim_reservations = sim_reservations.rename(
        columns={
            "id": "trip_ids",
            "person_id": "person_no",
            "mode_decision_time": "reservationfrom",
            "start_time_sec_destination": "reservationto",
            "closest_station_destination": "start_station_no",
        }
    )
    sim_reservations.index.name = "reservation_no"

    # amend with some more information
    sim_reservations["drive_km"] = sim_reservations["distance"] / 1000
    sim_reservations["duration"] = (sim_reservations["reservationto"] - sim_reservations["reservationfrom"]) / 60 / 60

    return sim_reservations


def load_trips(in_path_sim_trips):
    acts_gdf = pd.read_csv(in_path_sim_trips).set_index("id")
    if not "geom_origin" in acts_gdf.columns:
        warnings.warn("No geometry columns. Loading pure dataframe")
        return acts_gdf
    print("Loaded trips", len(acts_gdf))
    acts_gdf.dropna(subset=["geom_origin", "geom_destination"], inplace=True)
    acts_gdf["geom_origin"] = acts_gdf["geom_origin"].apply(wkt.loads)
    acts_gdf = gpd.GeoDataFrame(acts_gdf, geometry="geom_origin", crs="EPSG:2056")
    acts_gdf["geom_destination"] = gpd.GeoSeries(acts_gdf["geom_destination"].apply(wkt.loads))
    print("removed nan geometries and loaded geometry, leftover trips:", len(acts_gdf))
    return acts_gdf

