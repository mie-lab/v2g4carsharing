import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pickle
import os
import time
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
import scipy

from import_data.import_utils import to_datetime_bizend
from mode_choice_model.simple_choice_models import *

RANDOM_DATE = pd.to_datetime("2020-01-20")


def carsharing_availability_one_day(date=RANDOM_DATE):
    veh2base = pd.read_csv("../v2g4carsharing/data/vehicle_to_base.csv")
    station_df = pd.read_csv("../v2g4carsharing/data/station_df.csv")
    station_df["geom"] = station_df["geom"].apply(wkt.loads)
    station_df = gpd.GeoDataFrame(station_df, geometry="geom", crs="EPSG:4326")
    station_df = station_df.to_crs("EPSG:2056")
    station_df.set_index("station_no", inplace=True)

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

    station_count = veh_avail.groupby("station_no").agg({"vehicle_no": "count", "geom": "first"})
    station_count = gpd.GeoDataFrame(station_count, geometry="geom", crs="EPSG:2056")
    return station_count


def load_activities(cache_path="../external_repos/ch-zh-synpop/cache_2022/"):
    # load geometry data
    with open(
        os.path.join(cache_path, "synthesis.population.spatial.locations__4a0fb9027293545c4692f3be972f79b8.p",), "rb",
    ) as infile:
        geoms = pickle.load(infile).set_index(["person_id", "activity_index"])

    # load activity data
    with open(
        os.path.join(cache_path, "synthesis.population.activities__4a0fb9027293545c4692f3be972f79b8.p",), "rb",
    ) as infile:
        activities = pickle.load(infile).set_index(["person_id", "activity_index"])

    acts = activities.merge(geoms, how="left", left_index=True, right_index=True)
    print(
        "Number activities", len(acts), "nr unique users", len(acts.reset_index()["person_id"].unique()),
    )
    acts = gpd.GeoDataFrame(acts, geometry="geometry")

    # sort by person and activity_index
    acts = acts.reset_index().sort_values(["person_id", "activity_index"])
    # assign new IDs for activity identification
    acts["id"] = np.arange(len(acts))
    acts.set_index("id", inplace=True)

    # get the previous geometry of each row
    acts["prev_geometry"] = acts["geometry"].shift(1)
    # get distance that must be travelled to this activity
    acts["distance_from_prev"] = acts.distance(acts["prev_geometry"])

    return acts


def derive_decision_time(acts_gdf):
    avg_drive_speed = 50  # kmh
    print("max distance between activities", round(acts_gdf["distance_from_prev"].max()))
    # get appriximate travel time in minutes
    acts_gdf["drive_time"] = 60 * acts_gdf["distance_from_prev"] / (1000 * avg_drive_speed)
    # compute time for decision making
    acts_gdf["mode_decision_time"] = (
        acts_gdf["start_time"] - acts_gdf["drive_time"] * 60 - 10 * 60  # in seconds, giving 10min decision time
    )
    # drop first activity
    essential_col = acts_gdf.dropna(subset="mode_decision_time")
    # reduce to required columns
    essential_col = essential_col[
        [
            "person_id",
            "activity_index",
            "mode_decision_time",
            "start_time",
            "end_time",
            "duration",
            "purpose",
            "distance_from_prev",
            "distance_to_station",
            "closest_station",
            "prev_distance_to_station",
            "prev_closest_station",
        ]
    ]
    # drop the rows of activities that are repeated
    print("Number of activities", len(essential_col))
    essential_col = essential_col[essential_col["distance_from_prev"] > 0]
    print("Activities after dropping 0-distance ones:", len(essential_col))

    # correct wrong decision times (sometimes they are lower than the one of the previous activity,
    # due to rough approximation of vehicle speed)
    cond1, cond2 = np.array([True]), np.array([True])
    while np.sum(cond1 & cond2) > 0:
        essential_col["prev_dec_time"] = essential_col["mode_decision_time"].shift(1)
        essential_col["prev_person"] = essential_col["person_id"].shift(1)
        cond1 = essential_col["prev_dec_time"] > essential_col["mode_decision_time"]
        cond2 = essential_col["prev_person"] == essential_col["person_id"]
        # print(np.sum(cond1 & cond2))
        # reset decision time to after the previously chosen
        essential_col.loc[(cond1 & cond2), "mode_decision_time"] = (
            essential_col.loc[(cond1 & cond2), "prev_dec_time"] + 2 * 60
        )  # add five minutes to the previous decision time

    # now all the decision times should be sorted
    assert essential_col.equals(essential_col.sort_values(["person_id", "mode_decision_time"]))

    return essential_col


def assign_mode(essential_col):
    # now sort by mode decision time, not by person
    essential_col = essential_col.sort_values("mode_decision_time")
    # keep track in a dictionary how many vehicles are available at each station
    per_station_veh_avail = station_count["vehicle_no"].to_dict()
    # keep track for each person where their shared trips started
    shared_starting_station = {}

    tic = time.time()
    final_modes = []
    for i, row in essential_col.iterrows():
        # get necessary variables
        person_id = row["person_id"]
        closest_station = row["prev_closest_station"]  # closest station at previous activity for starting
        nr_avail = per_station_veh_avail[closest_station]
        distance = row["distance_from_prev"]

        # check if we already borrowed a car --> need to keep it for return trip
        shared_start = shared_starting_station.get(person_id, None)
        if shared_start:
            final_modes.append("shared")
            # check whether we are back at the start station --> give back the car
            if shared_start == row["closest_station"]:
                per_station_veh_avail[shared_start] = nr_avail + 1
                shared_starting_station[person_id] = None
                # print(person_id, "gave shared car back at station", shared_start)
            continue

        # otherwise: decide whether to borrow the car
        if nr_avail < 1:
            mode = "car"
        else:
            # mode = distance_dependent_mode_choice(distance, row["prev_distance_to_station"])
            mode = simple_mode_choice(distance)

        # if shared, set vehicle as borrowed and remember the pick up station (for return)
        if mode == "shared":
            per_station_veh_avail[closest_station] = nr_avail - 1
            shared_starting_station[person_id] = closest_station
            # print(person_id, "borrowed car at station", closest_station)

        assert per_station_veh_avail[closest_station] >= 0

        final_modes.append(mode)
    print("time for reservation generation:", time.time() - tic)
    essential_col["mode"] = final_modes
    # sort back
    essential_col.sort_values(["person_id", "activity_index"], inplace=True)
    return essential_col


def derive_reservations(shared_rides):
    shared_rides["index_temp"] = shared_rides.index.values
    shared_rides["next_person_id"] = shared_rides["person_id"].shift(-1).values
    shared_rides["next_activity_index"] = shared_rides["activity_index"].shift(-1).values

    # merge the bookings to subsequent activities:
    cond = pd.Series(data=False, index=shared_rides.index)
    cond_old = pd.Series(data=True, index=shared_rides.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        shared_rides["next_id"] = shared_rides["index_temp"].shift(-1).values

        # identify rows to merge
        cond0 = shared_rides["next_person_id"] == shared_rides["person_id"]
        cond3 = ~pd.isna(shared_rides["next_id"])
        cond1 = shared_rides["index_temp"] != shared_rides["next_id"]  # already merged
        cond2 = shared_rides["activity_index"] == shared_rides["next_activity_index"] - 1
        #     print(np.where(cond0), np.where(cond1), np.where(cond2), np.where(cond3))

        # todo: vehicle IDs match
        cond = cond0 & cond1 & cond2 & cond3

        # assign index to next row
        shared_rides.loc[cond, "index_temp"] = shared_rides.loc[cond, "next_id"]

        # check whether anything was changed
        cond_diff = cond != cond_old
        cond_old = cond.copy()

    # aggregate into car sharing bookings instead
    agg_dict = {
        "id": list,
        "person_id": "first",
        "distance_to_station": "last",
        "mode_decision_time": "first",  # first decision time is the start of the booking
        "start_time": "last",  # last start time is the end of the booking
        "distance_from_prev": "sum",  # covered distance
        "closest_station": "last",
    }
    sim_reservations = shared_rides.reset_index().groupby(by="index_temp").agg(agg_dict)
    sim_reservations = sim_reservations.rename(
        columns={
            "id": "trip_ids",
            "person_id": "person_no",
            "mode_decision_time": "reservationfrom",
            "start_time": "reservationto",
            "closest_station": "start_station_no",
        }
    )
    sim_reservations.index.name = "reservation_no"

    # amend with some more information
    sim_reservations["drive_km"] = sim_reservations["distance_from_prev"] / 1000
    sim_reservations["duration"] = (sim_reservations["reservationto"] - sim_reservations["reservationfrom"]) / 60 / 60

    return sim_reservations


if __name__ == "__main__":
    save_name = "simple"

    # load activities and shared-cars availability
    acts_gdf = load_activities()
    station_count = carsharing_availability_one_day()

    # merge both
    acts_gdf = acts_gdf.sjoin_nearest(station_count, distance_col="distance_to_station")
    acts_gdf.rename(columns={"index_right": "closest_station"}, inplace=True)
    acts_gdf.sort_values(["person_id", "activity_index"], inplace=True)
    acts_gdf["prev_closest_station"] = acts_gdf["closest_station"].shift(1)
    acts_gdf["prev_distance_to_station"] = acts_gdf["distance_to_station"].shift(1)

    # get time when decision is made
    acts_gdf = derive_decision_time(acts_gdf)

    # Run: iteratively assign modes
    acts_gdf_mode = assign_mode(acts_gdf)

    # get shared only and derive the reservations by merging subsequent car sharing trips
    shared_rides = acts_gdf_mode[acts_gdf_mode["mode"] == "shared"]
    simulated_reservations = derive_reservations(shared_rides)

    simulated_reservations.to_csv(os.path.join("outputs", "simulated_car_sharing", save_name + ".csv"))
