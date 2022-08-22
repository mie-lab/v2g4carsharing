from ast import Mod
import os
import pickle
import argparse
import pandas as pd
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper
from v2g4carsharing.mode_choice_model.features import compute_dist_to_station
from v2g4carsharing.simulate.car_sharing_patterns import (
    load_trips,
    load_station_scenario,
    derive_decision_time,
    derive_reservations,
    load_stations,
    assign_mode,
)

from v2g4carsharing.mode_choice_model.simple_choice_models import *

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_path_car_sharing", type=str, default="data", help="path to Mobility car sharing data"
    )
    parser.add_argument(
        "-s",
        "--in_path_sim_trips",
        type=str,
        default=os.path.join("..", "data", "simulated_population", "sim_2022"),
        help="path to simulated trips csv",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=os.path.join("outputs", "simulated_car_sharing", "sim_carsharing.csv"),
        help="path to save output",
    )
    parser.add_argument(
        "-t",
        "--station_scenario",
        type=str,
        default=os.path.join("csv", "station_scenario", "same_stations.csv"),
        help="path to station_scenario",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=os.path.join("trained_models", "rf_test.p"),
        help="path to mode choice model",
    )
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()

    in_path_carsharing = args.in_path_car_sharing
    in_path_sim_trips = args.in_path_sim_trips
    out_path = args.out_path

    # load activities and shared-cars availability
    acts_gdf = load_trips(os.path.join(in_path_sim_trips, "trips_features.csv"))

    # define mode choice model
    # mode_choice_model = simple_mode_choice
    with open(args.model_path, "rb") as infile:
        mode_choice_model = pickle.load(infile)

    station_scenario = load_station_scenario(args.station_scenario)

    # get closest station to origin
    stations_in_trips = np.union1d(acts_gdf["closest_station_origin"], acts_gdf["closest_station_destination"])
    # Check if all stations appear in the station scenario
    assert len(stations_in_trips) == len(
        np.intersect1d(stations_in_trips, station_scenario.index)
    ), "Problem: All stations that appear as the closest station must appear in the station scenario"

    if "same_stations" not in args.station_scenario:
        # If we have the same stations, the distance to the closest station is automatically computed in the feature
        # computation. If we have more stations, we need to recompute where is the closest station (or do it at runtime
        # for computing the closest station that has vehicles available)
        assert "geom_origin" in acts_gdf.columns, "acts gdf must have geometry to recompute distance to station"
        station_df = load_stations(in_path_carsharing)
        # add geometry to scenario
        station_scenario = station_scenario.merge(station_df[["geom"]], left_index=True, right_index=True, how="left")
        # compute dist to station for each trip start and end point
        acts_gdf = compute_dist_to_station(acts_gdf, station_scenario)

    # sort
    acts_gdf.sort_values(["person_id", "activity_index"], inplace=True)

    # get time when decision is made
    acts_gdf = derive_decision_time(acts_gdf)

    # keep track in a dictionary how many vehicles are available at each station
    per_station_veh_avail = station_scenario["vehicle_list"].to_dict()

    # Run: iteratively assign modes
    acts_gdf_mode = assign_mode(acts_gdf, per_station_veh_avail, mode_choice_model)

    # get shared only and derive the reservations by merging subsequent car sharing trips
    sim_reservations = derive_reservations(acts_gdf_mode)

    sim_reservations.to_csv(os.path.join(out_path))

