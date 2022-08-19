from ast import Mod
import os
import pickle
import argparse
import pandas as pd
from v2g4carsharing.mode_choice_model.random_forest import RandomForestWrapper
from v2g4carsharing.simulate.car_sharing_patterns import (
    load_trips,
    carsharing_availability_one_day,
    derive_decision_time,
    derive_reservations,
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

    station_count = carsharing_availability_one_day(in_path_carsharing)

    # get closest station to origin
    acts_gdf = acts_gdf.sjoin_nearest(station_count, distance_col="distance_to_station_origin")
    acts_gdf.rename(columns={"index_right": "closest_station_origin"}, inplace=True)
    # get closest station to destination
    acts_gdf.set_geometry("geom_destination", inplace=True)
    acts_gdf = acts_gdf.sjoin_nearest(station_count, distance_col="distance_to_station_destination")
    acts_gdf.rename(columns={"index_right": "closest_station_destination"}, inplace=True)
    # sort
    acts_gdf.sort_values(["person_id", "activity_index"], inplace=True)

    # get time when decision is made
    acts_gdf = derive_decision_time(acts_gdf)

    # keep track in a dictionary how many vehicles are available at each station
    per_station_veh_avail = station_count["vehicle_list"].to_dict()

    # Run: iteratively assign modes
    acts_gdf_mode = assign_mode(acts_gdf, per_station_veh_avail, mode_choice_model)

    # get shared only and derive the reservations by merging subsequent car sharing trips
    sim_reservations = derive_reservations(acts_gdf_mode)

    sim_reservations.to_csv(os.path.join(out_path))

