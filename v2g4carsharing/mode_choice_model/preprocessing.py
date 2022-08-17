import pandas as pd
import trackintel as ti
import os
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString

INP_PATH_MOBIS = "../../teaching/mobis_project/MOBIS_Covid_version_2_raubal"
OUT_PATH = "../data/mobis/"


def clean_legs(legs_model):
    legs_model = legs_model[legs_model["in_switzerland"]]
    print("in switzerland", len(legs_model))
    legs_model = legs_model[~legs_model["implausible"]]
    print("not implausible", len(legs_model))
    legs_model.drop_duplicates(inplace=True)
    print("no duplicates", len(legs_model))
    # remove unused columns
    legs_model.drop(
        ["treatment", "phase", "started_at_tz", "finished_at_tz", "in_switzerland", "implausible",],
        inplace=True,
        axis=1,
    )
    # rename columns:
    legs_model.rename(columns={"participant_id": "user_id", "trip_id": "mt_trip_id"}, inplace=True)
    return legs_model


def add_geom_legs(legs_model):
    def to_linestring(row):
        """
        Convert start, middle and end coordinates into linestring
        Cheat: Doing +1 for end because otherwise so many geometries are invalid (tours)
        """
        if pd.isna(row["mid_x"]):
            return LineString([Point(row["start_x"], row["start_y"]), Point(row["end_x"] + 1, row["end_y"] + 1),])
        else:
            return LineString(
                [
                    Point(row["start_x"], row["start_y"]),
                    Point(row["mid_x"], row["mid_y"]),
                    Point(row["end_x"] + 1, row["end_y"] + 1),
                ]
            )

    print("adding geometry", len(legs_model))
    legs_model["geom"] = legs_model.apply(to_linestring, axis=1)
    legs_model = gpd.GeoDataFrame(legs_model)
    legs_model.set_geometry("geom", inplace=True)
    legs_model = legs_model[legs_model.geometry.is_valid]
    legs_model.drop(["start_x", "start_y", "mid_x", "mid_y", "end_x", "end_y"], axis=1, inplace=True)
    print("only valid geometries", len(legs_model))
    return legs_model


def clean_acts(acts_of_carsharing_users):
    # drop columns
    acts_of_carsharing_users = acts_of_carsharing_users[acts_of_carsharing_users["in_switzerland"]]
    print("in switzerland", len(acts_of_carsharing_users))
    acts_of_carsharing_users.drop(
        ["treatment", "phase", "started_at_tz", "finished_at_tz", "in_switzerland"], inplace=True, axis=1,
    )
    # rename
    acts_of_carsharing_users.rename(
        columns={"participant_id": "user_id", "activity_id": "id", "imputed_purpose": "purpose"}, inplace=True
    )
    return acts_of_carsharing_users


def tracking_data_carsharing_users(out_path=OUT_PATH):
    print("------ Loading Legs ---------")
    legs = pd.read_csv(os.path.join(INP_PATH_MOBIS, "tracking", "legs.csv"), index_col="leg_id")
    print("Preprocessing legs...")
    # car sharing legs
    car_sharing_legs = legs[legs["mode"] == "Mode::CarsharingMobility"]
    car_sharing_users = car_sharing_legs["participant_id"].unique()

    # legs of car sharing users
    legs_of_carsharing_users = legs[legs["participant_id"].isin(car_sharing_users)]
    print("ratio car sharing users", len(legs_of_carsharing_users) / len(legs))
    print("number car sharing user legs", len(legs_of_carsharing_users))

    # clean: in switzerland and plausible
    legs_model = clean_legs(legs_of_carsharing_users)

    # add geom
    legs_model = add_geom_legs(legs_model)
    print(legs_model.columns)

    # save legs
    legs_model["id"] = np.arange(len(legs_model))
    legs_model.set_index("id", inplace=True)
    legs_model["geom"] = legs_model["geom"].apply(wkt.dumps)
    legs_model.to_csv(os.path.join(out_path, "legs.csv"))

    # 2) Activities
    print("--------- Activities ----------")
    acts = pd.read_csv(os.path.join(INP_PATH_MOBIS, "tracking", "activities.csv"))
    # car sharing users
    acts_of_carsharing_users = acts[acts["participant_id"].isin(car_sharing_users)]
    print("ratio car sharing users", len(acts_of_carsharing_users) / len(acts))
    print("number car sharing user legs", len(acts_of_carsharing_users))
    # clean
    acts_of_carsharing_users = clean_acts(acts_of_carsharing_users)
    # geom
    acts_of_carsharing_users["geom"] = acts_of_carsharing_users.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    acts_of_carsharing_users.drop(["x", "y"], axis=1, inplace=True)
    acts_of_carsharing_users["geom"] = acts_of_carsharing_users["geom"].apply(wkt.dumps)
    acts_of_carsharing_users.to_csv(os.path.join(out_path, "acts.csv"), index=False)


def preprocess_trackintel(inp_path):
    sp = ti.io.file.read_staypoints_csv(os.path.join(inp_path, "acts.csv"), index_col="id", crs="EPSG:2056")
    legs = ti.io.file.read_triplegs_csv(os.path.join(inp_path, "legs.csv"), index_col="id", crs="EPSG:2056")

    # add activity flag
    act_sp = sp.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=15)
    act_sp.loc[act_sp["purpose"] == "wait", "is_activity"] = False
    act_sp.loc[((act_sp["purpose"] == "other") | (act_sp["purpose"] == "unknown")), "is_activity"] = False
    # trips
    sp, legs, trips = ti.preprocessing.triplegs.generate_trips(act_sp, legs, gap_threshold=15, add_geometry=True)
    print(sp.crs, legs.crs, trips.crs)
    trips.crs = sp.crs
    # locations
    sp, locs = ti.preprocessing.staypoints.generate_locations(sp, method="dbscan", epsilon=20, num_samples=1)
    print(sp.crs, locs.crs)
    locs.crs = sp.crs
    # tours
    trips, tours = ti.preprocessing.trips.generate_tours(trips, max_time="1d", max_nr_gaps=1, print_progress=True)

    # save
    ti.io.file.write_staypoints_csv(sp, os.path.join(inp_path, "staypoints.csv"))
    ti.io.file.write_triplegs_csv(legs, os.path.join(inp_path, "triplegs.csv"))
    ti.io.file.write_trips_csv(trips, os.path.join(inp_path, "trips.csv"))
    ti.io.file.write_locations_csv(locs, os.path.join(inp_path, "locations.csv"))
    ti.io.file.write_tours_csv(tours, os.path.join(inp_path, "tours.csv"))


if __name__ == "__main__":
    tracking_data_carsharing_users(OUT_PATH)
    preprocess_trackintel(OUT_PATH)
