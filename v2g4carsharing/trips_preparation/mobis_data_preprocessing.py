import pandas as pd
import trackintel as ti
import os
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString


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
        [
            "treatment",
            "phase",
            "started_at_tz",
            "finished_at_tz",
            "in_switzerland",
            "was_confirmed",
            "labelled_purpose",
            "type",
        ],
        inplace=True,
        axis=1,
    )
    # rename
    acts_of_carsharing_users.rename(
        columns={"participant_id": "user_id", "activity_id": "id", "imputed_purpose": "purpose"}, inplace=True
    )
    return acts_of_carsharing_users


def tracking_data_carsharing_users(inp_path_mobis, out_path):
    print("------ Loading Legs ---------")
    legs = pd.read_csv(os.path.join(inp_path_mobis, "tracking", "legs.csv"), index_col="leg_id")
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
    acts = pd.read_csv(os.path.join(inp_path_mobis, "tracking", "activities.csv"))
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


class MobisTripPreprocessor:
    def __init__(self, path="../data/mobis/"):
        """
        Load acts, trips and triplegs and join into one big trips dataframe
        """
        self.path = path
        # Load trips, staypoints and legs
        self.trips = ti.io.file.read_trips_csv(
            os.path.join(path, "trips.csv"), tz="Europe/Amsterdam", index_col="id", geom_col="geom"
        )
        sp = ti.io.file.read_staypoints_csv(os.path.join(path, "staypoints.csv"), index_col="id")
        self.legs = ti.io.file.read_triplegs_csv(
            os.path.join(path, "triplegs.csv"), tz="Europe/Amsterdam", index_col="id", geom_col="geom"
        )
        # merge staypoints into trips
        self.trips = self.trips.merge(
            sp, how="left", left_on="origin_staypoint_id", right_index=True, suffixes=("", "_origin")
        )
        self.trips = self.trips.merge(
            sp, how="left", left_on="destination_staypoint_id", right_index=True, suffixes=(None, "_destination")
        )
        # delete unnecessary columns
        self.trips.drop(
            [
                "geom",
                "user_id_origin",
                "user_id_destination",
                "trip_id",
                "next_trip_id_destination",
                "trip_id_destination",
                "next_trip_id",
                "prev_trip_id",
                "is_activity_destination",
                "is_activity",
                "finished_at_destination",
                "origin_staypoint_id",
                "destination_staypoint_id",
                "prev_trip_id_destination",
                "location_id",
                "location_id_destination",
                "started_at",  # start of trip! info not available for synthetic
                "finished_at",
                "finished_at_destination",
                "duration",  # duration features are durations of the activities, not of the trip
                "duration_destination",
            ],
            axis=1,
            inplace=True,
        )
        # rename columns for clearness
        self.trips.rename(columns={"user_id": "person_id", "purpose": "purpose_origin"}, inplace=True)
        self.trips["purpose_origin"] = self.trips["purpose_origin"].str.lower()
        self.trips["purpose_destination"] = self.trips["purpose_destination"].str.lower()
        print("Loaded trips.")

    def add_survey_features(
        self, survey_path, survey_features=["p_birthdate", "p_sex", "p_caraccess"],
    ):
        print("Adding survey features...")
        car_sharing_users = self.trips["person_id"].unique()
        # add survey data
        survey = pd.read_csv(survey_path).set_index(["participant_id"])
        # get car sharing users
        survey_car_sharing_users = (survey.loc[car_sharing_users])[survey_features]
        # show all columns --> could keep p_language for example etc
        # list(survey_car_sharing_users.columns)

        # only keep the first entry for each participant (removing duplicates in survey)
        survey_car_sharing_users = survey_car_sharing_users.reset_index().groupby("participant_id").agg("first")

        # preprocess survey columns
        access_values = {"noaccess": 0, "after_consultation": 0.5, "always": 1}
        output_df = pd.DataFrame(index=survey_car_sharing_users.index)
        output_df["feat_age"] = 2022 - survey_car_sharing_users["p_birthdate"]
        output_df["feat_sex"] = survey_car_sharing_users["p_sex"].apply(lambda x: 0 if x == "male" else 1)
        output_df["feat_caraccess"] = survey_car_sharing_users["p_caraccess"].map(access_values)
        # output_df["feat_oev_accessibility"] = survey_car_sharing_users["oev_accessibility"].copy()
        # TODO: possibly add more features, such as oev accessibility by home coordinates, or ga / halbtax (also
        # available in synpp)
        self.trips = (
            self.trips.reset_index()
            .merge(output_df, how="left", left_on="person_id", right_on="participant_id")
            .set_index("id")
        )
        print("Finished adding survey")

    def add_mode_labels(self, mode_ratio_variable="length"):
        """ 
        mode_ratio_variable: str {length, duration}
            # TODO: currently duration doesn't exist
            The variable by which to aggregate the modes: If length, then they are aggregated by distance,
            e.g. 9 car, 1km walking would yield 0.9 car and 0.1 walking
        """
        print("Adding modes...")

        # add modes because for this we need the triplegs
        def ratio_of_modes(trip_rows, by_variable=mode_ratio_variable):
            """
            ratio_by: {length, duration}
            """
            sum_df = trip_rows.groupby("mode")[by_variable].sum()
            sum_df /= sum(sum_df)
            return pd.DataFrame(sum_df)

        modes = self.legs.groupby("trip_id").apply(ratio_of_modes)
        modes = modes.reset_index().pivot(index="trip_id", columns="mode").fillna(0)[mode_ratio_variable].reset_index()
        self.trips = self.trips.merge(modes, how="left", left_index=True, right_on="trip_id").drop("trip_id", axis=1)
        self.trips.index.name = "id"  # rename from trip_id to id
        print("Finished adding modes")

    def save_trips(self):
        self.trips.to_csv(os.path.join(self.path, "trips_enriched.csv"))
