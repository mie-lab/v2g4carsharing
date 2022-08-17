import pandas as pd
import numpy as np
import os
import time
import trackintel as ti
import geopandas as gpd
from shapely import wkt
from meteostat import Hourly, Daily
from meteostat import Point as MeteoPoint
from datetime import timedelta


class ModeChoiceFeatures:
    def __init__(self, path="../data/mobis"):
        self.trips = ti.io.file.read_trips_csv(
            os.path.join(path, "trips.csv"), tz="Europe/Amsterdam", index_col="id", geom_col="geom"
        )
        sp = ti.io.file.read_staypoints_csv(os.path.join(path, "staypoints.csv"), index_col="id")
        self.legs = ti.io.file.read_triplegs_csv(
            os.path.join(path, "triplegs.csv"), tz="Europe/Amsterdam", index_col="id", geom_col="geom"
        )

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
                "next_trip_id_destination",
                "trip_id_destination",
                "next_trip_id",
                "is_activity_destination",
                "is_activity",
                "finished_at_destination",
            ],
            axis=1,
            inplace=True,
        )

    def add_purpose_features(
        self, col_name="purpose_destination", included_purposes=["Home", "Leisure", "Work", "Shopping"]
    ):
        included_w_prefix = ["feat_" + col_name + "_" + p for p in included_purposes]
        one_hot = pd.get_dummies(self.trips[col_name], prefix="feat_" + col_name)[included_w_prefix]
        self.trips = self.trips.merge(one_hot, left_index=True, right_index=True)

    def add_distance_feature(self):
        # add distance feature
        dest_geom = gpd.GeoSeries(self.trips["geom_destination"])
        dest_geom.crs = "EPSG:2056"
        start_geom = gpd.GeoSeries(self.trips["geom_origin"])
        start_geom.crs = "EPSG:2056"
        self.trips["feat_distance"] = start_geom.distance(dest_geom)

    def add_pt_accessibility(self, origin_or_destination="origin", pt_path="../data/OeV_Gueteklassen_ARE.gpkg"):
        pt_accessibility = gpd.read_file(pt_path)
        self.trips = gpd.GeoDataFrame(self.trips, geometry="geom_" + origin_or_destination)
        self.trips = self.trips.sjoin(pt_accessibility, how="left")
        self.trips["Klasse"] = self.trips["Klasse"].replace([None, "D", "C", "B", "A"], [0, 1, 2, 3, 4])
        self.trips = self.trips.drop(["index_right"], axis=1).rename(
            columns={"Klasse": "feat_pt_accessibility" + origin_or_destination}
        )

    def add_weather(self):
        def get_daily_weather(row):
            loc = MeteoPoint(row["geom_destination"].y, row["geom_destination"].x)
            end = row["started_at_destination"].replace(tzinfo=None)
            data = Daily(loc, end - timedelta(days=1, minutes=1), end).fetch()
            if len(data) == 1:
                return pd.Series(data.iloc[0])
            else:
                for random_testing in [100, 250, 500]:
                    loc = MeteoPoint(row["geom_destination"].y, row["geom_destination"].x, random_testing)
                    data = Daily(loc, end - timedelta(days=1), end).fetch()
                    if len(data) == 1:
                        return pd.Series(data.iloc[0])
                return pd.NA

        weather_input = self.trips[["started_at_destination", "geom_destination"]].dropna()
        weather_input["geom_destination"] = weather_input["geom_destination"].to_crs("EPSG:4326")
        weather_data = weather_input.apply(get_daily_weather, axis=1)

        weather_data["prcp"] = weather_data["prcp"].fillna(0)
        weather_data.rename(columns={c: "feat_weather_" + c for c in weather_data.columns}, inplace=True)
        self.trips = self.trips.merge(weather_data, how="left", left_index=True, right_index=True)

    def add_dist2station(self, station_path="../v2g4carsharing/data/station.csv", origin_or_destination="origin"):
        # distance to next car sharing station

        # read station data
        station = pd.read_csv(station_path, index_col="station_no")
        station["geom"] = station["geom"].apply(wkt.loads)
        station = gpd.GeoDataFrame(station, geometry="geom")
        station.geometry.crs = "EPSG:4326"
        station["geom"] = station.geometry.to_crs("EPSG:2056")

        self.trips = self.trips.set_geometry(f"geom_{origin_or_destination}")
        self.trips.drop(["index_right", "station_no"], inplace=True, axis=1, errors="ignore")
        # get nearest station no
        self.trips = self.trips.sjoin_nearest(station.reset_index()[["station_no", "geom"]])
        # get corresponding geometry (after self.trips already has the attribute station no)
        self.trips = self.trips.merge(station[["geom"]], how="left", left_on="station_no", right_index=True)
        self.trips[f"feat_{origin_or_destination}_dist2station"] = self.trips["geom"].distance(
            self.trips[f"geom_{origin_or_destination}"]
        )
        # clean
        self.trips.drop("geom", axis=1, inplace=True)

    def add_mode_labels(self, by_variable="length"):
        def ratio_of_modes(trip_rows, by_variable=by_variable):
            """
            ratio_by: {length, duration}
            """
            sum_df = trip_rows.groupby("mode")[by_variable].sum()
            sum_df /= sum(sum_df)
            return pd.DataFrame(sum_df)

        modes = self.legs.groupby("trip_id").apply(ratio_of_modes)
        modes = modes.reset_index().pivot(index="trip_id", columns="mode").fillna(0)[by_variable].reset_index()

        self.trips = self.trips.merge(modes, how="left", left_index=True, right_on="trip_id")

    def add_survey_features(
        self, survey_path, survey_features=["p_birthdate", "p_sex", "oev_accessibility", "p_caraccess"],
    ):
        car_sharing_users = self.trips["user_id"].unique()
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
        output_df["feat_oev_accessibility"] = survey_car_sharing_users["oev_accessibility"].copy()
        self.trips = self.trips.merge(output_df, left_on="user_id", right_on="participant_id")

    def add_time_features(self, origin_or_destination="origin"):
        # TODO: sin cos
        col_name = "started_at_" + origin_or_destination
        self.trips[col_name] = pd.to_datetime(self.trips[col_name])
        self.trips[f"feat_{origin_or_destination}_hour"] = self.trips[col_name].apply(lambda x: x.hour)
        self.trips[f"feat_{origin_or_destination}_day"] = self.trips[col_name].apply(
            lambda x: x.dayofweek if ~pd.isna(x) else print(x)
        )

    def add_all_features(self):
        tic = time.time()
        self.add_distance_feature()
        print(time.time() - tic, "\nAdd purpose:")
        tic = time.time()
        self.add_purpose_features("purpose_destination")
        self.add_purpose_features("purpose")
        print(time.time() - tic, "\nAdd pt accessibility:")
        tic = time.time()
        self.add_pt_accessibility(origin_or_destination="origin")
        self.add_pt_accessibility(origin_or_destination="destination")
        print(time.time() - tic, "\nAdd dist2station:")
        tic = time.time()
        self.add_dist2station(origin_or_destination="origin")
        self.add_dist2station(origin_or_destination="destination")
        print(time.time() - tic, "\nAdd time:")
        tic = time.time()
        self.add_time_features(origin_or_destination="origin")
        self.add_time_features(origin_or_destination="destination")
        print(time.time() - tic, "\nAdd survey features:")
        tic = time.time()
        # warning only comes up when reading the survey data
        self.add_survey_features(
            os.path.join("../../teaching/mobis_project/MOBIS_Covid_version_2_raubal", "tracking/participants.csv"),
            survey_features=["p_birthdate", "p_sex", "oev_accessibility", "p_caraccess"],
        )
        print(time.time() - tic, "\nAdd mode labels:")
        tic = time.time()
        self.add_mode_labels()
        print(time.time() - tic, "\nAdd weather:")
        tic = time.time()
        self.add_weather()
        print(time.time() - tic)

    def save(self, path):
        # remove geom (for more efficient saving)
        out_trips = self.trips.drop([col for col in self.trips.columns if "geom" in col], axis=1)
        out_trips.to_csv(os.path.join(path, "trips_enriched.csv"))

    # -----------------------

    # def add_prev_mode_feature(self): # TODO
    #     """Feature: previous mode"""
    #     dataset = dataset.sort_values(["participant_id", "trip_id"])
    #     # add prev mode feature
    #     dataset["prev_mode"] = dataset["mode"].shift(1)
    #     one_hot_prev_mode = pd.get_dummies(dataset["prev_mode"], prefix="prev_mode")
    #     dataset = dataset.merge(one_hot_prev_mode, left_index=True, right_index=True)
    #     return dataset


if __name__ == "__main__":
    feat_collector = ModeChoiceFeatures()
    feat_collector.add_all_features()
    feat_collector.save("../data/mobis")
    # basic_trip_features = ["length"]
    #
    # time_features = ["started_at_hour", "started_at_day", "finished_at_hour"]  # TODO
    # # only keep important modes
    # drop_feats = [
    #     "prev_mode_Mode::Boat",
    #     "prev_mode_Mode::Cablecar",
    #     "prev_mode_Mode::MotorbikeScooter",
    #     "prev_mode_Mode::RidepoolingPikmi",
    #     "prev_mode_Mode::Ski",
    #     "prev_mode_Mode::TaxiUber",
    # ]
    # prev_mode_features = [c for c in one_hot_prev_mode.columns if c not in drop_feats]  # TODO

    # # TODO: load self.legs dataset
    # dataset = add_survey_features(dataset, survey_features)
    # dataset = add_time_features(dataset)
    # dataset = add_prev_mode_feature(dataset)

    # label_col = "mode"
    # dataset = dataset[survey_features + time_features + basic_trip_features + list(prev_mode_features) + [label_col]]
