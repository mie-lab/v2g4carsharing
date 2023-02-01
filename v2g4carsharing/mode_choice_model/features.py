import pandas as pd
import numpy as np
import os
import time
import json
import trackintel as ti
import geopandas as gpd
from shapely import wkt
from datetime import timedelta

from v2g4carsharing.simulate.car_sharing_patterns import load_trips


def compute_dist_to_station(trips, station):
    # delete columns if they already exist
    trips.drop(
        [
            "distance_to_station_origin",
            "closest_station_origin",
            "index_right",
            "distance_to_station_destination",
            "closest_station_destination",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    # get closest station to origin
    trips.set_geometry("geom_origin", inplace=True)
    trips = trips.sjoin_nearest(station[["geom"]], distance_col="distance_to_station_origin")
    trips.rename(columns={"index_right": "closest_station_origin"}, inplace=True)
    # get closest station to destination
    trips.set_geometry("geom_destination", inplace=True)
    trips.crs = "EPSG:2056"
    trips = trips.sjoin_nearest(station[["geom"]], distance_col="distance_to_station_destination")
    trips.rename(columns={"index_right": "closest_station_destination"}, inplace=True)
    return trips


class ModeChoiceFeatures:
    def __init__(self, path="../data/mobis"):
        self.path = path
        self.trips = load_trips(os.path.join(path, "trips_enriched.csv"))
        # check if labels are available, if yes, preprocess (restrict dataset to relevant modes)
        mode_columns = [col for col in self.trips.columns if col.startswith("Mode::")]
        self.mode_avail = len(mode_columns) > 0
        with open("config.json", "r") as infile:
            self.included_modes = json.load(infile)["included_modes"]
        # reduce to included modes
        if self.mode_avail:
            # reduce to included modes
            print("included_modes", self.included_modes, "size of dataset before", self.trips.shape)
            remove_modes = [
                col for col in self.trips.columns if col.startswith("Mode::") and col not in self.included_modes
            ]
            self.trips = (self.trips[self.trips[self.included_modes].sum(axis=1) > 0]).drop(remove_modes, axis=1)
            print("after removing other modes:", self.trips.shape)

    def subsample(self, nr_users_desired=100000):
        persons_sim = self.trips["person_id"].unique()
        rand_inds = np.random.permutation(len(persons_sim))[:nr_users_desired]
        persons_sampled = persons_sim[rand_inds]
        self.trips = self.trips[self.trips["person_id"].isin(persons_sampled)]
        print(f"After subsampling, there are {self.trips['person_id'].nunique()} unique persons in the simulated data")

    def add_purpose_features(
        self, col_name="purpose_destination", included_purposes=["home", "leisure", "work", "shopping", "education"]
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

    def add_prev_mode_feature(self):
        """Feature: previous mode"""
        if self.mode_avail:
            self.trips = self.trips.sort_values(["person_id", "started_at_origin"])
            # add prev mode feature
            self.trips["prev_person"] = self.trips["person_id"].shift(1)
            person_switch = self.trips["prev_person"] != self.trips["person_id"]
            for mode_col in self.included_modes:
                self.trips["feat_prev_" + mode_col] = self.trips[mode_col].shift(1)
                self.trips.loc[person_switch, "feat_prev_" + mode_col] = 0
            self.trips.drop(["prev_person"], axis=1, inplace=True)
        else:
            for mode_col in self.included_modes:
                self.trips["feat_prev_" + mode_col] = 0

    def add_weather(self):
        # import of meteostat here in order to remove the requirement
        from meteostat import Hourly, Daily
        from meteostat import Point as MeteoPoint
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

    def add_dist2station(self, station_path=os.path.join("data", "station.csv")):
        """Compute distance to next car sharing station"""
        # read station data
        station = pd.read_csv(station_path, index_col="station_no")
        # filter for stations that actually appear in our booking data
        station = station[station["in_reservation"]]
        # load geom
        station["geom"] = station["geom"].apply(wkt.loads)
        station = gpd.GeoDataFrame(station, geometry="geom")
        station.geometry.crs = "EPSG:4326"
        station["geom"] = station.geometry.to_crs("EPSG:2056")

        self.trips = compute_dist_to_station(self.trips, station)

        # use as features as well as for car sharing data generation
        for col in ["origin", "destination"]:
            self.trips["feat_distance_to_station_" + col] = self.trips["distance_to_station_" + col].copy()

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
        before_distance_0_removal = len(self.trips)
        self.trips = self.trips[self.trips["feat_distance"] > 0]
        print("Removed distance-0-trips (in %):", 1 - len(self.trips) / before_distance_0_removal)
        print(time.time() - tic, "\nAdd prev mode feature:")
        tic = time.time()
        self.add_prev_mode_feature()
        print(time.time() - tic, "\nAdd purpose:")
        tic = time.time()
        self.add_purpose_features("purpose_destination")
        self.add_purpose_features("purpose_origin")
        print(time.time() - tic, "\nAdd pt accessibility:")
        tic = time.time()
        self.add_pt_accessibility(origin_or_destination="origin")
        self.add_pt_accessibility(origin_or_destination="destination")
        print(time.time() - tic, "\nAdd dist2station:")
        tic = time.time()
        self.add_dist2station()
        print(time.time() - tic, "\nAdd time:")
        tic = time.time()
        self.add_time_features(origin_or_destination="origin")
        self.add_time_features(origin_or_destination="destination")
        print(time.time() - tic)

    def save(self, out_path=None, remove_geom=False):
        # remove geom (for more efficient saving)
        if remove_geom:
            out_trips = self.trips.drop([col for col in self.trips.columns if "geom" in col], axis=1)
        else:
            out_trips = self.trips
        if out_path is None:
            out_path = self.path
        out_trips.to_csv(os.path.join(out_path, "trips_features.csv"))

