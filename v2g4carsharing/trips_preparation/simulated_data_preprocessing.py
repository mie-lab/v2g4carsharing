from logging import warning
import os
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import shutil

from regex import D


class SimTripProcessor:
    def __init__(
        self, cache_path="../external_repos/ch-zh-synpop/cache_2022/", out_path="../data/simulated_population/sim_2022"
    ):
        """
        Load raw synthetic activities and transform into Geodataframe
        """
        self.cache_path = cache_path
        self.path = out_path

        # copy all relevant files from the cache into the data directory
        self.copy_relevant_files()

        # load geometry data
        with open(os.path.join(self.path, "raw_synthesis_population_spatial_locations.p",), "rb",) as infile:
            geoms = pickle.load(infile).set_index(["person_id", "activity_index"])

        # load activity data
        with open(os.path.join(self.path, "raw_synthesis_population_activities.p",), "rb",) as infile:
            activities = pickle.load(infile).set_index(["person_id", "activity_index"])

        self.acts = activities.merge(geoms, how="left", left_index=True, right_index=True)
        print(
            "Number activities", len(self.acts), "nr unique users", len(self.acts.reset_index()["person_id"].unique()),
        )
        self.acts = gpd.GeoDataFrame(self.acts, geometry="geometry")

        # sort by person and activity_index
        self.acts = self.acts.reset_index().sort_values(["person_id", "activity_index"])
        # assign new IDs for activity identification
        self.acts["id"] = np.arange(len(self.acts))
        self.acts.set_index("id", inplace=True)

    def copy_relevant_files(self):
        data_list = os.listdir(self.cache_path)
        files_to_copy_dict = {}
        for d in data_list:
            is_loc = "synthesis.population.spatial.locations" in d
            is_act = "synthesis.population.activities" in d
            is_survey = "synthesis.population.enriched" in d
            if is_loc or is_act or is_survey:
                new_file_name = ("raw." + d.split("__")[0]).replace(".", "_") + "." + d.split(".")[-1]
                if new_file_name[-1] == "p":
                    # pickled file
                    shutil.copy(os.path.join(self.cache_path, d), os.path.join(self.path, new_file_name))
                else:
                    # cache directory
                    shutil.copytree(
                        os.path.join(self.cache_path, d), os.path.join(self.path, new_file_name), dirs_exist_ok=True
                    )

        print("Files copied successfully")

    def transform_to_trips(self):
        """Transform activities into trips"""
        # get the location ID of the previous row:
        # Attention: location ID is always -1 for the home location of any person
        self.acts["location_id_origin"] = self.acts["destination_id"].shift(1)
        # get the previous geometry of each row
        self.acts["geom_origin"] = self.acts["geometry"].shift(1)
        # get distance that must be travelled to this activity
        self.acts["distance"] = self.acts.distance(self.acts["geom_origin"])
        # correct purpose
        self.acts.loc[self.acts["purpose"] == "shop", ["purpose"]] = "shopping"
        # get purpose of origin
        self.acts["purpose_origin"] = self.acts["purpose"].shift(1)
        # get previous start and end time
        self.acts["start_time_sec_origin"] = self.acts["start_time"].shift(1)
        self.acts["end_time_sec_origin"] = self.acts["end_time"].shift(1)
        # for start time origin, fill in the NaNs with random arrival times at home
        self.acts["on_prev_day"] = pd.isna(self.acts["start_time_sec_origin"])
        evening_arrive_min, evening_arrive_max = [17 * 60 * 60, 23.9 * 60 * 60]
        rand_arrival_time = (
            np.random.rand(sum(self.acts["on_prev_day"])) * (evening_arrive_max - evening_arrive_min)
            + evening_arrive_min
        ).astype(int)
        self.acts.loc[self.acts["on_prev_day"], "start_time_sec_origin"] = rand_arrival_time

        self.acts = self.acts.rename(
            columns={
                "geometry": "geom_destination",
                "purpose": "purpose_destination",
                "start_time": "start_time_sec_destination",
                "end_time": "end_time_sec_destination",
                "destination_id": "location_id_destination",
            }
        ).drop("duration", axis=1)

        # drop first activity --> trips
        self.trips = self.acts.dropna(subset="start_time_sec_destination")

    def add_simulated_datetimes(self, simulated_date):
        if not hasattr(self, "trips"):
            print("ERROR: Trips must first be created with transform_to_trips function!")
            return None

        # get following date because some activities end in following night
        if int(simulated_date[-2:]) > 29 or int(simulated_date[-2:]) < 1:
            raise RuntimeError(
                "Simulated date is 30th day of month or higher. this causes error with the subsequent date."
            )
        simulated_date_next = simulated_date[:-2] + str(int(simulated_date[-2:]) + 1)
        simulated_date_prev = simulated_date[:-2] + str(int(simulated_date[-2:]) - 1).zfill(2)

        for time_var in ["start_time_sec_origin", "start_time_sec_destination", "end_time_sec_origin"]:
            assert not any(pd.isna(self.trips[time_var])), f"NaNs in {time_var}"
            if time_var.startswith("start"):
                new_time_var_name = "started_at_" + time_var.split("_")[-1]
            else:
                new_time_var_name = "finished_at_" + time_var.split("_")[-1]
            # get seconds representation of time
            sec = self.trips[time_var]
            # transform into hours, minutes and seconds
            hours_float = sec // 3600
            on_prev_day = self.trips["on_prev_day"]
            on_next_day = hours_float >= 24
            on_this_day = (~on_next_day) & (~on_prev_day)
            hours = (hours_float % 24).astype(int).astype(str).str.zfill(2)
            minutes = ((sec // 60) % 60).astype(int).astype(str).str.zfill(2)
            seconds = (sec % 60).astype(int).astype(str).str.zfill(2)
            # add as a date string
            # 1) this day
            self.trips.loc[on_this_day, new_time_var_name] = (
                simulated_date + " " + hours[on_this_day] + ":" + minutes[on_this_day] + ":" + seconds[on_this_day]
            )
            # 2) next day
            self.trips.loc[on_next_day, new_time_var_name] = (
                simulated_date_next + " " + hours[on_next_day] + ":" + minutes[on_next_day] + ":" + seconds[on_next_day]
            )
            # 3) prev day
            self.trips.loc[on_prev_day, new_time_var_name] = (
                simulated_date_prev + " " + hours[on_prev_day] + ":" + minutes[on_prev_day] + ":" + seconds[on_prev_day]
            )
            # to datetime
            self.trips[new_time_var_name] = pd.to_datetime(self.trips[new_time_var_name])
            print("Successfully transformed time variable into date:", time_var)

    def add_survey_features(
        self, survey_features=["feat_caraccess", "feat_sex", "feat_age", "feat_ga", "feat_halbtax", "feat_employed"]
    ):
        """Add user-related information to the trips"""
        # load activity data
        with open(os.path.join(self.path, "raw_synthesis_population_enriched.p",), "rb",) as infile:
            survey = pickle.load(infile).set_index("person_id")
        survey["feat_caraccess"] = survey["car_availability"] / 2
        survey.rename(
            columns={
                "sex": "feat_sex",
                "age": "feat_age",
                "subscriptions_ga": "feat_ga",
                "subscriptions_halbtax": "feat_halbtax",
                "employed": "feat_employed",
            },
            inplace=True,
        )
        self.trips = self.trips.merge(survey[survey_features], left_on="person_id", right_index=True, how="left")

    def save_trips(self):
        if not hasattr(self, "trips"):
            print("ERROR: Trips must first be created with transform_to_trips function!")
        else:
            self.trips.to_csv(os.path.join(self.path, "trips_enriched.csv"))
