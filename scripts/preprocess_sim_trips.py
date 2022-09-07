import os
import argparse
from v2g4carsharing.trips_preparation.simulated_data_preprocessing import SimTripProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--cache_path",
        type=str,
        default=os.path.join("..", "external_repos", "ch-zh-synpop", "cache"),
        help="path to cache with synpp data",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=os.path.join("..", "data", "simulated_population", "sim_2019"),
        help="path to save the preprocessed data",
    )
    parser.add_argument(
        "-d", "--sim_date", type=str, default="2019-07-17", help="Date to simulate",
    )
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()

    # Preprocess Simulated data
    CACHE_PATH = args.cache_path
    OUT_PATH = args.out_path
    SIM_DATE = args.sim_date
    os.makedirs(OUT_PATH, exist_ok=True)

    processor = SimTripProcessor(CACHE_PATH, OUT_PATH)
    # transform subsequent activities into trips
    processor.transform_to_trips()
    # transform start and end time from seconds-representation to datetimes
    processor.add_simulated_datetimes(SIM_DATE)
    # add features from population enriched
    processor.add_survey_features()
    # save as csv
    processor.save_trips()
