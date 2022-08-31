from v2g4carsharing.trips_preparation.mobis_data_preprocessing import *
from v2g4carsharing.trips_preparation.simulated_data_preprocessing import SimTripProcessor


if __name__ == "__main__":

    # 1) MOBIS data
    INP_PATH_MOBIS = "../data/mobis/raw_data"
    PATH = "../data/mobis/"

    os.makedirs(PATH, exist_ok=True)
    # select only car sharing users
    tracking_data_carsharing_users(INP_PATH_MOBIS, PATH)
    # preprocess data with trackintel
    preprocess_trackintel(PATH)
    # Init preprocessor: merge staypoints and trips
    processor = MobisTripPreprocessor(PATH)
    # add modes and survey features
    processor.add_mode_labels()
    processor.add_survey_features(survey_path=os.path.join(INP_PATH_MOBIS, "tracking", "participants.csv"))
    # dump csv
    processor.save_trips()

    # 2) Simulated data
    CACHE_PATH = "../external_repos/ch-zh-synpop/cache_2022/"
    OUT_PATH = "../data/simulated_population/sim_2022"
    SIM_DATE = "2020-01-20"
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
