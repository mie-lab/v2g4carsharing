import os
from v2g4carsharing.trips_preparation.mobis_data_preprocessing import *


if __name__ == "__main__":

    # 1) MOBIS data
    INP_PATH_MOBIS = "../data/mobis/raw_data"
    PATH = "../data/mobis/"
    ONLY_CARSHARING = True

    os.makedirs(PATH, exist_ok=True)
    # select only car sharing users
    tracking_data_carsharing_users(INP_PATH_MOBIS, PATH, reduce_to_carsharing=ONLY_CARSHARING)
    # preprocess data with trackintel
    preprocess_trackintel(PATH)
    # Init preprocessor: merge staypoints and trips
    processor = MobisTripPreprocessor(PATH)
    # add modes and survey features
    processor.add_survey_features(survey_path=os.path.join(INP_PATH_MOBIS, "tracking", "participants.csv"))
    processor.add_mode_labels()
    # dump csv
    processor.save_trips()
