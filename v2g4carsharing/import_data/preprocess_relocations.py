import pandas as pd
from v2g4carsharing.import_data.import_utils import to_datetime_bizend

BASE_DATE = pd.to_datetime('2019-01-01 00:00:00.000')
FINAL_DATE = pd.to_datetime("2020-07-31 23:59:59.999")


def prepare_v2b(v2b):
    """
    Prepare vehicle to base input for transfer to relocation dataframe
    Add first row with relocation from Nirwana to station
    """
    v2b_reset = v2b.reset_index().drop("v2b_no", axis=1)
    v2b_reset["bizbeg"] = pd.to_datetime(v2b_reset["bizbeg"])

    v2b_reset["bizend"] = v2b_reset["bizend"].apply(
        lambda x: to_datetime_bizend(x)
    )

    new_lines_dict = []

    for veh_no, veh_df in v2b_reset.groupby("vehicle_no"):
        min_date = veh_df["bizbeg"].min()
        #     print(min_date)
        if min_date > BASE_DATE:
            new_lines_dict.append(
                {
                    "station_no": -1,
                    "vehicle_no": veh_no,
                    "bizbeg": BASE_DATE,
                    "bizend": min_date
                }
            )

    v2b_reset = pd.concat((v2b_reset, pd.DataFrame(new_lines_dict)))
    return v2b_reset


def v2b_to_relocations(v2b):
    # add first and last column
    v2b_reset = prepare_v2b(v2b)

    # Run grouping
    def transform_to_relocations(test):
        test = test.sort_values(["bizbeg", "bizend"])
        # shift
        test["next_bizbeg"] = test["bizbeg"].shift(-1)
        test["end_station_no"] = test["station_no"].shift(-1)

        return test

    relocations = v2b_reset.groupby("vehicle_no"
                                    ).apply(transform_to_relocations)

    # ----- cleaning -----------

    # transform columns
    relocation_columns = {
        "station_no": "start_station_no",
        "bizend": "start_time",
        "next_bizbeg": "end_time"
    }
    relocations = relocations.rename(columns=relocation_columns
                                     )[list(relocation_columns.values()) +
                                       ["end_station_no"]]

    # merge the ones where start and end are the same
    relocations = relocations[
        relocations["start_station_no"] != relocations["end_station_no"]]

    # fill nans in the ned with -1 station
    relocations["end_time"].fillna(FINAL_DATE, inplace=True)
    relocations["end_station_no"].fillna(-1, inplace=True)

    # remove the ones that are out of scope
    relocations = relocations[relocations["start_time"] < FINAL_DATE]

    # set index type
    relocations["end_station_no"] = relocations["end_station_no"].astype(int)

    # add necessary columns for service reservation
    relocations["drive_km"] = 0
    relocations["reservationtype"] = "Servicereservation"

    return relocations.reset_index().drop("level_1", axis=1)
