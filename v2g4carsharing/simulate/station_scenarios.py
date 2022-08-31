import os
import pandas as pd
import numpy as np

from v2g4carsharing.simulate.car_sharing_patterns import load_stations
from v2g4carsharing.import_data.import_utils import write_geodataframe


def simple_vehicle_station_scenario(in_path, out_path=os.path.join("csv", "station_scenario")):
    # Current rationale: maximum capacity of mobility: max no of cars that have been at a station at the same time
    # TODO: this should be moved into new file. it is only here for backup
    # TODO: CAREFUL: the vehicle IDs can appear in the vehicle_list for multiple stations! tow vehicles can therefore
    # be in use at the same time. this must be prevented in future scenarios.
    # simply check each month:
    def get_all_dates():
        date_template = "20JJ-MM-02"
        all_dates = []
        for JJ in [19, 20]:
            for MM in range(1, 13):
                if JJ == 2020 and MM > 7:
                    break
                date = date_template.replace("JJ", str(JJ).zfill(2))
                date = date.replace("MM", str(MM).zfill(2))
                all_dates.append(date)
        return all_dates

    def in_date_range(one_station, date):
        beg_before = one_station["bizbeg"] < date
        end_after = one_station["bizend"] > date
        return one_station[beg_before & end_after]

    def get_max_veh(one_station):
        nr_veh = []
        for date in all_dates:
            stations_in_date_range = in_date_range(one_station, date)
            # print(len(test_station[beg_before & end_after]), len(test_station))
            nr_veh.append(len(stations_in_date_range))
        most_veh = np.argmax(nr_veh)
        veh_list = in_date_range(one_station, all_dates[most_veh])["vehicle_no"].unique()
        return list(veh_list)

    all_dates = get_all_dates()

    # load veh2base and station
    veh2base = pd.read_csv(os.path.join(in_path, "vehicle_to_base.csv"))
    station_df = load_stations(in_path)

    # filter for stations that are in the reservations
    in_reservation = station_df[station_df["in_reservation"]]
    veh2base = veh2base[veh2base["station_no"].isin(in_reservation.index)]

    # get maximum available vehicles per station
    veh_per_station = veh2base.groupby("station_no").apply(get_max_veh)
    veh_per_station = pd.DataFrame(veh_per_station, columns=["vehicle_list"])

    # merge with geometry
    veh_per_station = station_df[["geom"]].merge(veh_per_station, left_index=True, right_index=True, how="right")

    write_geodataframe(veh_per_station, os.path.join(out_path, "same_stations.csv"))
