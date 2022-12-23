import os
import argparse
from v2g4carsharing.simulate.station_scenarios import *
from v2g4carsharing.import_data.import_utils import write_geodataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default="data")
    parser.add_argument("-o", "--out_path", type=str, default="csv/station_scenario")
    parser.add_argument("-v", "--vehicle_scenario", type=int, default=5000)
    parser.add_argument("-s", "--station_scenario", default="all_stations")
    parser.add_argument("-d", "--sim_date", default="2020-01-01 00:00:00")
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    # Run
    station_veh_scenario = station_scenario(
        station_scenario=args.station_scenario,
        vehicle_scenario=args.vehicle_scenario,
        in_path=in_path,
        sim_date=args.sim_date,
    )

    write_geodataframe(
        station_veh_scenario, os.path.join(out_path, f"scenario_{args.station_scenario}_{args.vehicle_scenario}.csv")
    )

