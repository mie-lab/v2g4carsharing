import os
import argparse
from v2g4carsharing.simulate.station_scenarios import *
from v2g4carsharing.import_data.import_utils import write_geodataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default="data")
    parser.add_argument("-o", "--out_path", type=str, default="csv/station_scenario")
    parser.add_argument("-v", "--vehicle_scenario", type=int, default=5000)
    parser.add_argument("-s", "--station_scenario", default="all")
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

    # #### Test generation of staions ####
    # # load station scenario, convert to geodataframe and extract the x and y coordinates
    # current_stations = pd.read_csv("csv/station_scenario/scenario_all_3500.csv")
    # current_stations["geom"] = current_stations["geom"].apply(wkt.loads)
    # current_stations = gpd.GeoDataFrame(current_stations, geometry="geom")
    # current_stations["x"] = current_stations["geom"].x
    # current_stations["y"] = current_stations["geom"].y
    # station_locations = np.array(current_stations[["x", "y"]])  # locations of stations
    # new_stations_gdf = place_new_stations(500, station_locations, subsample_population=100000)
    # from shapely import wkt
    # new_stations_gdf["geometry"] = new_stations_gdf["geometry"].apply(wkt.dumps)
    # new_stations_gdf.to_csv("csv/station_scenario/new_stations_500.csv")
