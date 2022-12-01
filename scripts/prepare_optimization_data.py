import os
import argparse
import warnings
from datetime import timedelta

warnings.filterwarnings(action="ignore")

from v2g4carsharing.optimization_data.data_io import load_ev_data
from v2g4carsharing.optimization_data.utils import (
    ts_to_index,
    index_to_ts,
    BASE_DATE,
    FINAL_DATE,
)
from v2g4carsharing.optimization_data.compute_soc import get_matrices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default="data", help="path to input data")
    parser.add_argument(
        "-o", "--out_path", type=str, default=os.path.join("outputs", "input_matrices"), help="path to save output"
    )
    parser.add_argument("-s", "--scenario", type=str, default="scenario_1", help="EV scenario name")
    parser.add_argument("-t", "--time_granularity", type=float, default=0.25, help="discretization in hours")
    parser.add_argument("--simulate", action="store_true", help="Whether the data is simulated reservations")
    args = parser.parse_args()

    # specify in and out paths here
    inp_path = args.in_path
    out_path = args.out_path
    time_granularity = args.time_granularity  # in reference to one hour, e.g. 0.5 = half an h
    sim_ev_mode = args.scenario

    if args.simulate:
        # just simulate one day
        overall_slots = ts_to_index(BASE_DATE + timedelta(days=2), time_granularity=time_granularity) + 1
    else:
        overall_slots = ts_to_index(FINAL_DATE, time_granularity=time_granularity) + 1
    os.makedirs(out_path, exist_ok=True)

    # Load data
    ev_reservation = load_ev_data(inp_path, sim_ev_mode=sim_ev_mode, simulate=args.simulate)

    # columns of resulting csv files
    columns = [
        index_to_ts(index, time_granularity=time_granularity, base_date=BASE_DATE) for index in range(overall_slots)
    ]

    # Run
    (station_matrix, reservation_matrix, required_soc) = get_matrices(
        ev_reservation, columns, overall_slots, time_granularity
    )

    # Save
    for matrix, name in zip(
        [station_matrix, required_soc, reservation_matrix], ["station_matrix", "soc_matrix", "reservation_matrix"]
    ):
        matrix.to_csv(os.path.join(out_path, f"{name}.csv"))
    # save ev specifications (models also for simulated EVs)
    ev_specs = ev_reservation.groupby("vehicle_no").agg(
        {key: "first" for key in ["model_name", "brand_name", "charge_power", "battery_capacity", "range"]}
    )
    ev_specs.to_csv(os.path.join(out_path, "ev_specifications.csv"), index=True)
