import numpy as np
import pandas as pd


from v2g4carsharing.optimization_data.utils import FINAL_DATE, ts_to_index


def get_matrices_per_vehicle(
    vehice_df, overall_slots, time_granularity, *args
):
    """
    Main function to produce station and SOC matrices
    Takes a list of reservations as input, as well as ev specifications
    Outputs discrete charging times and states at granularity time_granularity

    mode: min_soc_drive saves only the minimum soc (in kWh) for the next
        reservation
    """
    # sort
    vehice_df = vehice_df.sort_values(
        ["start_time", "end_time"], ascending=False
    )

    # init outputs
    station_matrix = np.zeros(overall_slots)
    reservation_matrix = np.zeros(overall_slots)
    required_soc = np.zeros(overall_slots)

    # aux variables
    next_booking_time = FINAL_DATE
    # veh_soc_next_booking = np.zeros(num_vehicles)
    next_station = -1

    # get vehicle specifications
    battery_capacity = vehice_df["battery_capacity"].dropna().unique()
    assert len(battery_capacity) == 1
    battery_range = vehice_df["range"].dropna().unique()
    assert len(battery_range) == 1
    consumption_per_km = battery_capacity[0] / battery_range[0]

    for res_no, row in vehice_df.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        start_station = row["start_station_no"]
        end_station = row["end_station_no"]

        # get start and end booking indices
        next_booking_index = ts_to_index(
            next_booking_time, time_granularity=time_granularity
        )
        start_booking_index = ts_to_index(
            start_time, time_granularity=time_granularity
        )
        end_booking_index = ts_to_index(
            end_time, time_granularity=time_granularity
        )

        # assert start <= end
        assert (
            start_booking_index <= end_booking_index
        ), f"start > end for res no {row}"
        if start_booking_index == end_booking_index and row[
            "reservationtype"] != "Servicereservation":
            print(
                "Warning: start and end at same time slot - res_no left out",
                res_no
            )

        # assert next station is equal to station at end of current trip
        if end_station != next_station and next_station != -1:
            print(
                f"Station mismatch for res no {res_no}, veh {row['vehicle_no']}"
            )

        # assert that end is not greater than next booking start
        if end_booking_index > next_booking_index:
            if end_booking_index < overall_slots:
                print(f"end booking > start of next for res no {res_no}")
                end_booking_index = next_booking_index
                assert end_booking_index >= start_booking_index
            else:
                end_booking_index = overall_slots

        # update next booking time and station
        next_booking_time = start_time
        next_station = start_station

        # fill matrices
        station_matrix[end_booking_index:next_booking_index] = end_station
        reservation_matrix[start_booking_index:end_booking_index] = res_no
        if row["reservationtype"] != "Servicereservation":
            needed_soc_for_trip = consumption_per_km * row["drive_km"]
            required_soc[start_booking_index] = min(
                [needed_soc_for_trip, battery_capacity[0]]
            )

    # fill the station assignment until the first booking
    station_matrix[:start_booking_index] = next_station
    return station_matrix, reservation_matrix, required_soc


def get_matrices(ev_reservation, columns, overall_slots, time_granularity):
    output_matrices = ev_reservation.groupby("vehicle_no").apply(
        lambda x: get_matrices_per_vehicle(x, overall_slots, time_granularity)
    )
    index = output_matrices.index
    station_matrix = pd.DataFrame(
        np.array([e[0] for e in output_matrices]),
        index=index,
        columns=columns
    )
    reservation_matrix = pd.DataFrame(
        np.array([e[1] for e in output_matrices]),
        index=index,
        columns=columns
    )
    required_soc_matrix = pd.DataFrame(
        np.array([e[2] for e in output_matrices]),
        index=index,
        columns=columns
    )
    return station_matrix, reservation_matrix, required_soc_matrix
