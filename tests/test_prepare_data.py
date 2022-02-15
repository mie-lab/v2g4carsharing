import os
import pandas as pd
import numpy as np
import pytest


class TestPrepareData:

    def test_compute_soc(self):
        data_path = "data"
        matrix_path = os.path.join("outputs", "input_matrices_v4")

        # read data
        reservation = pd.read_csv(
            os.path.join(data_path, "reservation.csv"),
            index_col="reservation_no"
        )
        relocation = pd.read_csv(
            os.path.join(data_path, "relocation.csv"),
            index_col="relocation_no"
        )
        vehicle = pd.read_csv(
            os.path.join(data_path, "vehicle.csv"), index_col="vehicle_no"
        )
        ev_models = pd.read_csv(os.path.join("csv", "ev_models.csv")
                                ).set_index(["brand_name", "model_name"])
        # restrict to EV
        reservation = reservation[reservation["energytypegroup"] == "Electro"]

        # read outputs
        station_matrix = pd.read_csv(
            os.path.join(matrix_path, "station_matrix.csv"),
            index_col="vehicle_no"
        )
        soc_matrix = pd.read_csv(
            os.path.join(matrix_path, "soc_matrix.csv"),
            index_col="vehicle_no"
        )
        reservation_matrix = pd.read_csv(
            os.path.join(matrix_path, "reservation_matrix.csv"),
            index_col="vehicle_no"
        )

        time_columns = np.array(list(station_matrix.columns))

        # Main test loop
        for veh_no, veh_df in reservation.groupby("vehicle_no"):
            # print("Vehicle", veh_no)
            station_veh = np.array(station_matrix.loc[veh_no])
            soc_veh = np.array(soc_matrix.loc[veh_no])
            res_veh = np.array(reservation_matrix.loc[veh_no])

            veh_row = vehicle.loc[veh_no]
            brand_model = (veh_row["brand_name"], veh_row["model_name"])
            veh_specifications = ev_models.loc[brand_model]

            for res_no, row in veh_df.iterrows():
                if pd.isna(row["drive_km"]) or row["drive_km"] == 0:
                    continue

                slots_of_reservation = res_veh == res_no

                station_vals = station_veh[slots_of_reservation]

                assert np.all(station_vals == 0)

                soc_vals = soc_veh[slots_of_reservation]
                if len(soc_vals) == 0:
                    print(row)
                    print(station_vals)

                required_for_drive = row["drive_km"] / veh_specifications[
                    "range"] * veh_specifications["battery_capacity"]

                soc_required = soc_vals[0]
                assert (
                    soc_required > required_for_drive - .0001
                    or soc_required == veh_specifications["battery_capacity"]
                )

                # check stations in adjacent columns
                slot_inds = np.where(slots_of_reservation)[0]
                ind_bef = slot_inds[0] - 1
                ind_after = slot_inds[-1] + 1
                if ind_bef < 0 or ind_after > len(time_columns) - 1:
                    continue
                if station_veh[ind_bef] != row["start_station_no"
                                               ] and station_veh[ind_bef] != 0:
                    relocs_veh = relocation[relocation["vehicle_no"] == veh_no
                                            ]["start_time"]
                    assert any(relocs_veh == row["reservationfrom"])
                assert station_veh[ind_after] == row[
                    "end_station_no"] or station_veh[ind_after
                                                     ] == 0, "error after"
