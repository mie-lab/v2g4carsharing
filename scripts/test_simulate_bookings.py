import argparse
import os
import pickle
from v2g4carsharing.simulate.car_sharing_patterns import *
from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_path_sim_trips",
        type=str,
        default=os.path.join("..", "data", "simulated_population", "sim_2022"),
        help="path to simulated trips csv",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=os.path.join("outputs", "simulated_car_sharing", "testing"),
        help="path to save output",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=os.path.join("trained_models", "rf_test.p"),
        help="path to mode choice model",
    )
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()

    in_path_sim_trips = args.in_path_sim_trips
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # load activities and shared-cars availability
    acts_gdf = pd.read_csv(os.path.join(in_path_sim_trips, "trips_features.csv"))

    # define mode choice model
    # mode_choice_model = simple_mode_choice
    with open(args.model_path, "rb") as infile:
        mode_choice_model = pickle.load(infile)

    # sort
    acts_gdf.sort_values(["started_at_destination"], inplace=True)

    # # debugging
    # for i, row in acts_gdf.iterrows():
    #     out = mode_choice_model(row)
    # apply model
    inp_rf = np.array(acts_gdf[mode_choice_model.feat_columns])
    pred = mode_choice_model.rf.predict(inp_rf)
    pred_str = np.array(mode_choice_model.label_meanings)[pred]
    # add mode labels and make fake vehicle IDs
    acts_gdf["mode"] = pred_str
    acts_gdf["vehicle_no"] = -1
    acts_gdf["mode_decision_time"] = acts_gdf["start_time_sec_destination"]

    uni, counts = np.unique(pred_str, return_counts=True)
    print({u: c for u, c in zip(uni, counts)})

    # Save trip modes
    acts_gdf[["person_id", "activity_index", "mode"]].to_csv(os.path.join(out_path, "sim_modes.csv"), index=False)

    # compare dist
    sim_reservations = derive_reservations(acts_gdf)

    # Save reservations
    sim_reservations.to_csv(os.path.join(out_path, "sim_reservations.csv"))

