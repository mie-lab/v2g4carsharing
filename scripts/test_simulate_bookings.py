import argparse
import os
import pickle
from v2g4carsharing.simulate.car_sharing_patterns import *
from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot
from v2g4carsharing.mode_choice_model.irl_wrapper import IRLWrapper

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_path_sim_trips",
        type=str,
        default=os.path.join("..", "data", "simulated_population", "sim_2019"),
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
    parser.add_argument(
        "-t", "--model_type", type=str, default="rf", help="one of rf or irl",
    )
    # path to use for postgis_json_path argument: "../../dblogin_mielab.json"
    args = parser.parse_args()

    in_path_sim_trips = args.in_path_sim_trips
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # load activities and shared-cars availability
    acts_gdf = pd.read_csv(os.path.join(in_path_sim_trips, "trips_features.csv"))

    # sort
    acts_gdf.sort_values(["started_at_destination"], inplace=True)
    acts_gdf = acts_gdf[acts_gdf["distance"] > 0]
    acts_gdf["feat_prev_Mode::Car"] = 1  # for the prevmode feature

    if args.model_type == "rf":
        # define mode choice model
        with open(args.model_path, "rb") as infile:
            mode_choice_model = pickle.load(infile)
        # # debugging
        # for i, row in acts_gdf.iterrows():
        #     out = mode_choice_model(row)
        # apply model
        inp_rf = np.array(acts_gdf[mode_choice_model.feat_columns])
        pred = mode_choice_model.rf.predict(inp_rf)
        mode_sim = np.array(mode_choice_model.label_meanings)[pred]
    elif args.model_type == "logistic":
        with open(args.model_path, "rb") as infile:
            mode_choice_model = pickle.load(infile)
        inp_rf = np.array(acts_gdf[mode_choice_model.feat_columns])
        mode_sim = mode_choice_model.model.predict(inp_rf)
    elif args.model_type == "irl":
        mode_choice_model = IRLWrapper(model_path=args.model_path)
        # fast version for testing: pass all at once in an array
        feature_vec = np.array(acts_gdf[mode_choice_model.feat_columns]).astype(float)
        feature_vec = (feature_vec - np.array(mode_choice_model.feat_mean)) / np.array(mode_choice_model.feat_std)
        action_probs = mode_choice_model.policy.predict_probs(feature_vec)
        # note to my future self: must use random.choice instead of argmax (greedy aproach)! Mainly because of entropy
        mode_sim = [
            np.random.choice(mode_choice_model.included_modes, p=action_probs[i]) for i in range(len(action_probs))
        ]
    else:
        raise NotImplementedError("model type must be one of irl or rf")

    # add mode labels and make fake vehicle IDs
    acts_gdf["mode"] = mode_sim
    acts_gdf["vehicle_no"] = -1
    acts_gdf["mode_decision_time"] = acts_gdf["start_time_sec_destination"]

    uni, counts = np.unique(mode_sim, return_counts=True)
    print({u: c for u, c in zip(uni, counts)})

    # Save trip modes
    acts_gdf[["person_id", "activity_index", "mode"]].to_csv(os.path.join(out_path, "sim_modes.csv"), index=False)

    # compare dist
    acts_gdf["start_station_no"] = acts_gdf["closest_station_origin"]
    acts_gdf["end_station_no"] = acts_gdf["closest_station_destination"]
    sim_reservations = derive_reservations(acts_gdf)

    # Save reservations
    sim_reservations.to_csv(os.path.join(out_path, "sim_reservations.csv"))

