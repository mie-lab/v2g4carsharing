import numpy as np
import pickle
import json
import os
import pandas as pd

base_path_sim = os.path.join("outputs", "simulated_car_sharing")
# ../../v2g4carsharing/outputs/simulated_car_sharing/
base_path_model = os.path.join("outputs", "mode_choice_model")
# ../../v2g4carsharing/outputs/mode_choice_model/

res = []
for mode in ["irl_2019_final", "xgb_2019_final"]:
# ["irl_2019_sim_prevmode", "irl_2019_sim_noprevmode", "xgb_2019_sim_prevmode", "xgb_2019_sim_noprevmode"]:
    path = os.path.join(base_path_sim, mode, "evaluation.txt")
    with open(path, "r") as infile:
        text = infile.readlines()

    res_dict = {
        "Model": mode,
        "reservation_z": float(text[2].split(" ")[-1][:-4]),
        "user_frequency": float(text[4].split(" ")[-1][:-4]),
        "user_frequency_z": float(text[5].split(" ")[-1][:-4]),
        "Duration (Wasserstein)": float(text[7].split(" ")[-1][:-4]),
        "Driven km (Wasserstein)": float(text[13].split(" ")[-1][:-4]),
        "Start time (Wasserstein)": float(text[9].split(" ")[-1][:-4]),
        "End time (Wasserstein)": float(text[11].split(" ")[-1][:-4]),
        "Avg. abs. station z-score": float(text[15].split(" ")[-1][:-4]),
        "Avg. mode share ratio": float(text[18].split(" ")[-1][:-4]),
    }

    prevmode_str = "_model" if "noprevmode" in mode else "_prevmode"
    if "irl" in mode:
        path = os.path.join(base_path_model, f"irl{prevmode_str}", "evaluation.txt")
        with open(path, "r") as infile:
            text_acc = infile.readlines()
            res_dict["Accuracy"] = float(text_acc[-2][5:-5])
            res_dict["Balanced Acc."] = float(text_acc[-1][14:-5])
    else:
        path = os.path.join(base_path_model, f"xgb{prevmode_str}", "stdout_random_forest.txt")
        with open(path, "r") as infile:
            text_acc = infile.readlines()
            acc = [float(elem[5:-5]) for elem in text_acc if elem[:4] == "Acc:"]
            bal_acc = [float(elem[14:-5]) for elem in text_acc if elem[:4] == "Bala"]
            res_dict["Accuracy"] = acc[0]
            res_dict["Balanced Acc."] = bal_acc[0]
            res_dict["Acc (train)"] = acc[1]
            res_dict["Bal. Acc (train)"] = bal_acc[1]

    res.append(res_dict)
pd.DataFrame(res).to_csv(os.path.join("outputs", "model_comparison_final.csv"))
