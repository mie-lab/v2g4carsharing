# preprocess mobis data
python scripts/preprocess_mobis_trips.py
# featurize mobis data
python scripts/featurize_trips.py -i "../data/mobis"
# train mode choice model on mobis data --> set name
python scripts/train_mode_choice_mlp.py -i "../data/mobis/trips_features.csv" -o "outputs/mode_choice_model" -s "test_model_name"

# preprocess the simulated trips
python scripts/preprocess_sim_trips.py -i "../external_repos/ch-zh-synpop/cache" -o "../data/simulated_population/sim_2019" -d "2019-07-17" 
# featurize and output the feature comparison csv
python scripts/featurize_trips.py -i "../data/simulated_population/sim_2019" --keep_geom  -r 100000

# testing the model in simple non-sequential scenario
python scripts/test_simulate_bookings.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/test_2019_xgb" -m "trained_models/xgb_model.p" -t "rf"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/test_2019_xgb"

# testing for irl mode choice model
python scripts/test_simulate_bookings.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/test_2019_irl" -t "irl" -m "../external_repos/guided-cost-learning/trained_models/best_model/model"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/test_2019_irl"

# generate station scenario (so far only simple scenario, no arguments)
python scripts/generate_station_scenario.py 
# generate realistic booking data
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/xgb_2019_sim" -m "trained_models/xgb_model.p"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/xgb_2019_sim"

# generate realistic booking data for irl
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/irl_2019_sim" -t "irl" -m "../external_repos/guided-cost-learning/trained_models/best_model/model"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/irl_2019_sim"


# FROM RAW TRIP DATA TO SCENARIO IN MATRIX FORM:
python scripts/featurize_trips.py -i "../data/simulated_population/sim_2030" --keep_geom  -r 100000 -o "../data/simulated_population/sim_2030/scenario_100k"
python scripts/featurize_trips.py -i "../data/simulated_population/sim_2030" --keep_geom  -r 250000 -o "../data/simulated_population/sim_2030/scenario_250k"
# simulte 250k scenario with 3500 vehicles
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2030/scenario_250k" -o "outputs/simulated_car_sharing/scenario_2030_xgbprevmode_u250k_all_3500" -m "trained_models/xgb_prevmode.p" --station_scenario csv/station_scenario/scenario_all_3500.csv
# simulate 100k scenario with current station data
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2030/scenario_100k/" -o "outputs/simulated_car_sharing/test_scenario_2030_xgbprevmode_u100k_s1" -m "trained_models/xgb_prevmode.p"
# convert into matrices
python scripts/prepare_optimization_data.py --simulate -i "outputs/simulated_car_sharing/scenario_2030_xgbprevmode_u100k_s1/" -o "outputs/simulated_car_sharing/scenario_2030_xgbprevmode_u100k_s1/input_matrices"


# MAKING SCENARIOS:
# run synpp pipeline with 2019 and 2030 --> use the following parameters
1) input_downsampling: 0.011 (for 100k users --> leading to 93k?), scaling_year: 2019, station_scenario: current
2) input_downsampling: 0.013 (for 115k users --> leading to 122k), scaling_year: 2030, station_scenario: all
3) input_downsampling:  0.0165 (for 150k users --> leading to 155k), scaling_year: 2030, station_scenario: all
4) input_downsampling: 0.027 (for 250k users --> leading to 253k), scaling_year: 2030, station_scenario: all
# for further 2030 scenarios, we can reuse the cache and only change the latter files

# AFTER EACH STEP, copy them over and convert into trips
e.g. 
1) python scripts/preprocess_sim_trips.py -i "../external_repos/ch-zh-synpop/cache_2019" -o "../data/simulated_population/sim_2019" -d "2019-07-17" 
2) python scripts/preprocess_sim_trips.py -i "../external_repos/ch-zh-synpop/cache" -o "../data/simulated_population/sim_2030_all_115k" -d "2030-07-17"

# Make validation scenario and evaluate
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/xgb_2019_final" -m "trained_models/xgb_model.p" -s "csv/station_scenario/scenario_current_3000.csv"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/xgb_2019_final"

# And later generate features and scenario
python scripts/featurize_trips.py -i "../data/simulated_population/sim_2030_all_115k" --keep_geom
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2030_all_115k" -o "outputs/simulated_car_sharing/xgb_2030_all_115k_3500" -m "trained_models/xgb_model.p" -s "csv/station_scenario/scenario_all_3500.csv"
python scripts/prepare_optimization_data.py --simulate -i "outputs/simulated_car_sharing/xgb_2030_all_115k_3500/" -o "outputs/simulated_car_sharing/xgb_2030_all_115k_3500/input_matrices"