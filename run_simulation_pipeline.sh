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
python scripts/test_simulate_bookings.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/test_2019_xgb" -m "trained_models/xgb_model.p"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/test_2019_xgb"

# generate station scenario (so far only simple scenario, no arguments)
python scripts/generate_station_scenario.py 
# generate realistic booking data
python scripts/generate_car_sharing_data.py -i "../data/simulated_population/sim_2019" -o "outputs/simulated_car_sharing/xgb_model_sim" -m "trained_models/xgb_model.p"
python scripts/evaluate_simulated_data.py -i "outputs/simulated_car_sharing/xgb_model_sim"