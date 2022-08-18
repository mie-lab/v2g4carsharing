# Vehicle-to-grid strategies for car sharing systems

## Data preparation for V2G optimization

The first module in this repo is for preprocessing car sharing data for optimizing charging and discharging operations. There are two modules:

### import data

This module has all code for importing the raw car sharing data, for preprocessing, and for writing data to a PostGIS database. To execute these functions, check out the script
```
python scripts/import_and_clean_data.py [-h] [-i IN_PATH] [-o OUT_PATH] [-p POSTGIS_JSON_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_PATH, --in_path IN_PATH
                        path to raw data
  -o OUT_PATH, --out_path OUT_PATH
                        path to save output
  -p POSTGIS_JSON_PATH, --postgis_json_path POSTGIS_JSON_PATH
                        path to postgis access json file
```

### optimization data

This module contains all code to transform the car sharing reservations into suitable input to an optimization algorithm. This includes:
* Computing the state of charge
* Discretizing (e.g. 15 min intervals)
* Creating matrix of arrival and departure slots

Run the corresponding script:
```
prepare_optimization_data.py [-h] [-i IN_PATH] [-o OUT_PATH] [-s SCENARIO] [-t TIME_GRANULARITY]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_PATH, --in_path IN_PATH
                        path to input data
  -o OUT_PATH, --out_path OUT_PATH
                        path to save output
  -s SCENARIO, --scenario SCENARIO
                        EV scenario name
  -t TIME_GRANULARITY, --time_granularity TIME_GRANULARITY
                        discretization in hours
```

## Simulation

### Training a mode choice model

For simulating car sharing patterns, we train a mode choice model. For each trip, dependent on geographic features, time and activity purpose and distance, we predict the used mode. 
Different models are implemented. 

1) Simple MLP:

```
python scripts/train_mode_choice_mlp.py
```

2) Naive models:

See [simple_choice_models](v2g4carsharing/mode_choice_model/simple_choice_models.py)

### Generating a synthetic population

The synthetic population is generated with the eqasim 