# Vehicle-to-grid for car sharing

This code was used for two papers, one about the simulating future scenarios for [V2G for car sharing in 2030](https://doi.org/10.1016/j.apenergy.2024.123731), published in Applied Energy, and one about [scheduling a national-scale fleet for ancillary services](https://doi.org/10.1186/s42162-023-00281-4) (Energy Informatics 2023).

We analyze the flexibilities for providing ancillary services in a national-scale car sharing fleet in Switzerland, first for 2019, and then in different scenarios in 2030. For simulating future scenarios, we train a mode choice model and develop an agent-based simulation of car sharing trips. 

The code base is part of a larger research project called V2G4Carsharing. We (the MIE Lab at ETH Zurich) collaborated with [Mobility car sharing](https://www.mobility.ch/en) and [Hive Power](https://www.hivepower.tech/) to analyze the potential of verhicle-to-grid technologies for car sharing. The project was funded by the Swiss Federal Office of Energy.

## Data preparation for V2G optimization

Unfortunately, the dataset used for this study is properiety data. However, the code should be applicable to car sharing data in similar format. If you are interested to run this code, please get in touch.

The code base provides the following functionalities:

#### Data preprocessing

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

#### Data preparation for scheduling optimization

This module contains all code to transform the car sharing reservations into suitable input to an optimization algorithm. This includes:
* Computing the state of charge
* Discretizing (e.g. 15 min intervals)
* Creating matrix of arrival and departure slots

Run the corresponding script:
```
python prepare_optimization_data.py [-h] [-i IN_PATH] [-o OUT_PATH] [-s SCENARIO] [-t TIME_GRANULARITY]

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

#### Generating a synthetic population

The synthetic population is generated with the synthetic population modula published by the IVT group. We use an ETH internal version of their [synpp codebase](https://github.com/eqasim-org/synpp). The code uses data from the fereral statistics office in Switzerland to generate mobility patterns for a synthetic simulation. The output are activities (with geographic coordinates for locations) and population metadata for a synthetic population. We use a suitable [configuration](v2g4carsharing/simulate/config.yml) for their pipeline.

We did only one major change to their code, which is copied here for version control. It regards the sampling of the population, which we do by the distance of the people to the next car sharing station. See our [script](v2g4carsharing/simulate/draw_car_sharing_population.py) for this code, which is excecuted in the synpp repository (it can however be tested here with data generated with synpp).

#### Preprocessing and featurizing MOBIS and simulated trips data

We first transform activities into trips for both of them (by shifting the dataframe such that we always have origin and destination geometry/purpose/starttime). 

Run:
```
python scripts/preprocess_trips.py
```
This uses the raw mobis data and the raw simulated population data (output of synpp module) and generates trips in the same format with the same features.

We then add more features to the trips, i.e. geographic features such as pt accessability, distance to stations, etc.
Run:
```
python scripts/featurize_trips.py -i ../data/simulated_population/sim_2022 --keep_geom
python scripts/featurize_trips.py -i ../data/mobis
```

#### Training a mode choice model

For simulating car sharing patterns, we train a mode choice model. For each trip, dependent on geographic features, time and activity purpose and distance, we predict the used mode. 
Different models are implemented. 

1) Simple MLP:

```
python scripts/train_mode_choice_mlp.py
```

2) Naive models:

See [simple_choice_models](v2g4carsharing/mode_choice_model/simple_choice_models.py)

#### Generate station scenario

We need to make a scenario which car sharing stations exist and what vehicles are available at these stations. So far, a simple scenario can be generated with
```
python scripts/generate_station_scenario.py
```

#### Generate synthetic car sharing booking data

The trained mode choice model (or a simple baseline model) can now be used to generate car sharing data. All trips are considered sequentially and a mode is assigned. Then, the mode-ammended-trips are converted to car sharing bookings.

Run:

```
python generate_car_sharing_data.py [-h] [-i IN_PATH_SIM_TRIPS] [-o OUT_PATH] [-s STATION_SCENARIO] [-m MODEL_PATH] [-t MODEL_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_PATH_SIM_TRIPS, --in_path_sim_trips IN_PATH_SIM_TRIPS
                        path to simulated trips csv
  -o OUT_PATH, --out_path OUT_PATH
                        path to save output
  -s STATION_SCENARIO, --station_scenario STATION_SCENARIO
                        path to station_scenario
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to mode choice model
  -t MODEL_TYPE, --model_type MODEL_TYPE
                        one of rf or irl
```

#### Evaluate the generation by comparing the distributions of simulated and real data

We can use the generated car sharing data and compare it to the real data from Mobility.

Run:
```
python evaluate_simulated_data.py [-h] [-i IN_PATH_SIM] [-d REAL_DATA_PATH] [-m MOBIS_DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_PATH_SIM, --in_path_sim IN_PATH_SIM
                        path to simulated data
  -d REAL_DATA_PATH, --real_data_path REAL_DATA_PATH
                        path to Mobility car sharing data
  -m MOBIS_DATA_PATH, --mobis_data_path MOBIS_DATA_PATH
                        path to MOBIS data
```

## References

If you build up on our work, please cite the corresponding publication:

Wiedemann, N., Xin, Y., Medici, V., Nespoli, L., Suel, E., & Raubal, M. (2024). Vehicle-to-grid for car sharing-A simulation study for 2030. Applied Energy, 372, 123731.

```bib
@article{wiedemann2024vehicle,
  title={Vehicle-to-grid for car sharing-A simulation study for 2030},
  author={Wiedemann, Nina and Xin, Yanan and Medici, Vasco and Nespoli, Lorenzo and Suel, Esra and Raubal, Martin},
  journal={Applied Energy},
  volume={372},
  pages={123731},
  year={2024},
  publisher={Elsevier}
}
```

Nespoli, L., Wiedemann, N., Suel, E., Xin, Y., Raubal, M., & Medici, V. (2023). National-scale bi-directional EV fleet control for ancillary service provision. Energy Informatics, 6(Suppl 1), 40.
```bib
@article{nespoli2023national,
  title={National-scale bi-directional EV fleet control for ancillary service provision},
  author={Nespoli, Lorenzo and Wiedemann, Nina and Suel, Esra and Xin, Yanan and Raubal, Martin and Medici, Vasco},
  journal={Energy Informatics},
  volume={6},
  number={Suppl 1},
  pages={40},
  year={2023},
  publisher={Springer}
}
```

Suel, E., Xin, Y., Wiedemann, N., Nespoli, L., Medici, V., Danalet, A., & Raubal, M. (2024). Vehicle-to-grid and car sharing: Willingness for flexibility in reservation times in Switzerland. Transportation Research Part D: Transport and Environment, 126, 104014.

```bib
@article{suel2024vehicle,
  title={Vehicle-to-grid and car sharing: Willingness for flexibility in reservation times in Switzerland},
  author={Suel, Esra and Xin, Yanan and Wiedemann, Nina and Nespoli, Lorenzo and Medici, Vasco and Danalet, Antonin and Raubal, Martin},
  journal={Transportation Research Part D: Transport and Environment},
  volume={126},
  pages={104014},
  year={2024},
  publisher={Elsevier}
}
```

