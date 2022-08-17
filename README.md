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



## Simulation

### Generating a synthetic population