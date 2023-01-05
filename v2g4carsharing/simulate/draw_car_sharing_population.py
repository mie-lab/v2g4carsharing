import numpy as np
import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import scipy
from shapely import wkt
import geopandas as gpd
from sklearn.neighbors import KernelDensity

from v2g4carsharing.simulate.evaluate_synthetic_population import *

# constants
base_year_prevalence = 2019
scenario_path_mapping = {
    "current": "../csv/station_scenario/scenario_current_3000.csv",
    "all": "../csv/station_scenario/scenario_all_3500.csv",
    "new1000": "../csv/station_scenario/scenario_new1000_7500.csv",
    "new1250": "../csv/station_scenario/scenario_new1250_7500.csv"
}

def configure(context):
    context.config("input_downsampling")
    context.config("station_scenario")
    context.stage("data.statpop.scaled")


def load_and_convert_crs(file_path, index_name, convert_crs=True):
    def read_geodataframe(in_path, geom_col="geom"):
        df = pd.read_csv(in_path)
        df[geom_col] = df[geom_col].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry=geom_col)
        return gdf

    df = read_geodataframe(file_path).set_index(index_name)
    if convert_crs:
        df.crs = "EPSG:4326"
        df = df.to_crs("EPSG:2056")
    else:
        df.crs = "EPSG:2056"
    return df


def fit_beta_distribution(car_sharing_user_distance, scaled_population):

    # fit beta to the car sharing user values
    beta_params_car_sharing_users = scipy.stats.beta.fit(car_sharing_user_distance)
    # fit beta to the general population
    beta_params_population = scipy.stats.beta.fit(scaled_population["nearest_station_distance"].values)

    # get initial probs (the probs of the values that are in the population)
    a, b, loc, scale = beta_params_population
    rv = scipy.stats.beta(a, b, loc, scale)
    scaled_population["initial_probs"] = rv.pdf(scaled_population["nearest_station_distance"].values)

    # get probs of the values in the target distribution (the car sharing user)
    a, b, loc, scale = beta_params_car_sharing_users
    rv = scipy.stats.beta(a, b, loc, scale)
    scaled_population["target_dist_probs"] = rv.pdf(scaled_population["nearest_station_distance"].values)

    # probs for shifting target / initial
    scaled_population["sample_probs"] = scaled_population["target_dist_probs"] / scaled_population["initial_probs"]
    return scaled_population


def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    """Speed up for KDE sample scoring"""
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


def fit_kde(car_sharing_user_distance, scaled_population, bandwidth=100):
    # fit KDE to the car sharing user values
    kde_fit_car_sharing = KernelDensity(atol=0.001, rtol=0.001, bandwidth=bandwidth)
    kde_fit_car_sharing.fit(car_sharing_user_distance.reshape(-1, 1))

    # fit KDE to general population
    kde_fit_population = KernelDensity(atol=0.001, rtol=0.001, bandwidth=bandwidth)
    kde_fit_population.fit(scaled_population["nearest_station_distance"].values.reshape(-1, 1))

    # get probs of the values in the target distribution (the car sharing user)
    samples_to_score = scaled_population["nearest_station_distance"].values.reshape(-1, 1)
    scaled_population["initial_probs"] = np.exp(parrallel_score_samples(kde_fit_car_sharing, samples_to_score))
    # get initial probs (the probs of the values that are in the population)
    scaled_population["target_dist_probs"] = np.exp(parrallel_score_samples(kde_fit_population, samples_to_score))
    # probs for shifting target / initial
    scaled_population["sample_probs"] = scaled_population["target_dist_probs"] / scaled_population["initial_probs"]

    return scaled_population


def fit_kmm(car_sharing_user_distance, scaled_population):
    from cvxopt import matrix, solvers
    import math

    def kernel_mean_matching(X, Z, kern="lin", B=1.0, eps=None):
        nx = X.shape[0]
        nz = Z.shape[0]
        if eps is None:
            eps = B / math.sqrt(nz)
        if kern == "lin":
            K = np.dot(Z, Z.T)
            kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
        elif kern == "rbf":
            K = compute_rbf(Z, Z)
            kappa = np.sum(compute_rbf(Z, X), axis=1) * float(nz) / float(nx)
        else:
            raise ValueError("unknown kernel")

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
        h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol["x"])
        return coef

    def compute_rbf(X, Z, sigma=1.0):
        K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
        for i, vx in enumerate(X):
            K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
        return K

    scaled_population["sample_probs"] = kernel_mean_matching(
        car_sharing_user_distance.reshape(-1, 1), scaled_population["nearest_station_distance"].values.reshape(-1, 1)
    )
    return scaled_population


def load_compute_distances(scaled_population, car_sharing_data_path, station_scenario, swiss_boundaries_path=None):
    """Load user and station data, add closest station and distance to closest station to them"""
    print("Reading users and station...")
    # read car sharing user data
    user_df = load_and_convert_crs(os.path.join(car_sharing_data_path, "user.csv"), "person_no")
    assert swiss_boundaries_path or ("in_switzerland" in user_df.columns), "Specify a path to Swiss boundaries!"
    # read station data and restrict to most active ones
    station_df = load_and_convert_crs(
        os.path.join(car_sharing_data_path, scenario_path_mapping[station_scenario]), "station_no", convert_crs=False
    )
    if "in_reservation" in station_df.columns:
        station_df = station_df[station_df["in_reservation"]]

    # Group scaled population by home location (to reduce computational effort in the following)
    print("Compute distance for population")
    scaled_locations = scaled_population[["home_x", "home_y"]].drop_duplicates()
    scaled_locations = gpd.GeoDataFrame(
        scaled_locations, geometry=gpd.points_from_xy(scaled_locations["home_x"], scaled_locations["home_y"]),
    )
    scaled_locations.crs = "EPSG:2056"
    # Compute nearest station distance for scaled population
    scaled_locations = scaled_locations.sjoin_nearest(
        station_df[["geom"]], distance_col="nearest_station_distance"
    ).rename(columns={"index_right": "closest_station"})
    # Problem: Some stations are duplicates and therefore added multiple times. We can simply remove the duplicates
    scaled_locations.drop_duplicates(subset=["home_x", "home_y"], inplace=True)

    print("Compute distance for car sharing users")
    # Compute nearest station distance for car sharing users
    # remove the users that live outside Switzerland:
    if "in_switzerland" not in user_df.columns:
        swiss_boundaries = gpd.read_file(swiss_boundaries_path)
        swiss_boundaries = swiss_boundaries.set_index("NAME").loc["Schweiz"]["geometry"]
        user_df = user_df[user_df.within(swiss_boundaries)]
    else:
        user_df = user_df[user_df["in_switzerland"]]
    print("Number of users after removing outside switzerland:", len(user_df))
    # distance to nearest station
    user_df = user_df.sjoin_nearest(station_df[["geom"]], distance_col="nearest_station_distance").rename(
        columns={"index_right": "closest_station"}
    )

    # merge with initial population
    scaled_population = scaled_population.merge(
        scaled_locations[["home_x", "home_y", "nearest_station_distance", "closest_station"]],
        left_on=["home_x", "home_y"],
        right_on=["home_x", "home_y"],
        how="left",
    )
    return scaled_population, user_df


def save_categorical_dist(
    feat_values,
    col_name,
    out_path=os.path.join("..", "data", "simulated_population", "prevalence_2019"),
    out_name="population",
):
    """Save a csv with the distribution"""
    uni, counts = np.unique(feat_values, return_counts=True)
    # normalize
    counts = counts / np.sum(counts)
    df = pd.DataFrame(uni, columns=[col_name])
    df["prob"] = counts
    df.to_csv(os.path.join(out_path, f"prevalence_{out_name}_{col_name}.csv"), index=False)


def sample_from_population(
    scaled_population,
    carsharing_users,
    sampling_strategy="station_age_gender",
    nr_samples=10000,
    station_scenario="current",
    path_prevalence=os.path.join("..", "data", "simulated_population", "prevalence_2019"),
    out_path_distribution_comparison=None,
):

    # Version 1: Sample with beta distribution
    if sampling_strategy == "beta":
        fit_func = fit_beta_distribution if sampling_strategy == "beta" else fit_kde
        scaled_population = fit_func(carsharing_users["nearest_station_distance"].values, scaled_population)
        population_sample = scaled_population.sample(
            nr_samples, replace=False, weights=scaled_population["sample_probs"]
        )
        return population_sample

    # Version 2: Sampling according to age, gender and station market share

    # align age and sex features
    print("Aligning gender and agegroup variables...")
    agegroup_mapping = {17: 1, 26: 2, 30: 3, 40: 4, 50: 5, 65: 6}
    agegroup_mapping_keys = np.array(sorted(agegroup_mapping.keys()))
    scaled_population["agegroup_int"] = scaled_population["age"].apply(
        lambda x: np.where(agegroup_mapping_keys <= x)[0][-1] + 1 if x > 16 else 0
    )
    carsharing_users["sex"] = carsharing_users["gender"].map({"m": 0, "w": 1, pd.NA: -1})

    # Path to prevalence in population
    # Save the population prevalence if it does not exist yet
    if not os.path.exists(path_prevalence): # TODO: uncomment if need to recreate new prevalence, e.g. for new stations
        os.makedirs(path_prevalence, exist_ok=True)
        save_categorical_dist(
            scaled_population["agegroup_int"].values,
            "agegroup_int",
            out_name="population",
            out_path=path_prevalence,
        )
        save_categorical_dist(scaled_population["sex"], "sex", out_name="population", out_path=path_prevalence)
        save_categorical_dist(
            scaled_population["closest_station"].dropna(),
            f"closest_station_{station_scenario}",
            out_name="population",
            out_path=path_prevalence,
        )
        # Saving the carsharing-user prevalence once is sufficent, uncomment only for recompute:
        save_categorical_dist(
            carsharing_users["sex"].dropna(), 
            "sex",
            out_name="real",
            out_path=path_prevalence,
        )
        save_categorical_dist(carsharing_users["agegroup_int"].dropna(), "agegroup_int", out_name="real",
                out_path=path_prevalence)
        save_categorical_dist(
            carsharing_users["closest_station"].dropna(), f"closest_station_{station_scenario}", out_name="real",
                out_path=path_prevalence
        )

    print("Compute sample weights...")
    scaled_population["sample_probs"] = 1
    for feat in ["sex", "agegroup_int", f"closest_station_{station_scenario}"]:
        feat_raw_name = "closest_station" if feat.startswith("closest") else feat
        # load prevalence in car sharing users
        prob_real = pd.read_csv(os.path.join(path_prevalence, f"prevalence_real_{feat}.csv"))
        # load prevalence in overall population
        prob_population = pd.read_csv(os.path.join(path_prevalence, f"prevalence_population_{feat}.csv"))
        # merge both
        merged = prob_real.merge(
            prob_population, left_on=feat, right_on=feat, how="outer", suffixes=("_real", "_population")
        )
        merged["prob_population"].fillna(1, inplace=True)  # need to fill with 1 to avoid zero division
        merged["prob_real"].fillna(0, inplace=True)
        # divide prevalences to yield the sample weight
        merged[f"sample_weight_{feat_raw_name}"] = merged["prob_real"] / merged["prob_population"]
        scaled_population = scaled_population.merge(
            merged[[feat, f"sample_weight_{feat_raw_name}"]], left_on=feat_raw_name, right_on=feat, how="left"
        )
        # multiply the final sample weight
        scaled_population["sample_probs"] = (
            scaled_population["sample_probs"] * scaled_population[f"sample_weight_{feat_raw_name}"]
        )
        print("Added sample weight of ", feat)

    # sample population
    print("Sample from population")
    population_sample = scaled_population.sample(nr_samples, replace=False, weights=scaled_population["sample_probs"])

    # If desired, compare the distributions of the sampled population with the original ones etc
    if out_path_distribution_comparison is not None:
        # check differences between distribution of our sampled population and the car sharing users
        compare_station_market_share(
            population_sample, scaled_population, carsharing_users, out_path_distribution_comparison
        )
        compare_categorical_distribution(
            population_sample, scaled_population, carsharing_users, out_path_distribution_comparison
        )
    return population_sample


def execute(context):
    if __name__ == "__main__":
        import pickle

        probability = 0.01
        # Load cached population
        cache_path = "../external_repos/ch-zh-synpop/cache_2019"
        population_filename = "data.statpop.scaled__eed15811a6c8151f74013561a07c9b03.p"
        with open(os.path.join(cache_path, population_filename), "rb") as infile:
            scaled_population = pickle.load(infile)
        # scaled_population = scaled_population.sample(n=100000) # NOTE: only for testing
        # Paths for further input data
        car_sharing_data_path = "data"
        swiss_boundaries_path = "../../general_data/swiss_boundaries/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp"
        out_path_distribution_comparison = "outputs/figures"
        station_scenario = "new1000"
        path_prevalence = "../data/simulated_population/prevalence_2019"
    else:
        # get from contect
        scaled_population = context.stage("data.statpop.scaled")
        # If we do not want to downsample, set the value to 1.0 in config
        probability = context.config("input_downsampling")
        car_sharing_data_path = "../../v2g4carsharing/data"
        out_path_distribution_comparison = "../../v2g4carsharing/outputs/figures"
        path_prevalence = "../../data/simulated_population/prevalence_2019"
        station_scenario = context.config("station_scenario")
        swiss_boundaries_path = None

    if probability < 1.0:
        print("Downsampling (%f)" % probability)

        household_ids = np.unique(scaled_population["household_id"])
        print("  Initial number of households:", len(household_ids))
        print("  Initial number of persons:", len(np.unique(scaled_population["person_id"])))

        # ######### VERSION 1: UNIFORM ###################
        # # during downsampling, households are selected randomly without
        # # specifying a seed, which means that running the pipeline twice will
        # # produce different populations
        # # resulting in potentially different simulation results
        # f = np.random.random(size=(len(household_ids), )) < probability
        # remaining_household_ids = household_ids[f]
        # print("  Sampled number of households:", len(remaining_household_ids))

        # df = df[df["household_id"].isin(remaining_household_ids)]
        # print("  Sampled number of persons:", len(np.unique(df["person_id"])))

        # ######### VERSION 2: CAR SHARING STATIONS ###################

        sampling_strategy = "station_age_gender"  # "beta"
        nr_samples = int(probability * len(scaled_population))

        # Load the data: Car sharing users and the overall (simulated) population
        scaled_population, car_sharing_users = load_compute_distances(
            scaled_population, car_sharing_data_path, station_scenario, swiss_boundaries_path=swiss_boundaries_path
        )

        # Sample population
        population_sample = sample_from_population(
            scaled_population,
            car_sharing_users,
            nr_samples=nr_samples,
            sampling_strategy=sampling_strategy,
            station_scenario=station_scenario,
            out_path_distribution_comparison=out_path_distribution_comparison,
            path_prevalence=path_prevalence,
        )

        # Clean
        # remove the children
        population_sample = population_sample[population_sample["age"] >= 18]
        # Add head for each household
        population_sample.drop("is_head", axis=1, inplace=True)
        max_age_idx = population_sample.loc[population_sample.groupby("household_id")["age"].idxmax()].reset_index(
            drop=True
        )
        population_sample["is_head"] = population_sample["person_id"].isin(max_age_idx["person_id"])

    return population_sample.drop("sample_probs", axis=1)


if __name__ == "__main__":
    population_sample = execute(None)
    # scaled_population.to_csv(os.path.join(cache_path, "scaled_with_sample_weights.csv"), index=False)
    cache_path = "../external_repos/ch-zh-synpop/cache"
    population_sample.to_csv(os.path.join(cache_path, "sampled_by_distance.csv"), index=False)

