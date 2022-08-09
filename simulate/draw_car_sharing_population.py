import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pickle
import os
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
import scipy
import geopandas as gpd
from sklearn.neighbors import KernelDensity

from geoml.io import read_geodataframe


def load_and_convert_crs(file_path, index_name):
    df = read_geodataframe(file_path).set_index(index_name)
    df.crs = "EPSG:4326"
    df = df.to_crs("EPSG:2056")
    return df


def fit_beta_distribution(car_sharing_user_distance, scaled_population):

    # fit beta to the car sharing user values
    beta_params_car_sharing_users = scipy.stats.beta.fit(car_sharing_user_distance)
    # fit beta to the general population
    beta_params_population = scipy.stats.beta.fit(scaled_population["nearest_station_distance"].values)

    # get initial probs (the probs of the values that are in the general population)
    a, b, loc, scale = beta_params_population
    rv = scipy.stats.beta(a, b, loc, scale)
    scaled_population["initial_probs"] = rv.pdf(scaled_population["nearest_station_distance"].values)

    # get probs of the values in the target distribution (the car sharing users)
    a, b, loc, scale = beta_params_car_sharing_users
    rv = scipy.stats.beta(a, b, loc, scale)
    scaled_population["target_dist_probs"] = rv.pdf(scaled_population["nearest_station_distance"].values)

    # probs for shifting target / initial
    scaled_population["sample_probs"] = scaled_population["target_dist_probs"] / scaled_population["initial_probs"]
    return scaled_population


def fit_kde(car_sharing_user_distance, scaled_population):
    # fit KDE to the car sharing user values
    kde_fit_car_sharing = KernelDensity()
    kde_fit_car_sharing.fit(car_sharing_user_distance.reshape(-1, 1))

    # fit KDE to general population
    kde_fit_population = KernelDensity()
    kde_fit_population.fit(scaled_population["nearest_station_distance"].values.reshape(-1, 1))

    # get probs of the values in the target distribution (the car sharing users)
    samples_to_score = scaled_population["nearest_station_distance"].values.reshape(-1, 1)
    scaled_population["initial_probs"] = kde_fit_car_sharing.score_samples(samples_to_score)
    # get initial probs (the probs of the values that are in the general population)
    scaled_population["target_dist_probs"] = kde_fit_population.score_samples(samples_to_score)
    # probs for shifting target / initial
    scaled_population["sample_probs"] = scaled_population["target_dist_probs"] / scaled_population["initial_probs"]

    return scaled_population


def fit_kmm(car_sharing_user_distance, scaled_population):
    from cvxopt import matrix, solvers
    import math

    def kernel_mean_matching(X, Z, kern="lin", B=1.0, eps=None):
        nx = X.shape[0]
        nz = Z.shape[0]
        if eps == None:
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


def compute_sample_weights(scaled_population, car_sharing_data_path, swiss_boundaries_path=None, dist="beta"):

    # read car sharing user data
    user_df = load_and_convert_crs(os.path.join(car_sharing_data_path, "user_2.csv"), "person_no")
    assert swiss_boundaries_path or ("in_switzerland" in user_df.columns), "Specify a path to Swiss boundaries!"
    # read station data
    station_df = load_and_convert_crs(os.path.join(car_sharing_data_path, "station.csv"), "station_no")

    # Group scaled population by home location (to reduce computational effort in the following)
    print("Compute distance for population")
    scaled_locations = scaled_population[["home_x", "home_y"]].drop_duplicates()
    scaled_locations = gpd.GeoDataFrame(
        scaled_locations,
        geometry=gpd.points_from_xy(scaled_locations["home_x"], scaled_locations["home_y"], crs="EPSG:2056"),
    )
    # Compute nearest station distance for scaled population
    scaled_locations = scaled_locations.sjoin_nearest(station_df[["geom"]], distance_col="nearest_station_distance")

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
    user_df = user_df.sjoin_nearest(station_df[["geom"]], distance_col="nearest_station_distance")

    # merge with initial population
    scaled_population = scaled_population.merge(
        scaled_locations[["home_x", "home_y", "nearest_station_distance"]],
        left_on=["home_x", "home_y"],
        right_on=["home_x", "home_y"],
        how="left",
    )

    # compute sample weights
    print("Fit Beta distribution")
    fit_func = fit_beta_distribution if dist == "beta" else fit_kde
    scaled_population = fit_func(user_df["nearest_station_distance"].values, scaled_population)

    # check with KS test whether our sample is good
    print("Sample from population")
    population_sample = scaled_population.sample(10000, replace=False, weights=scaled_population["sample_probs"])
    ks_test_check(
        user_df["nearest_station_distance"].values,
        scaled_population["nearest_station_distance"].values,
        population_sample["nearest_station_distance"].values,
        plot_path=car_sharing_data_path,
    )

    return scaled_population


def ks_test_check(car_sharing_user_distance, population_distance_prev, population_distance_sampled, plot_path=None):
    # compute previous ks divergence
    ks_result = scipy.stats.kstest(car_sharing_user_distance, population_distance_prev)
    print("Distribution difference in KS test (raw population):", ks_result)

    # compute new ks divergence after sampling
    ks_result = scipy.stats.kstest(car_sharing_user_distance, population_distance_sampled)
    print("Distribution difference in KS test (sampled):", ks_result)

    if plot_path is not None:
        plt.subplot(1, 3, 1)
        plt.hist(population_distance_prev[population_distance_prev < 1000])
        plt.title("Initial distribution")
        plt.xlim(-50, 1000)
        plt.subplot(1, 3, 2)
        plt.hist(car_sharing_user_distance[car_sharing_user_distance < 1000])
        plt.title("Target distribution")
        plt.xlim(-50, 1000)
        plt.subplot(1, 3, 3)
        plt.hist(population_distance_sampled[population_distance_sampled < 1000])
        plt.title("Sampled distribution")
        plt.xlim(-50, 1000)
        plt.tight_layout()
        plt.savefig(plot_path)


if __name__ == "__main__":

    cache_path = "../external_repos/ch-zh-synpop/cache"
    population_filename = "scaled_with_sample_weights.csv"  # data.statpop.scaled__014e06a497b942b1e80168c295cd152a.p
    car_sharing_data_path = "data"
    swiss_boundaries_path = "~/MIE/general_data/swiss_boundaries/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp"
    sample_ratio = 0.01

    # Load the full synthetic population (9mio persons)
    # with open(os.path.join(cache_path, population_filename), "rb") as infile: # TODO
    #     scaled_population = pickle.load(infile)
    scaled_population = pd.read_csv(os.path.join(cache_path, population_filename))
    nr_samples = int(sample_ratio * len(scaled_population))

    if "sample_probs" not in scaled_population.columns:
        scaled_population = compute_sample_weights(scaled_population, car_sharing_data_path, swiss_boundaries_path)
        scaled_population.to_csv(os.path.join(cache_path, "scaled_with_sample_weights.csv"), index=False)

    # sample population
    print("Sample from population")
    population_sample = scaled_population.sample(nr_samples, replace=False, weights=scaled_population["sample_probs"])
    population_sample.to_csv(os.path.join(cache_path, "sampled_by_distance.csv"), index=False)

