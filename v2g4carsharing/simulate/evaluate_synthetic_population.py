import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ks_test_check(car_sharing_user_distance, population_distance_prev, population_distance_sampled):
    # compute previous ks divergence
    ks_result = scipy.stats.kstest(car_sharing_user_distance, population_distance_prev)
    print("Distribution difference in KS test (raw population):", ks_result)

    # compute new ks divergence after sampling
    ks_result = scipy.stats.kstest(car_sharing_user_distance, population_distance_sampled)
    print("Distribution difference in KS test (sampled):", ks_result)


def compare_station_market_share(sampled, population, car_sharing_users, out_path="outputs/figures/"):
    assert "nearest_station_distance" in sampled.columns, "must have nearest station distance computed"
    assert "nearest_station_distance" in population.columns, "must have nearest station distance computed"
    assert "nearest_station_distance" in car_sharing_users.columns, "must have nearest station distance computed"
    count_station_all = population[["closest_station"]].drop_duplicates()
    for df, name in zip([sampled, population, car_sharing_users], ["sampled", "population", "mobility users"]):
        count_station = (
            df.groupby("closest_station")  # group by station
            .agg({"closest_station": "count"})  # count
            .rename(columns={"closest_station": name})  # rename the count-column
            .sort_values(name, ascending=False)
        )
        count_station[name] = count_station[name] / count_station[name].sum()
        #     if count_station_all is not None:
        count_station_all = count_station_all.merge(
            count_station, left_on="closest_station", right_index=True, how="left"
        )
    count_station_all["mobility-sampled"] = count_station_all["mobility users"] / count_station_all["sampled"]
    count_station_all["mobility-population"] = count_station_all["mobility users"] / count_station_all["population"]
    print("Mean ratios between sampled, population and car sharing users:", count_station_all.mean())

    # check distance to station
    plt.figure(figsize=(15, 6))
    names = ["Sampled", "Original population", "Mobility users"]
    for i, df in enumerate([sampled, population, car_sharing_users]):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df["nearest_station_distance"], cumulative=True, stat="probability")
        plt.xlim(0, 5000)
        plt.title(names[i])
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "sampled_vs_original.pdf"))


def compare_distance_sampling_to_station_sampling(
    sampled_by_distance, sampled_by_marketshare, car_sharing_users, lim=8000, out_path="outputs/figures/"
):
    """Two options implemented: Sampling by distance (beta distribution) and sampling by the station marketshare"""
    plt.figure(figsize=(15, 6))
    names = ["Sampled by distance", "Sampled by age, sex, stations", "Mobility users"]
    for i, df in enumerate([sampled_by_distance, sampled_by_marketshare, car_sharing_users]):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df["nearest_station_distance"], cumulative=True, stat="probability")
        plt.plot([0, lim], [1, 1])
        plt.xlim(0, lim)
        plt.title(names[i])
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "sampling_methods.png"))


def compare_categorical_distribution(sampled, population, car_sharing_users, out_path="outputs/figures/"):
    for var in ["sex", "agegroup_int"]:
        xlabel = "sex" if var == "sex" else "age group"
        plt.figure(figsize=(15, 6))
        names = ["Sampled", "Original population", "Mobility users"]
        for i, df in enumerate([sampled, population, car_sharing_users]):
            plt.subplot(1, 3, i + 1)
            uni, counts = np.unique(df[var].dropna(), return_counts=True)
            plt.bar(uni, counts)
            plt.xticks(uni)
            plt.xlabel(xlabel)
            plt.title(names[i])
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"{var}_sampled_vs_real.pdf"))
