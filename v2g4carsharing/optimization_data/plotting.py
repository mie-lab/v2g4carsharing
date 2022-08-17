import matplotlib.pyplot as plt
import numpy as np
import os

from auxiliary_representations import compute_availability, compute_usage


def plot_utilization(
    data_path, time_granularity=24, out_path=os.path.join("..", "figures")
):
    # compute usage
    (total_used_hours_ice, total_used_hours_ev
     ) = compute_usage(data_path, time_granularity=time_granularity)
    # computer availability
    (total_available_hours_ice,
     total_available_hours_ev) = compute_availability(
         data_path=data_path, time_granularity=time_granularity
     )

    plt.figure(figsize=(20, 7))
    x = np.arange(len(total_used_hours_ice))
    plt.bar(x, total_available_hours_ice, label="Total available normal")
    plt.bar(
        x,
        total_available_hours_ev,
        label="Total available EV",
        bottom=total_available_hours_ice
    )
    plt.bar(
        x,
        total_used_hours_ice,
        bottom=np.zeros(len(total_used_hours_ev)),
        label="Vehicles in use (normal)"
    )
    plt.bar(
        x,
        total_used_hours_ev,
        bottom=total_used_hours_ice,
        label="Vehicles in use (EV)"
    )
    plt.title("Utilization rate")
    plt.legend()
    plt.ylim(0, 5000)
    plt.savefig(os.path.join(out_path, "utilization.pdf"))

    # ev vs ice demand
    plt.figure(figsize=(10, 6))
    plt.plot(total_used_hours_ev / total_available_hours_ev, label="EV")
    plt.plot(total_used_hours_ice / total_available_hours_ice, label="ICE")
    plt.ylabel("Demand (hours in use / hours available)")
    plt.xlabel("Day of 2019 - 2020/07")
    plt.legend()
    plt.title("EV vs ICE demand - daily")
    plt.savefig(os.path.join(out_path, "ev_vs_ice_daily.jpg"))


if __name__ == "__main__":
    plot_utilization(os.path.join("..", "data"))
