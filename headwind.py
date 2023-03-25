# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:35:17 2023

@author: Andreas J. P.
"""
import re
import requests
import scipy.stats
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# regex for finding lat-lon coordinates in .gpx xml file
RE_LATLON = re.compile(r'lat="([\d.]+)" lon="([\d.]+)"')
PCOLOR = "#262A56"


def get_wind_data(start_date, end_date, coords):
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={coords[0]}&"
           f"longitude={coords[1]}&"
           f"start_date={start_date}&"
           f"end_date={end_date}"
           "&hourly=windspeed_10m,winddirection_10m")
    response = requests.get(url)
    json = response.json()
    data = json["hourly"]
    winddir = np.array(data["winddirection_10m"]) * np.pi/180
    wx = data["windspeed_10m"] * np.cos(winddir)
    wy = data["windspeed_10m"] * np.sin(winddir)
    # Conversion from km/h to m/s
    data["windvector"] = np.stack([wx, wy]).T / 3.6

    def to_datetime(s):
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M")
    data["time"] = [to_datetime(date) for date in data["time"]]
    data = {k: np.array(v) for k, v in data.items()}
    return data


def get_moments_string(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    moment_str = r"($n={},\ \mu={:.2f},\ \sigma={:.2f}$)"
    return moment_str.format(n, mean, std)


def get_coords(fname):
    with open(fname, "r") as file:
        xml = file.read()
    coords = [tuple(map(float, m.groups())) for m in RE_LATLON.finditer(xml)]
    coords = np.array(coords)
    coords_short = coords[[0, -1], :]
    return coords, coords_short


def get_commuting_data(data):
    weekday_mask = np.array([d.weekday() <= 5 for d in data["time"]])
    hour = np.arange(0, len(data["time"])) % 24
    morning = {k: v[weekday_mask & (hour == 8)] for k, v in data.items()}
    afternoon = {k: v[weekday_mask & (hour == 16)] for k, v in data.items()}
    afternoon["windspeed_eff"] *= -1
    diff = morning["windspeed_eff"] + afternoon["windspeed_eff"]
    total = {"time": morning["time"],
             "windspeed_eff": diff}
    return morning, afternoon, total


def get_p_value(commutes):
    X = commutes[2]["windspeed_eff"]
    n = len(X)
    S = np.sqrt(np.sum((X-np.mean(X))**2)/(n-1))
    TS = np.mean(X) / (S/np.sqrt(n))
    T = scipy.stats.t(n-1)
    return T.cdf(TS)


def plot_route(coords):
    short_coords = coords[[0, -1], :]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(*coords.T, "k-o", label="Bicycle route")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.plot(*short_coords.T, "--", label="Beeline route")
    axis.set_xlabel("Longtitude")
    axis.set_ylabel("Latitude")
    axis.plot(*coords.T[:, 0], "k*", markersize=10, label="Home")
    axis.plot(*coords.T[:, -1], "kP", markersize=10, label="University")
    axis.legend()
    return fig


def plot_wind_speeds(data):
    fig, axis = plt.subplots(figsize=(10, 6))
    wind = data["windspeed_eff"]
    # Produces bar leftmost value + 1 at the end for diffing
    cnts, bins = np.histogram(wind, bins=32)
    pcts = cnts / len(wind)
    # Shift x-axis with diff / 2 for centering. Little less than actual
    # diff bar width is used
    axis.bar(bins[:-1] + np.diff(bins) / 2, pcts, np.diff(bins)*0.8,
             color=PCOLOR)
    axis.set_ylabel("Occurence")
    axis.set_xlabel("Wind speed en route [m/s]")
    axis.set_title(get_moments_string(wind))
    axis.grid()
    return fig


def plot_commutes(commutes):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ["Commute to university", "Commute to home", "Total commute"]
    for i, (commute, label) in enumerate(zip(commutes, labels)):
        cnts, bins = np.histogram(commute["windspeed_eff"], bins=32)
        pcts = cnts / len(commute["windspeed_eff"])
        axes[i].bar(bins[:-1] + np.diff(bins) / 2, pcts, np.diff(bins)*0.8,
                    color=PCOLOR)
        axes[i].set_ylabel("Occurrence")
        moment_str = get_moments_string(commute["windspeed_eff"])
        axes[i].set_title(label + " " + moment_str)
    axes[-1].set_xlabel("Wind speed [m/s]")
    return fig


if __name__ == "__main__":
    date_begin = "2018-01-01"
    date_end = "2023-03-10"
    route_coords, coords = get_coords("waypoints.gpx")
    route = np.diff(coords, axis=0)
    route = route / np.linalg.norm(route)
    data = get_wind_data(date_begin, date_end, np.mean(coords, axis=0))
    data["windspeed_eff"] = np.dot(data["windvector"], route.T)
    commutes = get_commuting_data(data)
    p_more = get_p_value(commutes)

    print(f"P-value for μ≥0: {p_more}")

    plt.style.use("ggplot")
    plot_route(route_coords)
    plt.savefig("graphs/route.webp", dpi=100, bbox_inches="tight")
    plot_wind_speeds(data)
    plt.savefig("graphs/windspeed.webp", dpi=100, bbox_inches="tight")
    plot_commutes(commutes)
    plt.savefig("graphs/commutes.webp", dpi=100, bbox_inches="tight")
