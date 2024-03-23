#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
import numpy as np
from cartes.crs import Mercator
import matplotlib.pyplot as plt
import copy
import pandas as pd

def show_plot(flight, title, sub_title=None):
    """ Displays the plot of the provided flight data """

    with plt.style.context('traffic'):
        fig = plt.figure()
        ax = plt.axes(projection=Mercator())
        fig.suptitle(title)
        flight.plot(ax, color="green")
        if sub_title:
            ax.set_title(sub_title)
        plt.show()

def apply_kalman_filter(flight):
    """Applies a Kalman filter to the given flight data."""
    # Ensure the flight data has the necessary fields
    if 'x' not in flight.data.columns or 'y' not in flight.data.columns:
        raise ValueError("Flight data must contain 'x' and 'y' columns")

    Delta_t = 10  # Time step in seconds


    # Initialize the Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([flight.data.x.iloc[0], 0, flight.data.y.iloc[0], 0])  # Assuming initial velocities are zero
    kf.F = np.array([
    [1, Delta_t, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, Delta_t],
    [0, 0, 0, 1]
])
    kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    kf.R = np.array([[1000, 0], [0, 1000]])  # Measurement noise covariance matrix
    sigma_p_squared = 1.5**2  # Acceleration noise variance (σ_p^2)

    kf.Q = np.array([
        [1/4 * Delta_t**4 * sigma_p_squared, 1/2 * Delta_t**3 * sigma_p_squared, 0, 0],
        [1/2 * Delta_t**3 * sigma_p_squared, Delta_t**2 * sigma_p_squared, 0, 0],
        [0, 0, 1/4 * Delta_t**4 * sigma_p_squared, 1/2 * Delta_t**3 * sigma_p_squared],
        [0, 0, 1/2 * Delta_t**3 * sigma_p_squared, Delta_t**2 * sigma_p_squared]
    ])

    # Apply the Kalman Filter
    for i in range(len(flight.data)):
        kf.predict()
        kf.update(np.array([flight.data.x.iloc[i], flight.data.y.iloc[i]]))
        flight.data.at[i, "x"] = kf.x[0]
        flight.data.at[i, "y"] = kf.x[2]


    
    # Task 5 - Update the lat/lon values based on the filtered x/y values
    # Conversion back to lat/lon if necessary
    set_lat_lon_from_x_y(flight)

    return flight


def apply_kalman_filter2(flight):
    """Applies a Kalman filter to the given flight data."""
    # Ensure the flight data has the necessary fields
    if 'x' not in flight.data.columns or 'y' not in flight.data.columns:
        raise ValueError("Flight data must contain 'x' and 'y' columns")

    # Initialize the Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([flight.data.x.iloc[0], 0, flight.data.y.iloc[0], 0])  # Assuming initial velocities are zero
    kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    kf.R = np.array([[10, 0], [0, 10]])  # Measurement noise. Adjust based on your scenario
    kf.Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Process noise. Adjust as needed

    # Apply the Kalman Filter
    for i in range(len(flight.data)):
        kf.predict()
        kf.update(np.array([flight.data.x.iloc[i], flight.data.y.iloc[i]]))
        flight.data.at[i, "x"] = kf.x[0]
        flight.data.at[i, "y"] = kf.x[2]


    
    # Task 5 - Update the lat/lon values based on the filtered x/y values
    # Conversion back to lat/lon if necessary
    set_lat_lon_from_x_y(flight)

    return flight

def main():
    flights = get_ground_truth_data()
    flight_name = '9MFCL_032'     
    flight = flights[flight_name]
    # DMUPY_052
    # CALIBRA_039
    # 9MFCL_032
    # VHOMS_054
    # pd.set_option('display.max_columns', None)
    # print(flight.data.head())
    # show_plot(flight, flight_name)


    unfiltered_radar_data = get_radar_data_for_flight(flight)
    unfiltered_radar_data.data.head()

    radar_data = get_radar_data(flights)

    # Task 2: Plot sample flight
    # Task 3: Plot the original and the radar data for a single flight
    # show_plot(radar_data[flight_name], "Radar Data", flight_name)
    # exit()

    # Task 4: Apply Kalman filter to the radar data
    # flight_copy_2 = copy.deepcopy(radar_data[flight_name])
    flight_copy = copy.deepcopy(radar_data[flight_name])
    filtered_flight = apply_kalman_filter(flight_copy)
    show_plot(filtered_flight, "Filtered Data", flight_name)
    # filtered_flight_2 = apply_kalman_filter2(flight_copy_2)
    # show_plot(filtered_flight_2, "Filtered Data 2 ", flight_name)
    



#############################

if __name__ == "__main__":
    main()
