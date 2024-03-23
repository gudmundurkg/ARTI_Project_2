#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
import numpy as np
from cartes.crs import Mercator
import matplotlib.pyplot as plt
import copy
import pandas as pd
from geopy.distance import distance
import os

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

def apply_kalman_filter(flight, sigma_p, sigma_o):
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
    kf.R = np.array([[sigma_o, 0], [0, sigma_o]])  # Observation noise covariance matrix
    sigma_p_squared = sigma_p**2  # Acceleration noise variance (σ_p^2)

    # Process noise covariance matrix
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

def save_plot(flight, title, sub_title=None, folder='saved_plots'):
    """Saves the plot of the provided flight data to a folder."""
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    with plt.style.context('traffic'):
        fig = plt.figure()
        ax = plt.axes(projection=Mercator())
        fig.suptitle(title)
        flight.plot(ax, color="green")
        if sub_title:
            ax.set_title(sub_title)
        
        # Construct the filename and path to save the plot
        filename = f"{title.replace(' ', '_')}_{sub_title.replace(' ', '_') if sub_title else 'plot'}.png"
        filepath = os.path.join(folder, filename)
        
        # Save the figure
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        
        print(f"Plot saved to: {filepath}")


def main():
    flights = get_ground_truth_data()
    flight_name = 'BOE004_046'     
    flight = flights[flight_name]
    # DMUPY_052
    # CALIBRA_039
    # 9MFCL_032
    # VHOMS_054
    # pd.set_option('display.max_columns', None)
    # print(flight.data.head())
    # show_plot(flight, flight_name)
    
    res = []
    sigma_p = [0.1, 0.5, 1.5, 5, 10]
    sigma_o = [10, 100, 1000, 10000, 100000]
    
    radar_data = get_radar_data(flights)
    flight_copy = copy.deepcopy(radar_data)

    #for flight_name, flight in flights.items():
    for p in sigma_p:
        for o in sigma_o:
            unfiltered_radar_data = get_radar_data_for_flight(flight)
            filtered_flight = apply_kalman_filter(flight_copy[flight_name], p, o)
            distances = [distance(original, filtered).km for original, filtered in zip(unfiltered_radar_data.data[["latitude", "longitude"]].values, filtered_flight.data[["latitude", "longitude"]].values)]
            dist_mean = np.mean(distances)
            dist_max = np.max(distances)
            save_plot(filtered_flight, "Filtered Data", f"sigma_p {p}, sigma_o {o}")
            #show_plot(filtered_flight, f"Filtered Data: sigma_p: {p}, sigma_o: {o}", flight_name)
            res.append([flight_name, dist_mean, dist_max, sigma_p, sigma_o])

    res.sort(key=lambda x: x[1])  # Sort by mean distance
    print("Top 5 flights with the lowest mean distances:")
    for flight in res[:5]:
        print(f"{flight[0]}: Mean distance = {flight[1]:.4f} km, Max distance = {flight[2]:.4f} km, sigma_p = {flight[3]}, sigma_o = {flight[4]}")

    print("\nTop 5 flights with the highest mean distances:")
    for flight in res[-5:]:
        print(f"{flight[0]}: Mean distance = {flight[1]:.4f} km, Max distance = {flight[2]:.4f} km, sigma_p = {flight[3]}, sigma_o = {flight[4]}")

    # Task 2: Plot sample flight
    # Task 3: Plot the original and the radar data for a single flight
    # show_plot(radar_data[flight_name], "Radar Data", flight_name)
    # exit()

    # Task 4: Apply Kalman filter to the radar data
    # flight_copy_2 = copy.deepcopy(radar_data[flight_name])
   
    # filtered_flight_2 = apply_kalman_filter2(flight_copy_2)
    # show_plot(filtered_flight_2, "Filtered Data 2 ", flight_name)
    
    


#############################

if __name__ == "__main__":
    main()
