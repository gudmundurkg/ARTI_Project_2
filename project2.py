#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
# import geopy to calculated difference between two flights
from geopy.distance import geodesic
import numpy as np
from cartes.crs import Mercator
import matplotlib.pyplot as plt
import copy


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


def calculate_distance(flight1, flight2):
    """Calculates the distance between two flights"""
    # Ensure the flights have the necessary fields
    if 'latitude' not in flight1.data.columns or 'longitude' not in flight1.data.columns:
        raise ValueError("Flight data must contain 'latitude' and 'longitude' columns")
    if 'latitude' not in flight2.data.columns or 'longitude' not in flight2.data.columns:
        raise ValueError("Flight data must contain 'latitude' and 'longitude' columns")
    # Initialize total distance to 0
    total_distance = 0

    # Iterate over the flight data
    for i in range(len(flight1.data)):
        # Get the latitude/longitude for each flight
        coord1 = (flight1.data.latitude.iloc[i], flight1.data.longitude.iloc[i])
        coord2 = (flight2.data.latitude.iloc[i], flight2.data.longitude.iloc[i])

        # Calculate the distance between the two flights at this point
        distance = geodesic(coord1, coord2).meters

        # Add the distance to the total
        total_distance += distance

    # Return the total distance
    return total_distance

def calculate_distance_error(flight1, flight2):
    """Calculates mean and maximal distance error between two flights at the same time points."""
    # Ensure the flights have the necessary fields
    if 'latitude' not in flight1.data.columns or 'longitude' not in flight1.data.columns:
        raise ValueError("Flight data must contain 'latitude' and 'longitude' columns")
    if 'latitude' not in flight2.data.columns or 'longitude' not in flight2.data.columns:
        raise ValueError("Flight data must contain 'latitude' and 'longitude' columns")

    distances = []

    # Assuming flight1.data and flight2.data are aligned and have the same length
    for i in range(len(flight1.data)):
        coord1 = (flight1.data.latitude.iloc[i], flight1.data.longitude.iloc[i])
        coord2 = (flight2.data.latitude.iloc[i], flight2.data.longitude.iloc[i])
        
        # Calculate the distance between the two points
        distance = geodesic(coord1, coord2).meters
        distances.append(distance)
    
    # Calculate mean and maximal distance
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    return mean_distance, max_distance

def main():
    flights = get_ground_truth_data()
    flight_name = 'DMUPY_052'     
    flight = flights[flight_name]

    unfiltered_radar_data = get_radar_data_for_flight(flight)
    unfiltered_radar_data.data.head()

    radar_data = get_radar_data(flights)

    # Task 2: Plot sample flight
    # Task 3: Plot the original and the radar data for a single flight
    # show_plot(flight, "Original data", flight_name)
    # show_plot(radar_data[flight_name], "Radar Data", flight_name)

    # Task 4: Apply Kalman filter to the radar data
    flight_copy = copy.deepcopy(radar_data[flight_name])
    filtered_flight = apply_kalman_filter(flight_copy)
    # show_plot(filtered_flight, "Filtered Data", flight_name)

    # Task 6: Calculate mean and maximal distance errors between original and filtered flight data
    mean_distance, max_distance = calculate_distance_error(unfiltered_radar_data, filtered_flight)
    print(f"Mean distance error: {mean_distance:.2f} meters")
    print(f"Maximal distance error: {max_distance:.2f} meters")

#############################

if __name__ == "__main__":
    main()
