#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
import numpy as np
from cartes.crs import Mercator
import matplotlib.pyplot as plt
import copy
import pandas as pd
from geopy.distance import distance

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
    flights_list = [
    "9MFCL_032", "IGRAD_000", "PXR31F_001", "N441FS_002", "N441FS_003",
    "OKVOK_004", "FGALN_005", "FGALN_006", "HAYCG_007", "FGALN_008",
    "FAF4011_009", "PSWRD35_010", "F-CHDN_011", "D-KRPK_012", "D-6567_013",
    "SS_014", "F-CIJO_015", "G-CLPU_016", "D-KWFW_017", "REGA1_018",
    "SAMU31_019", "REGA1_020", "CALIBRA_021", "YS111N_022", "VOR05_023",
    "GTACN_024", "ADA4_025", "CALIBRA_026", "CALIBRA_027", "CALIBRA_028",
    "CALIBRA_029", "GTACN_030", "ADA4_031", "9MFCL_032", "NVC201_033",
    "VOR05_034", "CALIBRA_035", "FCK211_036", "NVC103_037", "VOR05_038",
    "CALIBRA_039", "CFL12_040", "CFL12_041", "YS111N_042", "CALIBRA_043",
    "FCK211_044", "GTEKV_045", "BOE004_046", "N56821_047", "ZEROG_048",
    "QFA7474_049", "THY1920_050", "AIB232E_051", "DMUPY_052", "TRA051_053",
    "VHOMS_054", "HAXAE_055", "OOLET_056"
    ]

    
    flight_name = 'FAF4011_009'     
    flight = flights[flight_name]
    # DMUPY_052
    # Mean distance between original and filtered data:  1.168052 km
    # Maximal distance between original and filtered data:  5.468760 km
    # CALIBRA_039
    # Mean distance between original and filtered data:  1.043040 km
    # Maximal distance between original and filtered data:  5.470307 km   
    # 9MFCL_032
    # Mean distance between original and filtered data:  0.504637 km
    # Maximal distance between original and filtered data:  2.029270 km
    # VHOMS_054
    # Mean distance between original and filtered data:  0.969023 km
    # Maximal distance between original and filtered data:  4.427027 km
    # pd.set_option('display.max_columns', None)
    # print(flight.data.head())
    show_plot(flight, flight_name)
    exit()


    unfiltered_radar_data = get_radar_data_for_flight(flight)
    unfiltered_radar_data.data.head()

    radar_data = get_radar_data(flights)

    flight_distances = {}

    for flight_name in flights_list:
        flight_copy = copy.deepcopy(radar_data[flight_name])
        filtered_flight = apply_kalman_filter(flight_copy)

        unfiltered_radar_data = get_radar_data_for_flight(flights[flight_name])
        distances = [distance(original, filtered).km for original, filtered in zip(unfiltered_radar_data.data[["latitude", "longitude"]].values, filtered_flight.data[["latitude", "longitude"]].values)]

        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        flight_distances[flight_name] = {'mean_distance': mean_distance, 'max_distance': max_distance}

    # Convert dictionary to a list of tuples and sort by mean distance
    sorted_flights_by_mean = sorted(flight_distances.items(), key=lambda item: item[1]['mean_distance'])
    # Extracting the flight names and distances for top 5 smallest and highest mean distances
    top_5_smallest_mean = [(flight, metrics['mean_distance'], metrics['max_distance']) for flight, metrics in sorted_flights_by_mean[:5]]
    top_5_highest_mean = [(flight, metrics['mean_distance'], metrics['max_distance']) for flight, metrics in sorted_flights_by_mean[-5:]]

    print("Top 5 flights with the smallest mean distances:")
    for flight, mean_distance, max_distance in top_5_smallest_mean:
        print(f"{flight}: Mean distance = {mean_distance:.6f} km, Max distance = {max_distance:.6f} km")

    print("\nTop 5 flights with the highest mean distances:")
    for flight, mean_distance, max_distance in top_5_highest_mean:
        print(f"{flight}: Mean distance = {mean_distance:.6f} km, Max distance = {max_distance:.6f} km")




    # flight_mean_distances = []


    # for flight_name in flights_list:
    #     flight_copy = copy.deepcopy(radar_data[flight_name])
    #     filtered_flight = apply_kalman_filter(flight_copy)

    #     unfiltered_radar_data = get_radar_data_for_flight(flights[flight_name])
    #     distances = [distance(original, filtered).km for original, filtered in zip(unfiltered_radar_data.data[["latitude", "longitude"]].values, filtered_flight.data[["latitude", "longitude"]].values)]

    #     mean_distance = np.mean(distances)
    #     max_distance = np.max(distances)
    #     flight_mean_distances.append((flight_name, mean_distance, max_distance))


    # flight_mean_distances.sort(key=lambda x: x[1])
    # top_5_smallest_mean = flight_mean_distances[:5]
    # top_5_highest_mean = flight_mean_distances[-5:]

    # print("Top 5 flights with the smallest mean distances:")
    # for flight, mean_distance in top_5_smallest_mean:
    #     print(f"{flight}: {mean_distance:.6f} km {max_distance:.6f} km")

    # print("\nTop 5 flights with the highest mean distances:")
    # for flight, mean_distance in top_5_highest_mean:
    #     print(f"{flight}: {mean_distance:.6f} km {max_distance:.6f} km")
    # Task 2: Plot sample flight
    # Task 3: Plot the original and the radar data for a single flight
    # show_plot(radar_data[flight_name], "Radar Data", flight_name)
    # exit()

    # Task 4: Apply Kalman filter to the radar data
    # flight_copy_2 = copy.deepcopy(radar_data[flight_name])
    # flight_copy = copy.deepcopy(radar_data[flight_name])
    # filtered_flight = apply_kalman_filter(flight_copy)
    # show_plot(filtered_flight, "Filtered Data", flight_name)
    # filtered_flight_2 = apply_kalman_filter2(flight_copy_2)
    # show_plot(filtered_flight_2, "Filtered Data 2 ", flight_name)
    
    # distances = [distance(original, filtered).km for original, filtered in zip(unfiltered_radar_data.data[["latitude", "longitude"]].values, filtered_flight.data[["latitude", "longitude"]].values)]
    # print(f"Mean distance between original and filtered data: {np.mean(distances): 4f} km")
    # print(f"Maximal distance between original and filtered data: {np.max(distances): 4f} km")


#############################

if __name__ == "__main__":
    main()
