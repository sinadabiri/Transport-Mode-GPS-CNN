import numpy as np
import pickle
from geopy.distance import vincenty
import os
import math

A = math.degrees(-math.pi)
# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
filename = '../Combined Trajectory_Label_Geolife/Revised_Trajectory_Label_Array.pickle'
with open('Revised_Trajectory_Label_Array.pickle', 'rb') as f:
    Trajectory_Label_Array = pickle.load(f)

# Identify the Speed and Acceleration limit
SpeedLimit = {0: 7, 1: 12, 2: 120./3.6, 3: 180./3.6, 4: 120/3.6}
# Online sources for Acc: walk: 1.5 Train 1.15, bus. 1.25 (.2), bike: 2.6, train:1.5
AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}
# Choose based on figure visualization for JerkP:{0: 4, 1: 4, 2: 4, 3: 11, 4: 6}
JerkLimitP = {0: 40, 1: 40, 2: 40, 3: 110, 4: 60}
# Choose based on figure visualization for JerkN:{0: -4, 1: -4, 2: -2.5, 3: -11, 4: -4}
JerkLimitN = {0: -40, 1: -40, 2: -200.5, 3: -110, 4: -40}
# Total_Instance_InSequence checks the number of GPS points for each instance in all users
Total_Instance_InSequence = []
# Total_Motion_Instance: each element is an array include the four channels for each instance
Total_Motion_Instance = []
# Save the 4 channels for each user separately
Total_RelativeDistance = []
Total_Speed = []
Total_Acceleration = []
Total_Jerk = []
Total_BearingRate = []
Total_Label = []
Total_InstanceNumber = []
Total_Outlier = []
Total_Descriptive_Stat = []
Total_Delta_Time = []
Total_Velocity_Change = []
# Count the number of times that NoOfOutlier happens
NoOfOutlier = 0
for z in range(len(Trajectory_Label_Array)):

    Descriptive_Stat = []
    Data = Trajectory_Label_Array[z]
    if len(Data) == 0:
        continue

    Shape = np.shape(Trajectory_Label_Array[z])
    # InstanceNumber: Break a user's trajectory to instances. Count number of GPS points for each instance
    delta_time = []
    tempSpeed = []
    for i in range(len(Data) - 1):
        delta_time.append((Data[i+1, 2] - Data[i, 2]) * 24. * 3600)
        if delta_time[i] == 0:
            # Prevent to generate infinite speed. So use a very short time = 0.1 seconds.
            delta_time[i] = 0.1
        A = (Data[i, 0], Data[i, 1])
        B = (Data[i + 1, 0], Data[i + 1, 1])
        tempSpeed.append(vincenty(A, B).meters/delta_time[i])
    # Since there is no data for the last point, we assume the delta_time as the average time in the user guide
    # (i.e., 3 sec) and speed as tempSpeed equal to last time so far.
    delta_time.append(3)
    tempSpeed.append(tempSpeed[len(tempSpeed) - 1])

    # InstanceNumber: indicate the length of each instance
    InstanceNumber = []
    # Label: For each created instance, we need only one mode to be assigned to.
    # Remove the instance with less than 10 GPS points. Break the whole user's trajectory into trips with min_trip
    # Also break the instance with more than threshold GPS points into more instances
    Data_All_Instance = []  # Each of its element is a list that shows the data for each instance (lat, long, time)
    Label = []
    min_trip_time = 20 * 60  # 20 minutes equal to 1200 seconds
    threshold = 200  # fixed of number of GPS points for each instance
    i = 0
    while i <= (len(Data) - 1):
        No = 0
        ModeType = Data[i, 3]
        Counter = 0
        # index: save the instance indices when an instance is being created and concatenate all in the remove
        index = []
        # First, we always have an instance with one GPS point.
        while i <= (len(Data) - 1) and Data[i, 3] == ModeType and Counter < threshold:
            if delta_time[i] <= min_trip_time:
                Counter += 1
                index.append(i)
                i += 1
            else:
                Counter += 1
                index.append(i)
                i += 1
                break

        if Counter >= 10:  # Remove all instances that have less than 10 GPS points# I
            InstanceNumber.append(Counter)
            Data_For_Instance = [Data[i, 0:3] for i in index]
            Data_For_Instance = np.array(Data_For_Instance, dtype=float)
            Data_All_Instance.append(Data_For_Instance)
            Label.append(ModeType)

    if len(InstanceNumber) == 0:
        continue

    Label = [int(i) for i in Label]

    RelativeDistance = [[] for _ in range(len(InstanceNumber))]
    Speed = [[] for _ in range(len(InstanceNumber))]
    Acceleration = [[] for _ in range(len(InstanceNumber))]
    Jerk = [[] for _ in range(len(InstanceNumber))]
    Bearing = [[] for _ in range(len(InstanceNumber))]
    BearingRate = [[] for _ in range(len(InstanceNumber))]
    Delta_Time = [[] for _ in range(len(InstanceNumber))]
    Velocity_Change = [[] for _ in range(len(InstanceNumber))]
    User_outlier = []
    # Create channels for every instance (k) of the current user
    for k in range(len(InstanceNumber)):
        Data = Data_All_Instance[k]
        # Temp_RD, Temp_SP are temporary relative distance and speed before checking for their length
        Temp_Speed = []
        Temp_RD = []
        outlier = []
        for i in range(len(Data) - 1):
            A = (Data[i, 0], Data[i, 1])
            B = (Data[i+1, 0], Data[i+1, 1])
            Temp_RD.append(vincenty(A, B).meters)
            Delta_Time[k].append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600 + 1)  # Add one second to prevent zero time
            S = Temp_RD[i] / Delta_Time[k][i]
            if S > SpeedLimit[Label[k]] or S < 0:
                outlier.append(i)
            Temp_Speed.append(S)

            y = math.sin(math.radians(Data[i+1, 1]) - math.radians(Data[i, 1])) * math.radians(math.cos(Data[i+1, 0]))
            x = math.radians(math.cos(Data[i, 0])) * math.radians(math.sin(Data[i+1, 0])) - \
                math.radians(math.sin(Data[i, 0])) * math.radians(math.cos(Data[i+1, 0])) \
                * math.radians(math.cos(Data[i+1, 1]) - math.radians(Data[i, 1]))
            # Convert radian from -pi to pi to [0, 360] degree
            b = (math.atan2(y, x) * 180. / math.pi + 360) % 360
            Bearing[k].append(b)

        # End of operation of relative distance, speed, and bearing for one instance
        # Now remove all outliers (exceeding max speed) in the current instance
        Temp_Speed = [i for j, i in enumerate(Temp_Speed) if j not in outlier]
        if len(Temp_Speed) < 10:
            InstanceNumber[k] = 0
            NoOfOutlier += 1
            continue
        Speed[k] = Temp_Speed
        Speed[k].append(Speed[k][-1])

        # Now remove all outlier instances, where their speed exceeds the max speed.
        # Then, remove their corresponding points from other channels.
        RelativeDistance[k] = Temp_RD
        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
        RelativeDistance[k].append(RelativeDistance[k][-1])
        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
        Bearing[k].append(Bearing[k][-1])
        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]

        InstanceNumber[k] = InstanceNumber[k] - len(outlier)

        # Now remove all outlier instances, where their acceleration exceeds the max acceleration.
        # Then, remove their corresponding points from other channels.
        Temp_ACC = []
        outlier = []
        for i in range(len(Speed[k]) - 1):
            DeltaSpeed = Speed[k][i+1] - Speed[k][i]
            ACC = DeltaSpeed/Delta_Time[k][i]
            if abs(ACC) > AccLimit[Label[k]]:
                outlier.append(i)
            Temp_ACC.append(ACC)

        Temp_ACC = [i for j, i in enumerate(Temp_ACC) if j not in outlier]
        if len(Temp_ACC) < 10:
            InstanceNumber[k] = 0
            NoOfOutlier += 1
            continue
        Acceleration[k] = Temp_ACC
        Acceleration[k].append(Acceleration[k][-1])
        Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]
        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]

        InstanceNumber[k] = InstanceNumber[k] - len(outlier)

        # Now remove all outlier instances, where their jerk exceeds the max speed.
        # Then, remove their corresponding points from other channels.

        Temp_J = []
        outlier = []
        for i in range(len(Acceleration[k]) - 1):
            Diff = Acceleration[k][i+1] - Acceleration[k][i]
            J = Diff/Delta_Time[k][i]
            Temp_J.append(J)

        Temp_J = [i for j, i in enumerate(Temp_J) if j not in outlier]
        if len(Temp_J) < 10:
            InstanceNumber[k] = 0
            NoOfOutlier += 1
            continue

        Jerk[k] = Temp_J
        Jerk[k].append(Jerk[k][-1])
        Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]
        Acceleration[k] = [i for j, i in enumerate(Acceleration[k]) if j not in outlier]
        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]

        InstanceNumber[k] = InstanceNumber[k] - len(outlier)
        # End of Jerk outlier detection.

        # Compute Breating Rate from Bearing, and Velocity change from Speed
        for i in range(len(Bearing[k]) - 1):
            Diff = abs(Bearing[k][i+1] - Bearing[k][i])
            BearingRate[k].append(Diff)
        BearingRate[k].append(BearingRate[k][-1])

        for i in range(len(Speed[k]) - 1):
            Diff = abs(Speed[k][i+1] - Speed[k][i])
            if Speed[k][i] != 0:
                Velocity_Change[k].append(Diff/Speed[k][i])
            else:
                Velocity_Change[k].append(1)
        Velocity_Change[k].append(Velocity_Change[k][-1])

        # Now we apply the smoothing filter on each instance:
        def savitzky_golay(y, window_size, order, deriv=0, rate=1):
            r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
            The Savitzky-Golay filter removes high frequency noise from data.
            It has the advantage of preserving the original shape and
            features of the signal better than other types of filtering
            approaches, such as moving averages techniques.
            Parameters
            ----------
            y : array_like, shape (N,)
                the values of the time history of the signal.
            window_size : int
                the length of the window. Must be an odd integer number.
            order : int
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.
            deriv: int
                the order of the derivative to compute (default = 0 means only smoothing)
            Returns
            -------
            ys : ndarray, shape (N)
                the smoothed signal (or it's n-th derivative).
            Notes
            -----
            The Savitzky-Golay is a type of low-pass filter, particularly
            suited for smoothing noisy data. The main idea behind this
            approach is to make for each point a least-square fit with a
            polynomial of high order over a odd-sized window centered at
            the point.
            Examples
            --------
            t = np.linspace(-4, 4, 500)
            y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
            ysg = savitzky_golay(y, window_size=31, order=4)
            import matplotlib.pyplot as plt
            plt.plot(t, y, label='Noisy signal')
            plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
            plt.plot(t, ysg, 'r', label='Filtered signal')
            plt.legend()
            plt.show()
            References
            ----------
            .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
               Data by Simplified Least Squares Procedures. Analytical
               Chemistry, 1964, 36 (8), pp 1627-1639.
            .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
               W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
               Cambridge University Press ISBN-13: 9780521880688
            """
            import numpy as np
            from math import factorial

            try:
                window_size = np.abs(np.int(window_size))
                order = np.abs(np.int(order))
            except ValueError:
                raise ValueError("window_size and order have to be of type int")
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order + 1)
            half_window = (window_size - 1) // 2
            # precompute coefficients
            b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
            m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
            lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve(m[::-1], y, mode='valid')

        # Smoothing process
        RelativeDistance[k] = savitzky_golay(np.array(RelativeDistance[k]), 9, 3)
        Speed[k] = savitzky_golay(np.array(Speed[k]), 9, 3)
        Acceleration[k] = savitzky_golay(np.array(Acceleration[k]), 9, 3)
        Jerk[k] = savitzky_golay(np.array(Jerk[k]), 9, 3)
        BearingRate[k] = savitzky_golay(np.array(BearingRate[k]), 9, 3)

    Total_RelativeDistance.append(RelativeDistance)
    Total_Speed.append(Speed)
    Total_Acceleration.append(Acceleration)
    Total_Jerk.append(Jerk)
    Total_BearingRate.append(BearingRate)
    Total_Delta_Time.append(Delta_Time)
    Total_Velocity_Change.append(Velocity_Change)
    Total_Label.append(Label)
    Total_InstanceNumber.append(InstanceNumber)
    Total_Outlier.append(User_outlier)
    Total_Instance_InSequence = Total_Instance_InSequence + InstanceNumber

with open('Revised_InstanceCreation+NoJerkOutlier+Smoothing.pickle', 'wb') as f:
    pickle.dump([Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Label,
                 Total_InstanceNumber, Total_Instance_InSequence, Total_Delta_Time, Total_Velocity_Change], f)
