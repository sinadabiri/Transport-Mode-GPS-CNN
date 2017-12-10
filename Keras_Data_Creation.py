import numpy as np
import pickle
import os
from scipy.signal import savgol_filter

filename = '../Combined Trajectory_Label_Geolife/Revised_InstanceCreation+NoJerkOutlier+Smoothing.pickle'
# Each of the following variables contain multiple lists, where each list belongs to a user
with open(filename, 'rb') as f:
    Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Label,\
    Total_InstanceNumber, Total_Instance_InSequence, Total_Delta_Time, Total_Velocity_Change = pickle.load(f, encoding='latin1')

# Create the data in the Keras form
# Threshold: Is the max of number of GPS point in an instance

Threshold = 200
Zero_Instance = [i for i, item in enumerate(Total_Instance_InSequence) if item == 0]
Number_of_Instance = len(Total_Instance_InSequence) - len(Zero_Instance)
TotalInput = np.zeros((Number_of_Instance, 1, Threshold, 4), dtype=float)
FinalLabel = np.zeros((Number_of_Instance, 1), dtype=int)
counter = 0

for k in range(len(Total_InstanceNumber)):
    # Create Keras shape with 4 channels for each user
    #  There are 4 channels(in order: RelativeDistance, Speed, Acceleration, BearingRate)
    RD = Total_RelativeDistance[k]
    SP = Total_Speed[k]
    AC = Total_Acceleration[k]
    J = Total_Jerk[k]
    BR = Total_BearingRate[k]
    LA = Total_Label[k]
    # IN: the instances and number of GPS points in each instance for each user k
    IN = Total_InstanceNumber[k]

    for i in range(len(IN)):
        end = IN[i]
        if end == 0 or sum(RD[i]) == 0:
            continue
        TotalInput[counter, 0, 0:end, 0] = SP[i]
        TotalInput[counter, 0, 0:end, 1] = AC[i]
        TotalInput[counter, 0, 0:end, 2] = J[i]
        TotalInput[counter, 0, 0:end, 3] = BR[i]
        FinalLabel[counter, 0] = LA[i]
        counter += 1

TotalInput = TotalInput[:counter, :, :, :]
FinalLabel = FinalLabel[:counter, 0]

with open('Revised_KerasData_NoSmoothing.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([TotalInput, FinalLabel], f)

