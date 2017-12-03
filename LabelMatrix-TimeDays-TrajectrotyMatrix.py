import numpy as np
import pickle
from datetime import datetime
import os

# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Transport Project Mode\Combined Trajectory_Label_Geolife')

# create 'daysDate' function to convert start and end time to a number of days


def days_date(time_str):
    date_format = "%Y/%m/%d %H:%M:%S"
    current = datetime.strptime(time_str, date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30', date_format)
    no_days = current - bench
    delta_time_days = no_days.days + current.hour / 24.0 + current.minute / (24. * 60.) + current.second / (24. * 3600.)
    return delta_time_days

# Change Mode Name to Mode index
Mode_Index = {"walk": 0, "run": 9, "bike": 1, "bus": 2, "car": 3, "taxi": 3, "subway": 4, "railway": 4,
              "train": 4, "motocycle": 8, "boat": 9, "airplane": 9, "other": 9}

# Ground modes are the modes that we use in this paper.
Ground_Mode = ['walk', 'bike', 'bus', 'car', 'taxi', 'subway', 'railway', 'train']

# Trajectory_Array and Label_Array are the final lists which each of its element is for one user
Trajectory_Array = []
Label_Array = []
Trajectory_Label_Array = []
UserNon = range(180)

# 1
for k in range(len(UserNon)):
    InputFile = "combined" + str(UserNon[k]) + ".plt"
    table = []
    try:
        with open(InputFile, 'rb') as inp:
            for row in inp:
                row = row.rstrip()
                row = row.decode("utf-8")
                row = row.split(',')
                if len(row) == 7:
                    table.append(row)
    except IOError:
        continue

# TrajectoryMatrix = contains lat, long, date in each column
    table_array = np.array(table, dtype=object)
    TrajectoryMatrix = np.stack((table_array[:, 0], table_array[:, 1], table_array[:, 4]), axis=-1)
    for i in range(len(table_array[:, 0])):
        for j in range(3):
            TrajectoryMatrix[i, j] = float(TrajectoryMatrix[i, j])

    Trajectory_Array.append(TrajectoryMatrix)


# end1
# 2.Modify the labels file and create array with start_time, end_time in days and labels
    InputFile = "labels" + str(UserNon[k]) + ".txt"
    table = []
    with open(InputFile, 'rb') as inp:
        for row in inp:
            row = row.rstrip()
            row = row.decode("utf-8")
            row = row.split('\t')
            if len(row) == 3:
                table.append(row)

    LabelFile = np.array(table, dtype=object)
# StartTime and EndTime in days after 1899/12/30 for each data point in labels.cv
# Modify label for those rows that don't have any time and labels
# LabelMatrix = the array that has Start time(days), End time(days), and labels

    StartTime = []
    EndTime = []
    label = []
    Error = []

    for i in range(len(LabelFile[:, 0])):
        try:
            if LabelFile[i, 2] in Ground_Mode:
                StartTime.append(days_date(LabelFile[i, 0]))
                EndTime.append(days_date(LabelFile[i, 1]))
                label.append(Mode_Index[LabelFile[i, 2]])
        except ValueError:
            Error.append(i)

    LabelMatrix = (np.vstack((StartTime, EndTime, label))).T
    Label_Array.append(LabelMatrix)

    # End2
    # 3.Assign the labels to the trajectories
    # Trajectory = zip(lat, long, date)
    Dates = np.split(TrajectoryMatrix, 3, axis=-1)[2]
    Sec = 1 / (24.0 * 3600.0)
    # C_list: all the rows in the TrajectoryMatrix that should be picked up
    C_list = []
    # Mode_Trajectory: all labels
    Mode_Trajectory = []
    for index, row in enumerate(LabelMatrix):
        A = np.where(Dates >= (float(row[0]) - Sec))
        B = np.where(Dates <= (float(row[1]) + Sec))
        C = list(set(A[0]).intersection(B[0]))
        if len(C) == 0:
            print("error")
        [Mode_Trajectory.append(row[2]) for i in C]
        [C_list.append(i) for i in C]

    TrajectoryMatrix = [TrajectoryMatrix[i, :] for i in C_list]
    TrajectoryMatrix = np.array(TrajectoryMatrix)
    Mode_Trajectory = np.array(Mode_Trajectory)
    Trajectory_Label = (np.vstack((TrajectoryMatrix.T, Mode_Trajectory))).T

    Trajectory_Label_Array.append(Trajectory_Label)

    # End3

# Save Trajectory_Array and Label_Array for all users
with open("Revised_Trajectory_Label_Array.pickle", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Trajectory_Label_Array, f)
