import os
import sys
import shutil
import math
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
from enum import Enum
from scipy.spatial import Voronoi
sys.path.append("/home/esraan/CellDeathSpreading/src/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times

def get_real_distance(cell1_xy, cell2_xy):
    """
    calculte the actual euclidean distance between cells
    Args:
        cell1_xy (tuple): x, y cells location
        cell2_xy (tuple): x, y cells location
    Returns:
        int, float: euclidean distance
    """
    cell1_x, cell1_y = cell1_xy
    cell2_x, cell2_y = cell2_xy
    return ((cell1_x - cell2_x)**2 + (cell1_y - cell2_y)**2)**.5

def get_neighbors(XY, dist_threshold=100, filter_neighbors_by_distance=1, filter_neighbors_by_level=3):
    """get morphological/structural niegbors of each cells based on the parameters given
    Args:
        XY (np.array): np.array of cells location
        dist_threshold (int, optional): in microns. Defaults to 100.
        filter_neighbors_by_distance (int, optional): control filtering by distance (0,1- True, False for filtering by distance or not). Defaults to 1.
        filter_neighbors_by_level (int, optional): control filtering by level, chose niegbors from degree 1, degree 2, degree 3. Defaults to 3.
    Returns:
        list|tuple[list]: according to selected parmaters, list of nieghbors are returned, first degree, second degree, and third degree
    """
    vor = Voronoi(XY)
    neighbors = vor.ridge_points
    neighbors_list = []
    neighbors_list2 = []
    neighbors_list3 = []
    for i in range(len(XY)):
        neighbors_list.append([])
        neighbors_list2.append([])
        neighbors_list3.append([])
    for x in neighbors:
        neighbors_list[x[0]].append(x[1])
        neighbors_list[x[1]].append(x[0])
    for i in range(len(XY)):
        for j in neighbors_list[i]:
            neighbors_list2[i] = list(set(neighbors_list2[i]+neighbors_list[j]))
    for i in range(len(XY)):
        for j in neighbors_list2[i]:
            neighbors_list3[i] = list(set(neighbors_list3[i]+neighbors_list2[j]))
    if filter_neighbors_by_distance==1:
        for i in range(len(XY)):
            neighbors_list[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list[i]))
            neighbors_list2[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold , neighbors_list2[i]))
            neighbors_list3[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list3[i]))
    if filter_neighbors_by_level== 1:
        return neighbors_list, [],[]
    elif filter_neighbors_by_level == 2:
        neighbors_list2 = [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))]
        return neighbors_list, neighbors_list2, []
    elif filter_neighbors_by_level == 3:
        neighbors_list2 = [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))]
        neighbors_list3 = [list(set(neighbors_list3[i]) - set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list3))]
        return neighbors_list, neighbors_list2, neighbors_list3
    return neighbors_list, [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))], [list(set(neighbors_list3[i]) - set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list3))]

def get_time_difference(death_times, cell1_idx, cell2_idx):
    """
    calculate the time difference between two cells death times
    Args:
        death_times (np.array): array of cells death times
        cell1_idx (int): index of first cell
        cell2_idx (int): index of second cell
    Returns:
        int, float: time difference in seconds
    """
    return abs(death_times[cell1_idx] - death_times[cell2_idx])