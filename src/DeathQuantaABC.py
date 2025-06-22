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
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
from abc import ABC, abstractmethod
sys.path.append("/home/esraan/CellDeathSpreading/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times
from src.quanta_utils import get_neighbors, get_time_difference, normalize_death_times

class DeathQuanta(ABC):
    def __init__(self,
                cells_xy: np.ndarray, 
                death_times: np.ndarray,
                death_modes: Union[np.ndarray,list] = None,
                dist_threshold: Union[int, float] = 50,
                filter_neighbors_by_distance: bool = True,
                neighbors_level: int = 1,
                **kwargs
                )-> None:
        try:
            sorted_indices = np.argsort(np.concatenate(death_times))
            flag = False
        except Exception as e:
            flag = True
            sorted_indices = np.argsort(death_times)
        self.death_times = death_times[sorted_indices]
        self.cells_xy = cells_xy[sorted_indices]
        self.dist_threshold = dist_threshold
        self.le = LabelEncoder()
        self.filter_neighbors_by_distance = filter_neighbors_by_distance
        self.neighbors_level = neighbors_level
        self._init_other_attr(**kwargs)
        if not flag:
            self.death_modes = np.concatenate(death_modes[sorted_indices]) if death_modes is not None else np.array([self.pure_type] * len(self.cells_xy))
        else:
            self.death_modes = death_modes[sorted_indices] if death_modes is not None else np.array([self.pure_type] * len(self.cells_xy))
        if self.normalize:
            self.death_times = normalize_death_times(self.death_times, kwargs.get('n_type', 'median_and_percentile_range'))
        self.death_modes = self.le.fit_transform(self.death_modes)
        self.nieghbors_levels_init(neighbors_level)
    
    def nieghbors_levels_init(self, neighbors_level):
        """
        Initialize neighbors based on the distance threshold and level.
        This method sets up the neighbors for each cell based on the specified distance threshold and level.
        """
        if self.filter_neighbors_by_distance: 
            if self.filter_neighbors_by_level:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.cells_xy, self.dist_threshold, True, neighbors_level)
            else:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.cells_xy, self.dist_threshold, True, 3)
        elif self.filter_neighbors_by_level:
            if not self.filter_neighbors_by_distance:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.cells_xy, self.dist_threshold, False, neighbors_level)
        else:
            self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.cells_xy, self.dist_threshold, False, 3)
    
    def _init_other_attr(self, **kwargs):
        self.filter_neighbors_by_level = kwargs.get("filter_neighbors_by_level", True)
        self.pure_type = kwargs.get('pure_type', 'necrosis')
        self.n_permutation = kwargs.get('n_permutation', 1000)
        self.normalize = True if kwargs.get('normalize', True) else False

    @abstractmethod
    def create_scramble(self):
        pass