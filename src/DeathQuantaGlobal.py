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
from itertools import chain
from scipy.spatial import Voronoi
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance
sys.path.append("/home/esraan/CellDeathSpreading/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times
from src.quanta_utils import get_neighbors, get_time_difference, normalize_death_times
from src.DeathQuanta import DeathQuanta

class DeathQuantaGlobal(DeathQuanta):
    def __init__(self,
                cells_xy: np.ndarray, 
                death_times: np.ndarray,
                death_modes: Union[np.ndarray,list] = None,
                dist_threshold: Union[int, float] = 50,
                filter_neighbors_by_distance: bool = True,
                neighbors_level: int = 1,
                **kwargs
                )-> None:
        super().__init__(cells_xy, death_times, death_modes, dist_threshold, filter_neighbors_by_distance, neighbors_level, **kwargs)
        self.create_scramble()

    def _calculate_deltaTOD_neighbors_pairs(self, actual_death_times = None):
        single_mode_tods_pairs = {item:[] for item in self.le.classes_}
        if actual_death_times is None:
            actual_death_times = self.death_times
        for cell_idx in range(len(actual_death_times)):
            temp_tod_for_cell = [abs(actual_death_times[cell_idx] - actual_death_times[neighbor]) for neighbor in self.neighbors_level_1[cell_idx] if (self.le.inverse_transform([self.death_modes[cell_idx]])[0] == self.le.inverse_transform([self.death_modes[neighbor]])[0])]
            single_mode_tods_pairs.get(self.le.inverse_transform([self.death_modes[cell_idx]])[0]).extend(temp_tod_for_cell)
            if self.filter_neighbors_by_level==1:
                continue
            temp_tod_for_cell = [abs(actual_death_times[cell_idx] - actual_death_times[neighbor]) for neighbor in self.neighbors_level_2[cell_idx] if (self.le.inverse_transform([self.death_modes[cell_idx]])[0] == self.le.inverse_transform([self.death_modes[neighbor]])[0])]
            single_mode_tods_pairs.get(self.le.inverse_transform([self.death_modes[cell_idx]])[0]).extend(temp_tod_for_cell)
            if self.filter_neighbors_by_level==2:
                continue
            temp_tod_for_cell = [abs(actual_death_times[cell_idx] - actual_death_times[neighbor]) for neighbor in self.neighbors_level_3[cell_idx] if (self.le.inverse_transform([self.death_modes[cell_idx]])[0] == self.le.inverse_transform([self.death_modes[neighbor]])[0])]
            single_mode_tods_pairs.get(self.le.inverse_transform([self.death_modes[cell_idx]])[0]).extend(temp_tod_for_cell)
        return single_mode_tods_pairs
    
    def get_org_deltaTOD_neighbors_pairs(self, uniform_deltaTOD:bool = False):
        if len(self.le.classes_) < 2:
            self.org_TOD_pairs = self._calculate_deltaTOD_neighbors_pairs().get(self.pure_type, [])
            return self.org_TOD_pairs
        self.org_TOD_pairs = self._calculate_deltaTOD_neighbors_pairs()
        if uniform_deltaTOD:
            self.org_TOD_pairs = np.array(list(chain.from_iterable([self.org_TOD_pairs.get(mode, []) for mode in self.le.classes_])))
        return self.org_TOD_pairs
    
    def create_scramble(self):
        self.permuted_death_time = []
        for _ in range(self.n_permutation):
            result = self.death_times.copy()
            for mode in np.unique(self.death_modes):
                mask = self.death_modes == mode
                shuffled_withen = self.death_times[mask].copy()
                np.random.shuffle(shuffled_withen)
                result[mask] = shuffled_withen
            self.permuted_death_time.append(result)
        
    def get_permuted_deltaTOD_neighbors_pairs(self, uniform_deltaTOD:bool = True):
        if uniform_deltaTOD:
            permuted_deltaTOD_pairs = []
            permuted_deltaTOD_pairs_means = []
            for idx in range(self.n_permutation):
                org_TOD_pairs = self._calculate_deltaTOD_neighbors_pairs(self.permuted_death_time[idx])
                permuted_deltaTOD_pairs.append(np.array(list(chain.from_iterable([org_TOD_pairs.get(mode, []) for mode in self.le.classes_]))))
                permuted_deltaTOD_pairs_means.append(np.mean(permuted_deltaTOD_pairs[-1]))
            threshold = np.percentile(permuted_deltaTOD_pairs_means,5)
            indices = [i for i, score in enumerate(permuted_deltaTOD_pairs_means) if score <= threshold]
            self.permuted_deltaTOD_pairs_means = permuted_deltaTOD_pairs_means
            self.permuted_deltaTOD_pairs = permuted_deltaTOD_pairs
            return permuted_deltaTOD_pairs_means[min(indices)] if indices else None, permuted_deltaTOD_pairs[min(indices)]
            
        else:
            permuted_deltaTOD_pairs = {item:[] for item in self.le.classes_}
            permuted_deltaTOD_pairs_means = []
            for idx in range(self.n_permutation):
                permuted_deltaTOD_pairs.append(self._calculate_deltaTOD_neighbors_pairs(self.permuted_death_time[idx]))
                permuted_deltaTOD_pairs_means.append({mode: np.mean(permuted_deltaTOD_pairs[-1].get(mode, [])) for mode in permuted_deltaTOD_pairs[-1].keys()})
            scores = np.array([np.mean(list(d.values())) for d in permuted_deltaTOD_pairs_means])
            threshold = np.percentile(scores, 5)
            indices = [i for i, score in enumerate(scores) if score <= threshold]
            self.permuted_deltaTOD_pairs_means = permuted_deltaTOD_pairs_means
            self.permuted_deltaTOD_pairs = permuted_deltaTOD_pairs
            return permuted_deltaTOD_pairs_means[min(indices)] if indices else None, permuted_deltaTOD_pairs[min(indices)]
    def calc_p_val(self):
        pass
        




if __name__ == "__main__":
    # Example usage
    # OLD DATA
    exps_dir_name = "/sise/assafzar-group/assafzar/Esraa/Others/fully_annotated_data/TimeFrames/"
    meta_data_file_full_path= "/sise/assafzar-group/assafzar/Esraa/Others/ManuallyAnnotatedRoisuu.csv"
    # meta_data_extract_exp_names= pd.read_csv(meta_data_file_full_path)
    # exp_names = meta_data_extract_exp_names.iloc[:,1]
    # print(exp_names[0])
    # exp_full_path = os.path.join(exps_dir_name, exp_names[0])
    # csv_file = pd.read_csv(exp_full_path)
    # dist_threshold = 100
    # cells_location = csv_file[["cell_x","cell_y"]].values
    # di_times = csv_file[['death_time']].values
    # death_modes = csv_file[["Mode"]].values
    # de_qu = DeathQuanta(cells_xy=cells_location,
    #                     death_times=di_times, 
    #                     death_modes=death_modes,
    #                     dist_threshold=dist_threshold,
    #                     filter_neighbors_by_distance=True,
    #                     neighbors_level=1)
    # tod_pairs = de_qu.get_org_deltaTOD_neighbors_pairs()
    # print(tod_pairs)