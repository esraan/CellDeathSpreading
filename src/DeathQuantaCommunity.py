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
sys.path.append("/home/esraan/CellDeathSpreading/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times
from src.quanta_utils import get_neighbors, get_time_difference, normalize_death_times

class DeathQuanta:
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
        self.n_scramble = kwargs.get('n_scramble', 1000)
        self.normalize = True if kwargs.get('normalize', True) else False

    def find_nucleator(self, level):
        nuc_blobs_identifier = {}
        set_of_all_cells = set()
        for cell_idx, loci in enumerate(self.cells_xy):
            if cell_idx in set_of_all_cells:
                continue
            if level > self.neighbors_level:
                raise ValueError("Level don't corspond to the attributet level!")
            elif level == 3:
                level_1, level_2, level_3 = self.neighbors_level_1[cell_idx], self.neighbors_level_2[cell_idx], self.neighbors_level_3[cell_idx]
            elif level == 2:
                level_1, level_2, level_3 = self.neighbors_level_1[cell_idx], self.neighbors_level_2[cell_idx],[]
            elif level == 1:
                level_1, level_2, level_3 = self.neighbors_level_1[cell_idx],[],[]
            all_level_niegbors_in_dist_thr = level_1 + level_2 + level_3
            # any_smaller_die_times_neighbors = all([self.death_times[neighbor] >= self.death_times[cell_idx] for neighbor in all_level_niegbors_in_dist_thr])
            any_smaller_die_times_neighbors = all([self.death_times[neighbor] >= self.death_times[cell_idx] for neighbor in all_level_niegbors_in_dist_thr])
            if not any_smaller_die_times_neighbors:
                to_remove = [neighbor for neighbor in all_level_niegbors_in_dist_thr if (self.death_times[neighbor] < self.death_times[cell_idx] or neighbor in set_of_all_cells)]
                if len(to_remove)>0:
                    nuc_blobs_identifier[cell_idx]= list(set(all_level_niegbors_in_dist_thr)-set(to_remove))
                    set_of_all_cells.add(cell_idx)
                    set_of_all_cells.update(set(nuc_blobs_identifier.get(cell_idx)))
                else:
                    continue
            else: 
                #the cell is the first to die but might have niegbors that were discovered already
                set_new = set(all_level_niegbors_in_dist_thr)
                set_new = set_new - set_of_all_cells       
                nuc_blobs_identifier[cell_idx] = list(set_new)
                set_of_all_cells.add(cell_idx)
                set_of_all_cells.update(set_new)
        for leader, community in nuc_blobs_identifier.items():
            nuc_blobs_identifier[leader] = [cell for cell in community if self.death_modes[cell] == self.death_modes[leader]]
        self.nuc_leader_and_their_community = nuc_blobs_identifier
        return nuc_blobs_identifier

    def get_TOD_from_nucleator(self, level, permute= False):
        if not hasattr(self, 'nuc_leader_and_their_community'):
            self.find_nucleator(level)
        tods = {}
        death_times_cp = self.death_times.copy()
        for leader, community in self.nuc_leader_and_their_community.items():
            if permute:
                death_times_cp = self.permute_death_times_single_commonity(community, self.death_times)
            if level == 3:
                tods[leader] = [get_time_difference(death_times_cp, leader, cell) for cell in community]
            elif level == 2:
                needed_cells = list(set(community) & set(self.neighbors_level_1[leader])) + list(set(community) & set(self.neighbors_level_2[leader]))
                needed_cells = set(needed_cells)
                tods[leader] = [get_time_difference(death_times_cp, leader, cell) for cell in needed_cells]
            elif level == 1:
                needed_cells = list(set(community) & set(self.neighbors_level_1[leader]))
                needed_cells = set(needed_cells)
                tods[leader] = [get_time_difference(death_times_cp, leader, cell) for cell in needed_cells]
            else:
                raise ValueError("Level don't corspond to the attributet level!")
        return tods
    
    def permute_death_times_single_commonity(self, community, original_death_times):
        """
        Permute the death times of a single community.
        Args:
            community (list): List of cell indices in the community.
            original_death_times (np.ndarray): Original death times of all cells.
        Returns:
            np.ndarray: Permuted death times for the community.
        """
        permuted_death_times = original_death_times.copy()
        shuffled_times = permuted_death_times[community].copy()
        np.random.shuffle(shuffled_times)
        permuted_death_times[community] = shuffled_times
        return permuted_death_times
    
    def _calculate_deltaTOD_neighbors_pairs(self, actual_death_times = None):
        single_mode_tods_pairs = {item:[] for item in self.le.classes_}
        if not actual_death_times:
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
    
    def get_org_deltaTOD_neighbors_pairs(self):
        if len(self.le.classes_) < 2:
            return self._calculate_deltaTOD_neighbors_pairs().get(self.pure_type, [])
        return self._calculate_deltaTOD_neighbors_pairs()
    
    def create_scramble(self):
        self.scrambles = []
        for _ in range(self.n_scramble):
            result = self.death_times.copy()
            for mode in np.unique(self.death_modes):
                mask = self.death_modes == mode
                shuffled_withen = self.death_times[mask].copy()
                np.random.shuffle(shuffled_withen)
                result[mask] = shuffled_withen
            self.scrambles.append(result)
        
    def find_commonities(self, level):
        nuc_blobs_identifier = {}
        set_of_all_cells = set()
        for idx, cell in enumerate(self.cells_xy):
            if idx in set_of_all_cells:
                continue
            if level > self.neighbors_level:
                raise ValueError("Level don't corspond to the attributet level!")
            elif level == 3:
                level_1, level_2, level_3 = self.neighbors_level_1[idx], self.neighbors_level_2[idx], self.neighbors_level_3[idx]
            elif level == 2:
                level_1, level_2, level_3 = self.neighbors_level_1[idx], self.neighbors_level_2[idx], []
            elif level == 1:
                level_1, level_2, level_3 = self.neighbors_level_1[idx], [], []
            all_level_niegbors_in_dist_thr = level_1 + level_2 + level_3
            if len(all_level_niegbors_in_dist_thr) == 0:
                continue
            # Filter neighbors by death mode
            all_level_niegbors_in_dist_thr = [neighbor for neighbor in all_level_niegbors_in_dist_thr if self.death_modes[neighbor] == self.death_modes[idx]]
            any_smaller_die_times_neighbors = all([self.death_times[neighbor] >= self.death_times[idx] for neighbor in all_level_niegbors_in_dist_thr])
            if not any_smaller_die_times_neighbors:
                to_remove = [neighbor for neighbor in all_level_niegbors_in_dist_thr if (self.death_times[neighbor] < self.death_times[idx] or neighbor in set_of_all_cells)]
                if len(to_remove)>0:
                    nuc_blobs_identifier[idx]= list(set(all_level_niegbors_in_dist_thr)-set(to_remove))
                    set_of_all_cells.add(idx)
                    set_of_all_cells.update(set(nuc_blobs_identifier.get(idx)))
                else:
                    continue
            else: 
                #the cell is the first to die but might have niegbors that were discovered already
                set_new = set(all_level_niegbors_in_dist_thr)
                set_new = set_new - set_of_all_cells  
                blobs = [self.death_times[cell] >= self.death_times[idx] for cell in set_new ]   
                if not all(blobs):
                    continue
                nuc_blobs_identifier[idx] = list(set_new)
                set_of_all_cells.add(idx)
                set_of_all_cells.update(set_new)
        for leader, community in nuc_blobs_identifier.items():
            nuc_blobs_identifier[leader] = np.array(community)
        return nuc_blobs_identifier, set_of_all_cells



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