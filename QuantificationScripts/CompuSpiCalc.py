import os
import math
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
sys.path.append("/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/")
from utils import read_experiment_cell_xy_and_death_times
from QuantificationScripts.SpiCalc import SpiCalc

class CompuSpiCalc(SpiCalc):
    # die_times must be in frames number, dist_threshold is assumed to be in micron
    def __init__(self, XY, die_times, temporal_resolution, n_scramble=1000,
                 dist_threshold=100, **kwargs):

        super().__init__( XY, die_times, temporal_resolution, n_scramble,
                dist_threshold, **kwargs)
        super().create_sramble()
        self.nuc_leader_and_their_community = {}
        self.neighbors_difference_death_times = self.get_neighbors_difference_death_times()
        self.original_difference_death_times = self.calc_stat(self.neighbors_difference_death_times[0])
        self.scramble_signficance_95, self.scramble_signficance_98, self.scramble_mean_time_death, self.propagation_index = (0,0,0,-1)
        self.statistic_score, self.all_mean_shuffles_sorted = self.assess_stat()
        self.propagation_index = ((self.scramble_signficance_95 - self.original_difference_death_times) / self.scramble_signficance_95)

    def get_uspis(self):
        return self.propagation_index

    def assess_stat(self):
        better_mean = 0
        time_death_means = []
        real_mean_time_death = self.calc_stat(self.neighbors_difference_death_times[0])
        if self.time_unit == 'frames':
             self.mean_time_death = real_mean_time_death * self.temporal_resolution
        else:
             self.mean_time_death = real_mean_time_death 
        for i in range(self.n_scramble):
            temp_mean_time_death = self.calc_stat(self.neighbors_difference_death_times[i + 1])
            time_death_means.append(temp_mean_time_death)
            if temp_mean_time_death < real_mean_time_death:
                better_mean += 1
        time_death_means.sort()
        if self.time_unit == 'frames':
            self.scramble_signficance_95 = time_death_means[int(self.n_scramble * 5 / 100)] * self.temporal_resolution
            self.scramble_signficance_98 = time_death_means[int(self.n_scramble * 2 / 100)] * self.temporal_resolution
            self.scramble_mean_time_death = (sum(time_death_means) / len(time_death_means)) * self.temporal_resolution
        else:
            self.scramble_signficance_95 = time_death_means[int(self.n_scramble * 5 / 100)]
            self.scramble_signficance_98 = time_death_means[int(self.n_scramble * 2 / 100)]
            self.scramble_mean_time_death = sum(time_death_means) / len(time_death_means)
        return better_mean / self.n_scramble , time_death_means

    def calc_stat(self, dist_for_calc):
        return np.nanmean(dist_for_calc)
    
    def get_time_from_neighbors(self, times, XY):
        """get time of death of neighbors

        Args:
            times (np.array): time of death for all cells (could be pernuted, not nessecly observed)
            XY (np.array): cells location, if needed, neigbors can be calculted each time. not optimized

        Returns:
            array: array of TOD's from the chosen niegbors.
        """
        time_diff_from_nighbors_list = []
        if self.filter_neighbors_by_distance==1 or self.filter_neighbors_by_distance==0:
            #TODO: need optimization, in case we are back to shuffiling xy and not times of death - as we calculate the lists each time
            if self.find_nighbors_each_time:
                neighbors_level1, neighbors_level2, neighbors_level3 = SpiCalc.get_neighbors(XY, self.dist_threshold, self.filter_neighbors_by_distance, self.filter_neighbors_by_level)
            else:
                neighbors_level1 = self.neighbors_list
                neighbors_level2 = self.neighbors_list2
                neighbors_level3 = self.neighbors_list3
            for idx in range(self.n_instances):
                tod_of_single_cell_first_degree = [abs(times[idx] - times[neighbor]) for neighbor in neighbors_level1[idx]]
                # time_diff_from_nighbors_list.append(self.calc_stat(tod_of_single_cell_first_degree))
                if self.filter_neighbors_by_level==1:
                    time_diff_from_nighbors_list.append(self.calc_stat(tod_of_single_cell_first_degree))
                    continue
                tod_of_single_cell_second_degree = [abs(times[idx] - times[neighbor]) for neighbor in neighbors_level2[idx]]
                if self.filter_neighbors_by_level==2:
                    time_diff_from_nighbors_list.append(self.calc_stat(tod_of_single_cell_first_degree+tod_of_single_cell_second_degree))
                    continue
                tod_of_single_cell_third_degree = [abs(times[idx] - times[neighbor]) for neighbor in neighbors_level3[idx]]
                time_diff_from_nighbors_list.append(self.calc_stat(tod_of_single_cell_first_degree+tod_of_single_cell_second_degree+tod_of_single_cell_third_degree))
                # if tod_of_single_cell_first_degree == [] or tod_of_single_cell_second_degree == [] or tod_of_single_cell_third_degree ==[]:
                #     print('no nighbors')
        else:
            vor = Voronoi(XY)
            neighbors = vor.ridge_points
            for i in range(len(neighbors)):
                time_diff_from_nighbors_list.append(abs(times[neighbors[i][0]] - times[neighbors[i][1]]))
        return np.array(time_diff_from_nighbors_list)

    def get_neighbors_difference_death_times(self):
        """create the full permutation of cell detah time as n_scramble

        Returns:
            list[arrays[int]]: list of as n_scramble expremnt of permuted death time on the population
        """
        TODs_dist = []
        TODs_dist.append(self.get_time_from_neighbors(self.die_times, self.XY))
        for i in range(self.n_scramble):
            TODs_dist.append(self.get_time_from_neighbors(self.scrambles[i], self.XY)) #self.die_times, self.scrambles[i]))
        return TODs_dist

if __name__ == '__main__':
    single_exp_full_path = '/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeMinutesData/20160820_10A_FB_xy11.csv'
    cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path, need_sorting=True)
    ob = CompuSpiCalc(XY=cells_loci,
                        die_times=cells_times_of_death,
                        treatment='FB',
                        temporal_resolution=30,
                        n_scramble=1000,
                        draw=False,
                        dist_threshold=300, 
                        filter_neighbors_by_distance=True,
                        filter_neighbors_by_level= 3,
                        time_unit='minutes',)
    print(ob.get_uspis())
    print(ob.get_stat_score())
    print(ob.find_nucluator(3))
    to_val = ob.nuc_leader_and_their_community
    # Count the occurrences of a specific value in the dictionary keys and values
    specific_value = 3  # Replace with the value you want to count
    for i in range(ob.n_instances):
        specific_value = i
        count = sum(1 for key, values in to_val.items() if key == specific_value or specific_value in values)
        print(f"The value {specific_value} appears {count} times in the dictionary keys and values.")
    print(f'number of pockets: {len(to_val.keys())}')
    # experiments_dir = '/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeMinutesData/'
    # experiment_files = glob.glob(os.path.join(experiments_dir, '*.csv'))

    # for single_exp_full_path in experiment_files:
    #     print(f"Processing file: {single_exp_full_path}")
    #     cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
    #     ob = CompuSpiCalc(XY=cells_loci,
    #                   die_times=cells_times_of_death,
    #                   treatment='',
    #                   temporal_resolution=10,
    #                   n_scramble=1000,
    #                   draw=False,
    #                   dist_threshold=200, 
    #                   filter_neighbors_by_distance=True,
    #                   filter_neighbors_by_level=3,
    #                   time_unit='minutes',)
    #     print(f"Propagation Index (uSpi): {ob.get_uspis()}")
    #     print(f"Statistic Score: {ob.get_stat_score()}")