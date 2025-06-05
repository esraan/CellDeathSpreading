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

class mSpiCalc(SpiCalc):
    """ Median-based SPI calculation"""
    # die_times must be in frames number, dist_threshold is assumed to be in micron
    def __init__(self, XY, die_times, temporal_resolution, n_scramble=1000,
                dist_threshold=100, **kwargs):
        """ 
        Median-SPI calculator
        Args:
            XY (np.array): location numpy array of cells
            die_times (np.array): death times of cells
            temporal_resolution (int): temporal resultion if needed to convert times from frames. needs to set time_unit t0 frames.
            n_scramble (int, optional): _description_. Defaults to 1000.
            dist_threshold (int, optional): set distant to the value where neighbors cells are considered for calculation. Defaults to 100 micron. Don't use pixels!
            set filter_neighbors_by_distance, filter_neighbors_by_level, for further control on neighbors thresholding.
        """
        super().__init__( XY, die_times, temporal_resolution, n_scramble,
                dist_threshold, **kwargs)
        self.create_scramble()
        self.neighbors_difference_death_times = self.get_neighbors_difference_death_times()
        self.original_difference_death_times = np.median(self.neighbors_difference_death_times[0])
        self.scramble_signficance_95, self.median_time_death, self.scramble_median_time_death, self.scramble_signficance_98, self.propagation_index = (0,0,0,0,-1)
        self.statistic_score, self.all_shuffled_median_sorted = self.assess_stat()
        self.propagation_index = ((self.scramble_signficance_95 - self.original_difference_death_times) / self.scramble_signficance_95)

    def get_mspis(self):
        """calculate median-SPI

        Returns:
            mspi (float): calculated spi value in the constructor
        """
        return self.propagation_index

    def assess_stat(self):
        """calculate the median of TODs from neigbors.

        Returns:
            tuple(int,int): p-value of the result, the actual result
        """
        better_median = 0
        time_death_medians = []
        if self.time_unit == 'frames':
            self.median_time_death = self.original_difference_death_times * self.temporal_resolution
        else:
            self.median_time_death = self.original_difference_death_times
        for i in range(self.n_scramble):
            temp_median_time_death = self.calc_stat(self.neighbors_difference_death_times[i + 1])
            time_death_medians.append(temp_median_time_death)
            if temp_median_time_death > self.original_difference_death_times:
                better_median += 1
        time_death_medians.sort()
        if self.time_unit == 'frames':
            self.scramble_signficance_95 = time_death_medians[int(self.n_scramble * 5 / 100)] * self.temporal_resolution
            self.scramble_signficance_98 = time_death_medians[int(self.n_scramble * 2 / 100)] * self.temporal_resolution
        else:
            self.scramble_signficance_95 = time_death_medians[int(self.n_scramble * 5 / 100)]
            self.scramble_signficance_98 = time_death_medians[int(self.n_scramble * 2 / 100)]
        return better_median / self.n_scramble, time_death_medians

    def calc_stat(self, dist_for_calc):
        """ calculate median of TODs"""
        return np.median(dist_for_calc)


if __name__ == '__main__':
    single_exp_full_path = '/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeMinutesData/20160820_10A_FB_xy11.csv'
    cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
    ob = mSpiCalc(XY=cells_loci,
                        die_times=cells_times_of_death,
                        temporal_resolution=30,
                        n_scramble=1000,
                        dist_threshold=200,
                        filter_neighbors_by_distance=True,
                        filter_neighbors_by_level= 3,
                        time_unit='minutes',)
    print(ob.get_mspis())