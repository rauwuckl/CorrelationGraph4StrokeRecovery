from telnetlib import SE
import pandas as pd
import numpy as np
from . import utils
from .register_functions import class_register, register
import functools


class LimosSequenceBuilder():
    """used to normalize the LIMOS-sequences

        - convert the time-series to a time-series of deltas (i.e. differences between consecutive measurements)
        - remove mean (i.e. ensure zero-mean)
    
    """


    def __init__(self, measurement_table):
        """
            measurement_table: the limos dataframe
        """
        
        if np.any(pd.isnull(measurement_table)):
            raise ValueError("The table has missing values")
        
        
        self.ids = measurement_table.index.get_level_values('uID_eth').unique()
        self.maxlength = len(utils.get_config()['timepoints'])-1
        self.table = measurement_table
        self.columns = measurement_table.columns.values
        
        self.make_sequence_collection()
        
    @property
    def dim(self):
        return len(self.columns)
        
    def process_sequence(self, sequence):
        # turn the sequence into the sequence of deltas, i.e. differences between consectutive measurements
        
        length, dims = sequence.shape
        assert dims == len(self.columns)
        
        deltas = sequence[1:]-sequence[:-1] # increase from t to t+1
        return deltas
        
        
    def make_sequence_collection(self):
        """
            make collection of sequence such the the global expected value is 0
        """
        
        global_sum = np.zeros(self.dim)
        global_count = 0
        
        delta_sequences = list()
        for iid in self.ids: # go though patients 
            this = self.table.loc[iid]
            assert this.index.is_monotonic_increasing
            
            deltas = self.process_sequence(this.values)
            delta_sequences.append(deltas)
            
            global_sum = global_sum + deltas.sum(axis=0) 
            global_count += len(deltas)
            
        global_mean = global_sum/global_count
        assert global_mean.shape == (7,) # one value for each of 7 limos categories
        
        self.normalized_delta_sequences = list()
        
        for seq in delta_sequences:
            normalized = seq - global_mean
            
            self.normalized_delta_sequences.append(normalized)