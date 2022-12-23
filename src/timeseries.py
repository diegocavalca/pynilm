import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/multi-nilm')
import utils.chaotic_toolkit as ct

import numpy as np
from matplotlib import pyplot as plt

class RecurrencePlot:
    
    def __init__(self, data):
        self.data = data
        
    def calculate_mutual_information(self, delay_range=20, debug=True):
        mutual_information_per_delay = []
        for i in range(1, delay_range):
        #     print(f"iteration {i}")
            mutual_information_per_delay = np.append(
                mutual_information_per_delay,
                [ct.compute_mutual_information(self.data, i, 50 if len(self.data) >= 50 else len(self.data))]
                )

        local_min = [i 
                    for i in range(1, len(mutual_information_per_delay) -1) 
                                    if (mutual_information_per_delay[i] < mutual_information_per_delay[i-1]
                                    and mutual_information_per_delay[i] < mutual_information_per_delay[i+1])]    
        if local_min:
            local_min_idx = [i + 1 for i in local_min]
        else:
            local_min = [0]
            local_min_idx = [1]
        
        if debug: 
            print('local_min:', local_min)    
            plt.plot(range(1, delay_range), mutual_information_per_delay)
            plt.plot(local_min_idx, mutual_information_per_delay[local_min], 'x')
            plt.xlabel('delay')
            plt.ylabel('mutual information')
            plt.show()
            delay = local_min_idx[0]
            
            print(f"Delay = {delay}")
        
        return delay, mutual_information_per_delay[delay]
    
    def calculate_embedding_dimension(self, max_dimensions=15, debug=True):
        
        trials = list(range(1, max_dimensions))
        false_neighbors = []
        for i in trials:
            false_neighbors.append(ct.calculate_false_nearest_neighours(self.data, 1, i) / len(self.data))

        if debug:
            plt.plot(trials, false_neighbors)
            plt.xlabel('embedding dimension')
            plt.ylabel('Fraction of false neighbors')
                    
            embedding_dimension = np.argmax(false_neighbors) + 1 # cause trials starts at 1..N
            false_neighbors_max = false_neighbors[embedding_dimension - 1]
            plt.plot(embedding_dimension, false_neighbors_max, 'x')
            plt.show()

            print('false_neighbors:', false_neighbors)
            print('\nembedding_dimension =', embedding_dimension, f'(false_neighbors ~= {false_neighbors_max:.3f})')
        
        return embedding_dimension, false_neighbors_max