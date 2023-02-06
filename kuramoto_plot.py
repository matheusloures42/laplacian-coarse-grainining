# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:48:06 2022

@author: Costelinha

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kuramoto import Kuramoto, plot_phase_coherence, plot_activity

class KuramotoPlot:
    
    
    def __init__(self,G):
        self.G=G
        self.n=G.number_of_nodes()
        
    """
    nat_freqs= Natural frquencys distribution.
    cor=color of the plot.
    number_of_simulation=number of simulations to be averaged.

    """
    def plot_order_param_vs_coupling_avg(self,cor,nat_freqs,number_of_simulations):
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.6)
    
        # Instantiate a random graph and transform into an adjacency matrix
        
        graph = nx.to_numpy_array(self.G)
        n_nodes=len(graph)
    
        coupling_vals = np.linspace(0, 1.0, 100)
        mean_of_simulations=np.zeros(len(coupling_vals))
        for j in range(number_of_simulations):
            r=np.zeros(len(coupling_vals))
            
            runs = []
            
            for coupling in coupling_vals:
                model = Kuramoto(coupling=coupling, dt=0.1, T=500, n_nodes=len(graph))
                model.natfreqs =  nat_freqs  # reset natural frequencies
                act_mat = model.run(adj_mat=graph)
                runs.append(act_mat)
    
            # Check that natural frequencies are correct (we need them for prediction of Kc)
    
    
            runs_array = np.array(runs)
    
    
            for i, coupling in enumerate(coupling_vals):
                r_mean = np.mean([model.phase_coherence(vec)
                              for vec in runs_array[i, :, -1000:].T]) # mean over last 1000 steps
                r[i]= r_mean
            mean_of_simulations+=r/number_of_simulations
    
        # Predicted Kc – analytical result (from paper)
        #Kc = np.sqrt(8 / np.pi) * np.std(model.natfreqs) # analytical result (from paper)
        #plt.vlines(Kc, 0, 1, linestyles='--', color='orange', label='analytical prediction')
        np.savetxt(str(self.G),r,fmt='%.2f')
        plt.scatter(coupling_vals, r, c=cor, s=20, alpha=0.7)
        #plt.legend()
        #plt.grid(linestyle='--', alpha=0.8)
        plt.ylabel('order parameter (r)')
        plt.xlabel('coupling (K)')
        sns.despine()
    
    
    def plot_opxt(self):
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.6)
    
        # Instantiate a random graph and transform into an adjacency matrix
        
        graph = nx.to_numpy_array(self.G)
    
        # Instantiate model with parameters
        model = Kuramoto(coupling=3, dt=0.01, T=10, n_nodes=len(graph))
    
        # Run simulation - output is time series for all nodes (node vs time)
        act_mat = model.run(adj_mat=graph)
        plot_phase_coherence(act_mat)