# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:36:51 2022

@author: Costelinha
"""
import numpy as np
import networkx as nx
import powerlaw

def Average_degree(G):
    degrees = [G.degree(n) for n in G.nodes()]
    mean=np.mean(degrees)
    return mean

def nodes_connected(G,u, v):
     return u in G.neighbors(v)
 

def average_degree_square(G):
    degrees = [G.degree(n) for n in G.nodes()]
    mean=np.mean(np.square(degrees))
    return mean


def Average_gamma_barabasi(N,m,number_of_simulations,starting_point):
    gammas_list=[]
    error_list=[]
    for i in range(number_of_simulations):
        if i>0:
            G.clear()
        G=nx.barabasi_albert_graph(N,m)
        degrees = [G.degree(n) for n in G.nodes()]
        l=degrees
        fit = powerlaw.Fit(np.array(l),xmin=starting_point,discrete=True)
        gammas_list.append(fit.power_law.alpha)
        error_list.append(fit.power_law.sigma)
    gamma_mean=np.mean(gammas_list)
    
    return gamma_mean

def calculate_gamma(G,starting_point):
        degrees = [G.degree(n) for n in G.nodes()]
        l=degrees
        fit = powerlaw.Fit(np.array(l),xmin=starting_point,discrete=True)
        return  fit.power_law.alpha
    
def Average_degree_weighted(G):
    degrees = [G.degree(n,weight='weight') for n in G.nodes()]
    mean=np.mean(degrees)
    return mean


def c_0(G):
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    ksqrmean=average_degree_square(G)
    N=G.number_of_nodes()
    norm=(ksqrmean-kmean)**2/(float(N)*kmean**3)
    return norm

def k_0(G):
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    ksqrmean=average_degree_square(G)
    k0=(kmean/ksqrmean)**(-1)
    return k0