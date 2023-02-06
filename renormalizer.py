
import numpy as np
import networkx as nx
from numpy.linalg import pinv
from utils import *


"""
coarse graining step use G as unput and make one step of renormalization using the pseudo inverse of laplacian.

"""

def coarse_graining_step_old(G):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
    most_correlated_nodes=[]
    correlated_nodes=[]
    correlations=[]
    for i in range(len(C)):
        for j in range(len(C)):
            if i>j:
                correlated_nodes.append([i,j])
                correlations.append(C[i,j])
                
    correlations=np.array(correlations)          
    x=correlations.argsort()
    correlated_nodes=np.array(correlated_nodes)
    correlated_nodes_sorted= correlated_nodes[x[::-1]]
    
    list_of_used_nodes=[]
    for i in range(len(x)):
        for j in range(2):
            if i==0:
                most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
               
            if int(correlated_nodes_sorted[i][j]) not in list_of_used_nodes:
                if j==0:
                    if int(correlated_nodes_sorted[i][j+1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j])
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j+1])
                        
                if j==1:
                    if int(correlated_nodes_sorted[i][j-1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j-1]))        
    most_correlated_nodes.remove(most_correlated_nodes[0])    
    #print('\n')
    #print(most_correlated_nodes)
    #print(list_of_used_nodes.sort())
    #print('\n')
    
    #print(len(most_correlated_nodes))

    
    for k in range(len(most_correlated_nodes)):
       u=int(most_correlated_nodes[k][0])
       v=int(most_correlated_nodes[k][1])
       G = nx.contracted_nodes(G,u,v)
    #print(G.number_of_nodes())
    G=nx.convert_node_labels_to_integers(G)
    return G         


def coarse_graining_step(G):
    
    # Getting the Laplacian of G
    L_aux = nx.laplacian_matrix(G).todense()
    L = np.array(L_aux)

    # Getting the pseudo-inverse of the Laplacian
    C = np.linalg.pinv(L)

    most_correlated_nodes = []
    correlated_nodes = []
    correlations = []

    for i in range(len(C)):
        for j in range(i):
            correlated_nodes.append([i,j])
            correlations.append(C[i,j])
    
    # Sorting the nodes by their correlations (descending)
    x = np.array(correlations).argsort()
    correlated_nodes_sorted = np.array(correlated_nodes)[x[::-1]]
    
    used_nodes_lst = []
    for i in range(len(x)): # Most correlated nodes

        corr_pair = [int(j) for j in correlated_nodes_sorted[i]]
        no_duplicates = corr_pair not in most_correlated_nodes
        both_nodes_unused = all(x not in used_nodes_lst for x in corr_pair)

        if both_nodes_unused and no_duplicates:
            most_correlated_nodes.append(corr_pair)
            used_nodes_lst.extend(corr_pair)

    # Creating the super-nodes
    for pair in most_correlated_nodes:
       [u, v] = pair
       G = nx.contracted_nodes(G, u, v)

    return nx.convert_node_labels_to_integers(G)
"""
Laplacian renorm uses G and the number of steps of renormalization that you want as input.
"""
def laplacian_renorm(G,number_of_steps):
    for j in range(number_of_steps):
           G=coarse_graining_step(G)
    return G





