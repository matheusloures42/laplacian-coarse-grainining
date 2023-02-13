# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:43:45 2022

@author: Costelinha
"""
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import powerlaw
import scienceplots

from scipy.stats import norm,lognorm,cauchy
from numpy.linalg import eigh,eigvalsh,inv, norm, matrix_rank, pinv
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm
from utils import *
from scipy.linalg import expm

"""
All functions take as input the graph G.
For the power-law distribution the input x is the starting point of the fitting.
"""


    

def plot_ccdf(G,lb,mark):
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    l=degrees/kmean
    heights,bins=np.histogram(l,bins=200)
    pk=heights/np.sum(heights)
    x=np.linspace(0, np.amax(l),num=len(heights))
    ccdf=1-np.cumsum(pk)
    plt.style.use(['science','notebook'])
    plt.scatter(x,ccdf,label=lb, marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-4),top=1)
    plt.legend()
    plt.xlim(left=10**(-2),right=10**2)
    plt.ylabel('$P_c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    

def clustering_per_kl(G,lb, mark):
    
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    #ksqrmean=average_degree_square(G)
    #N=G.number_of_nodes()
    #norm=(ksqrmean-kmean)**2/(float(N)*kmean**3)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
   

    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    cpd=[]
    for degree in d:
        K+=1
        clustering_coeff = nx.clustering(G, d[degree])
        
        cpd.append(sum(clustering_coeff.values())/len(clustering_coeff))
    plt.scatter(l,cpd,marker=mark,label=lb,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    plt.xlim(left=10**(-1),right=10**1)
    #plt.ylim(bottom=10**(-3),top=1)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)

def clustering_per_kl_c0(G,lb, mark):
    
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    ksqrmean=average_degree_square(G)
    N=G.number_of_nodes()
    norm=(ksqrmean-kmean)**2/(float(N)*kmean**3)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
   

    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    cpd=[]
    for degree in d:
        K+=1
        clustering_coeff = nx.clustering(G, d[degree])
        
        cpd.append(sum(clustering_coeff.values())/len(clustering_coeff)*(1/norm))
    plt.style.use(['science','notebook'])
    plt.scatter(l,cpd,label=lb,marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)/c_0$')
    plt.xlabel('$k/<k>$')
    plt.ylim(bottom=10**(-1),top=10)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)





def average_neighbor_degree_x_kl(G,lb,mark):
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    ksqrmean=average_degree_square(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
    
    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    annd=[]
    for degree in d:
        K+=1
        nearest_neighbors_degree = nx.average_neighbor_degree(G, nodes=d[degree])
        
        
        
        annd.append(sum(nearest_neighbors_degree.values())/len(nearest_neighbors_degree)*(kmean/ksqrmean))
    
    plt.scatter(l,annd,label=lb,marker=mark,s=100)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-1),top=10)
    plt.ylabel('$k_{nn,n}(k/<k>)$')
    plt.xlabel('$k/<k>$')
    
def heat_map_covariance(G):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
    
    for j in range(len(L)):
        C[j,j]=0
    plt.imshow(C,norm=LogNorm(),cmap='Greens')
    #plt.imshow(C, interpolation='none')
    plt.colorbar()
    
    
def node_communcability_adjacency(G):
    A=nx.adjacency_matrix(G)
    A=A.todense()
    A=np.array(A)
    Gpq=expm(A)
    plt.imshow(Gpq,norm=LogNorm())
    plt.colorbar()
    plt.show() 
    
def node_communcability_laplacian(G):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    Gpq=expm(L)
    plt.imshow(Gpq,norm=LogNorm())
    plt.colorbar()
    plt.show() 
    
def average_GD(G,lb):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    C=np.linalg.pinv(L)
    GDavg=[]
    N=len(C)
    node_list=list(np.arange(N))
    
    for j in range(N):
        avg=0
        for i in range(N):
            if i!=j:
                avg+= C[i,j]
        GDavg.append(avg/(N-1))
        
    plt.scatter(node_list,GDavg,label=lb)
    plt.yscale('symlog')
    plt.xscale('log')
    #plt.ylim(top=0.005,bottom=-0.005)
    plt.plot(GDavg)
    plt.legend()
    plt.ylabel('$<G^D>$')
    plt.xlabel('nodes')
    
def average_GD_x_kl(G,lb):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    C=np.linalg.pinv(L)
    
    GDavg=[]
    N=len(C)
    node_list=list(np.arange(N))
    
    for j in range(N):
        avg=0
        for i in range(N):
            if i!=j:
                avg+= C[i,j]
        GDavg.append(avg/(N-1))
        
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
   
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean     
    
    d = defaultdict(list)
  
    for u in G.nodes():
       d[G.degree(u)].append(u)
    cpd=[]
    
    for degree in d:
        sum=0
        for index in d[degree]:
            sum+=GDavg[int(index)]
        cpd.append(sum/len(d[degree]))
    plt.scatter(l,cpd,label=lb)
    #plt.yscale('symlog')
    plt.xscale('log')
    plt.ylim(top=0.005,bottom=-0.005)
    #plt.plot(l,cpd)
    plt.legend()
    plt.ylabel('$<G^D>$')
    plt.xlabel('$k/<k>$')

def heat_map_adjacency(G):
    A=nx.adjacency_matrix(G)
    A=A.todense()
    A=np.array(A)
    A=A.astype(int)
    plt.imshow(A,norm=LogNorm())
    plt.colorbar()
    
    #plt.colorbar()
    
def BA_avg_ccdf(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    ccdf_avg=0
    kmeans=[]
    maxk=[]
    for j in range(100):
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        kmean=Average_degree(list_of_graphs[j])
        l=degrees/kmean
        kmeans.append(kmean)
        maxk.append(np.amax(degrees))
        heights,bins=np.histogram(l,bins=200)
        pk=heights/np.sum(heights)
        
        ccdf=1-np.cumsum(pk)
        ccdf_avg+=ccdf/100
    x=np.linspace(0, np.mean(maxk)/np.mean(kmeans),num=len(heights))
    plt.style.use(['science','notebook'])
    plt.scatter(x,ccdf_avg,label=lb,marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-4),top=1)
    plt.legend()
    plt.xlim(left=10**(-2),right=10**2)
    plt.ylabel('$P_c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    
def ER_avg_ccdf(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/erdos_data/p01_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    ccdf_avg=0
    kmeans=[]
    ccdf_avg=0
    kmeans=[]
    maxk=[]
    for j in range(100):
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        kmean=Average_degree(list_of_graphs[j])
        l=degrees/kmean
        kmeans.append(kmean)
        maxk.append(np.amax(degrees))
        heights,bins=np.histogram(l,bins=200)
        pk=heights/np.sum(heights)
        
        ccdf=1-np.cumsum(pk)
        ccdf_avg+=ccdf/100
    x=np.linspace(0, np.mean(maxk)/np.mean(kmeans),num=len(heights))
    plt.style.use(['science','notebook'])
    plt.scatter(x,ccdf_avg,label=lb, marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-4),top=1)
    plt.legend()
    plt.xlim(left=10**(-2),right=10**2)
    plt.ylabel('$P_c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    

def BA_avg_clustering_per_kl(l,lb, mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    c = defaultdict(list)
    kmeans=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            clustering_coeff= nx.clustering(list_of_graphs[j], d[degree])
            
            c[degree].append(sum(clustering_coeff.values())/len(clustering_coeff))
            
        clustering_coeff.clear()
    cpd=[]
    s=[]
    for degree in c:
        s.append(degree)
       
        cpd.append(np.mean(c[degree]))
        
    l=s/np.mean(kmeans)
        
    plt.scatter(l,cpd,label=lb,marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    plt.xlim(left=10**(-1),right=10**1)
    #plt.ylim(bottom=10**(-3),top=1)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)
    

def ER_avg_clustering_per_kl(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/erdos_data/p01_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    c = defaultdict(list)
    kmeans=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            clustering_coeff= nx.clustering(list_of_graphs[j], d[degree])
            
            c[degree].append(sum(clustering_coeff.values())/len(clustering_coeff))
            
        clustering_coeff.clear()
    cpd=[]
    s=[]
    for degree in c:
        s.append(degree)
       
        cpd.append(np.mean(c[degree]))
        
    l=s/np.mean(kmeans)
        
    plt.scatter(l,cpd,label=lb,marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    plt.xlim(left=10**(-1),right=10**1)
    #plt.ylim(bottom=10**(-3),top=1)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)
    
def BA_avg_clustering_per_kl_c0(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    c = defaultdict(list)
    kmeans=[]
    c_0_list=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        c_0_list.append(c_0(list_of_graphs[j]))
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            clustering_coeff= nx.clustering(list_of_graphs[j], d[degree])
            
            c[degree].append(sum(clustering_coeff.values())/len(clustering_coeff))
            
        clustering_coeff.clear()
    cpd=[]
    s=[]
    for degree in c:
        s.append(degree)
       
        cpd.append(np.mean(c[degree]))
        
    l=s/np.mean(kmeans)
    
    c0mean=np.mean(c_0_list)
        
    plt.scatter(l,cpd/c0mean,label=lb,marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)$')
    plt.xlabel('$k/<k>$')
    plt.xlim(left=10**(-1),right=10**1)
    #plt.ylim(bottom=10**(-3),top=1)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)
    

def ER_avg_clustering_per_kl_c0(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/erdos_data/p01_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    c = defaultdict(list)
    kmeans=[]
    c_0_list=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        c_0_list.append(c_0(list_of_graphs[j]))
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            clustering_coeff= nx.clustering(list_of_graphs[j], d[degree])
            
            c[degree].append(sum(clustering_coeff.values())/len(clustering_coeff))
            
        clustering_coeff.clear()
    cpd=[]
    s=[]
    for degree in c:
        s.append(degree)
       
        cpd.append(np.mean(c[degree]))
        
    l=s/np.mean(kmeans)
    
    c0mean=np.mean(c_0_list)
        
    plt.scatter(l,cpd/c0mean,label=lb, marker=mark,s=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$c(k/<k>)/c_0$')
    plt.xlabel('$k/<k>$')
    plt.xlim(left=10**(-1),right=10**1)
    #plt.ylim(bottom=10**(-3),top=1)
    #plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)
    


def BA_avg_average_neighbor_degree_x_kl(l,lb,mark,s=100):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    knn = defaultdict(list)
    kmeans=[]
    k_0_list=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        k_0_list.append(k_0(list_of_graphs[j]))
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            nearest_neighbors_degree = nx.average_neighbor_degree(list_of_graphs[j], nodes=d[degree])
           
            knn[degree].append(sum(nearest_neighbors_degree.values())/len(nearest_neighbors_degree))
            
        nearest_neighbors_degree.clear()
    annd=[]
    s=[]
    for degree in knn:
        s.append(degree)
       
        annd.append(np.mean(knn[degree]))
        
    l=s/np.mean(kmeans)
    
    k0mean=np.mean(k_0_list)
        
    plt.scatter(l,annd/k0mean,label=lb,marker=mark,s=100)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-1),top=10)
    plt.ylabel('$k_{nn,n}(k/<k>)$')
    plt.xlabel('$k/<k>$')
    
    
def ER_avg_average_neighbor_degree_x_kl(l,lb,mark):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/erdos_data/p01_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    knn = defaultdict(list)
    kmeans=[]
    k_0_list=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        k_0_list.append(k_0(list_of_graphs[j]))
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            nearest_neighbors_degree = nx.average_neighbor_degree(list_of_graphs[j], nodes=d[degree])
            
            knn[degree].append(sum(nearest_neighbors_degree.values())/len(nearest_neighbors_degree))
            
        nearest_neighbors_degree.clear()
    annd=[]
    s=[]
    for degree in knn:
        s.append(degree)
       
        annd.append(np.mean(knn[degree]))
        
    l=s/np.mean(kmeans)
    
    k0mean=np.mean(k_0_list)
        
    plt.scatter(l,annd/k0mean,label=lb,marker=mark,s=100)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-1),top=10)
    plt.ylabel('$k_{nn,n}(k/<k>)$')
    plt.xlabel('$k/<k>$')
    
def BA_avg_gamma(l,starting_point):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    ccdf_avg=0
    kmeans=[]
    l=[]
    maxk=[]
    for j in range(100):
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        kmean=Average_degree(list_of_graphs[j])
        l.extend(degrees)
    plt.hist(l, bins=100, density=True, alpha=0.6, color='g')
    fit = powerlaw.Fit(np.array(l),xmin=starting_point,discrete=True)
    plt.ylabel('$P(k)$')
    plt.xlabel('$k$')
    plt.xscale('log')
    plt.yscale('log')
    fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit pdf')
    fit.plot_pdf( color= 'b')
    
    plt.ylabel('degree distribution')
    plt.xlabel('degrees')
    plt.show()
    print('gama= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)
    return  fit.power_law.alpha

def BA_avg_average_neighbor_degree_x_kl_without_k0(l,lb):
    list_of_graphs={}
    for j in range(100):
        list_of_graphs[j]=nx.read_edgelist('Data/barabasi_data/m5_'+str(l)+'_edgelist_loop_'+str(j)+'.txt')
        
    
    knn = defaultdict(list)
    kmeans=[]
    k_0_list=[]
   
    for j in range(100):
        d = defaultdict(list)
        degrees=[]
        degrees = [list_of_graphs[j].degree(n) for n in list_of_graphs[j].nodes()]
        
        kmean=Average_degree(list_of_graphs[j])
        kmeans.append(kmean)
        k_0_list.append(k_0(list_of_graphs[j]))
        for u in list_of_graphs[j].nodes():
               d[list_of_graphs[j].degree(u)].append(u)
                
        for degree in d:
            nearest_neighbors_degree = nx.average_neighbor_degree(list_of_graphs[j], nodes=d[degree])
           
            knn[degree].append(sum(nearest_neighbors_degree.values())/len(nearest_neighbors_degree))
            
        nearest_neighbors_degree.clear()
    annd=[]
    s=[]
    for degree in knn:
        s.append(degree)
       
        annd.append(np.mean(knn[degree]))
        
    l=s/np.mean(kmeans)
    
    k0mean=np.mean(k_0_list)
        
    plt.scatter(l,annd,label=lb)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.ylabel('$k_{nn,n}(k/<k>)$')
    plt.xlabel('$k/<k>$')
    