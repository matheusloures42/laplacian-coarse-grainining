import numpy as np
import networkx as nx
from scipy.linalg import null_space
from numpy.linalg import eigh,eigvalsh,inv, norm, matrix_rank, pinv

"""
Define the Clique complex from the graph G
"""

class CliqueComplex:
    def __init__(self,G):
        self.G=G
    """
    Calculate the boundary matrix G of order n
    """
    
    def high_order_incidence_matrix(self,n): 
        
        c=list(nx.enumerate_all_cliques(self.G)) #all cliques in graph G (including faces of higher siplex)
       
        n_simplex=[]
        n_minus_1_simplex=[]
        n_l=0
        n_c=0
        for j in range(len(c)): #print k-cliques or (k-1)-simplex
            if len(c[j])==n+1:   
                n_simplex.append(c[j])
                n_c+=1
            if len(c[j])==n:
                n_minus_1_simplex.append(c[j])
                n_l+=1
        Bn=np.zeros([n_l,n_c])
        for j in range(len(n_simplex)):
            for i in range( len(n_minus_1_simplex)):
                ind=0
                if set(n_minus_1_simplex[i]).issubset(set( n_simplex[j]))==True:
                        for k in range(len(n_minus_1_simplex[i])):
                            if n_minus_1_simplex[i][k]!= n_simplex[j][k]:
                                Bn[i,j]=(-1)**k
                                ind=1
                                break
                        if ind==0:
                            Bn[i,j]=(-1)**(n)
        Bn=Bn.astype('float32') 
        return Bn
    """
   betti number  of order n, 
   ismaxdim= 0 if n is not the maximal dimension of the simplicial complex
   ismaxdim!=0 if n is not the maximal dimension of the simplicial complex
    """
    def betti_number(self,n,ismaxdim):
        clique=CliqueComplex(self.G)
        Bn=clique.high_order_incidence_matrix(n)
        Bn_plus_1=clique.high_order_incidence_matrix(n+1)
        kerBn= null_space(Bn)
        if ismaxdim==0:
            Bn_plus_1=clique.high_order_incidence_matrix(n+1)
            dim_kerBn= len(kerBn[0])
     
            rankBn_plus_1= matrix_rank(Bn_plus_1)
            bn=dim_kerBn- rankBn_plus_1
        else:
            dim_kerBn= len(kerBn[0])
     
           
            bn=dim_kerBn
        return(bn)
    """
   number of simplices of order n
    """
    def number_of_simplices(self,n):
       
        c=list(nx.enumerate_all_cliques(self.G))
        #print(c)
       
        count=0
        ind=0
        for j in range(len(c)): 
            if len(c[j])==n+1:   
                #print(c[j])
                ind=1
                count+=1
          
        if ind==0:  
            print('there is no maximal simplex with dimension=',n )
           
        return count
