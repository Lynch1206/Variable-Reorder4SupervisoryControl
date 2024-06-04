"""
This file aims to execute the sorting for a clustering result
from Sander's phd thesis, the A' (an individual clustering "group") contains a list of nodes.
First, take a node at pseudo-peripheral node of A'.
> A pseudo-peripheral nodes is a node that has the largest eccentricity in the graph. (the one with largest distance to another in graph)
It takes 2 sorting actions for each iteration: 
1. sort the nodes in each A' by their weighted values (in paper, its Adjacency matrix)
    If nodes in the list are equal weights, 2. sort those nodes by their vertex degree.
"""
import hypernetx as hnx
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class order4DS(object):
    def __init__(self, data, r, name_C) -> None:
        self.cluster = data     # the dataframe of the clustering result - with cluster_label
        self.r = r              # All pseudo-peripheral nodes in subgraphs
        self.name_C = name_C    # the label name in the dataframe
    def C(self, i):
        """
        C: the set of clusters, including the PPnode
        """
        if i > len(self.r):
            raise ValueError("i should be less than the number of clusters")
        else:
            self.cluster[self.cluster[self.name_C]==i]
        return self.cluster[self.cluster[self.name_C]==i]
    def Remove_PPnode(self,i):
        """
        Remove the PPnode from a cluster
        i : the PPnode index
        """
        # index of the corresponding PPnode
        # Target_0[Affinity_Ap[Affinity_Ap[name_label]== 0]['label'].isin(PPnodes)].index
        C = self.C(i)
        # index_row = C[self.Dropped_index(i)].index
        index_row = C[self.cluster[self.cluster[self.name_C]==i]['label'].isin(self.r)].index
        return C.drop(index_row)
    def Dropped_index(self,i):
        """
        Return the index of the dropped nodes
        i : the PPnode index
        """
        C = self.C(i)
        return C[self.cluster[self.cluster[self.name_C]==i]['label'].isin(self.r)]
    def sort_W_D(self, data):
        """
        Sort the dataframe by weight, if equal in weight values, sort by degree
        Adj: the Adjacency matrix
        Degree: the degree of the nodes
        """
        # Descending weight
        data = data.sort_values(by=['Adj'], ascending=False)
        # Ascending degree
        return data.sort_values(by=['Degree'], ascending=True)

    def DS_sorting4clusters(self, input_data, R=None):
        """
        input_data: the dataframe of the clustering result - with cluster_label
        R : the index of the pseudo-peripheral node
        """
        # specificed the R for the pseudo-peripheral nodes
        # if not specified, use all the pseudo-peripheral nodes
        if R is None:
            i = len(self.r)
            if i < 1:
                raise ValueError("i should be larger than 0")
            if i == 1:
                return input_data
            # create a new dataframe to store the sorting result
            sorting_result = pd.DataFrame()
            # Step 1: len(self.r) --> numbers of the clusters
            for i in range(len(self.r)):
                # Step 2: Remove PPnode for the corresponding cluster
                Drop_TA = self.Dropped_index(i)
                Sort_TA = self.Remove_PPnode(i)
                # Step 3: Sort the dataframe by weight and degree
                Sort_TA = self.sort_W_D(Sort_TA)
                Result = pd.concat([Drop_TA, Sort_TA], ignore_index=True)
                # Step 4: Concate the dataframe and save in sorting_result:
                sorting_result = pd.concat([sorting_result, Result], ignore_index=True)
            return sorting_result     
        else:
            # R should be integer
            if isinstance(R, int) is not True:
                raise TypeError("R should be integer")
            if R < 0:
                raise ValueError("R should be larger than 0 or equal to 0")
            if R > len(self.r):
                raise ValueError("R should be less than the number of clusters")
            Drop_TA = self.Dropped_index(R)
            Sort_TA = self.Remove_PPnode(R)
            Sort_TA = self.sort_W_D(Sort_TA)
            Result = pd.concat([Drop_TA, Sort_TA], ignore_index=True)
            return Result
