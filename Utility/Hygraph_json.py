# import necessary packages
import pathlib
import sys, os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.sparse import csgraph
# import networkx as nx
from matplotlib import pyplot as plt
import itertools as itt
import pandas as pd
import json
import os
import hypernetx as hnx


# os.path.join(os.path.dirname(__file__))
class Hyper_Read(object):
    """
    Read json files
    Convert json file to dataframe
    Convert dataframe to hypergraph
    """
    def __init__(self, json_file, path=None):
        self.json_file = json_file
        self.path = path

    def read_js(self):
        with open(self.json_file) as f:
            data = json.load(f)
        return data
    # change js2df to more simple function
    # def js2df(self):
    #     df = self.read_js()
    #     df_nodes = pd.DataFrame(df['nodes'])
    #     df_edges = pd.DataFrame(df['edges'])
    #     df_edges = df_edges.set_index('id')
    #     edges_dict = df_edges.to_dict('index')
    #     return df_nodes, edges_dict
    def js2df(self):
        """
        New json file for node: nodes, operation, BDDnodes.
        """
        df = self.read_js()
        if "BDDnodes" not in df.keys():
            # old file
            if len(df.keys()) > 1:
                # old file
                df_nodes = pd.DataFrame(df['nodes'])
                df_edges = pd.DataFrame(df['edges'])
                df_edges = df_edges.set_index('id')
                edges_dict = df_edges.to_dict('index')
                # return ndoes, edges
                return df_nodes, edges_dict
            else:
                if "edges" in df.keys():
                    # need to filter empty hyperedges
                    df = [edge for edge in df["edges"] if edge["hyperedge"]]
                    df_edges = pd.DataFrame(df)
                    df = df_edges.drop(columns=['id'])
                    df = df.reset_index()
                    df = df.rename(columns={'index':'id'})
                    print(f"return hyperedges data")
                    return df
                if "nodes" in df.keys():
                    df = pd.DataFrame(df["nodes"])
                    return df
        else:
        #  new data, include BDD, operation
            return [pd.DataFrame(df["nodes"]), df['operation'][0]['operation_no.'][0], df['BDDnodes'][0]['BDD_nodes'][0]]


    def df2hnx(self, data_new = False, nodes = None, edges = None):
        """
        For new data, the file_dir for Hyper_Read still mandatory, but not effective.
        Manuallt input with nodes and edges
        """
        if data_new is False:
            nodes, edges = self.js2df()
            Graph = {}
            for key in edges:
                # map nodes 'id' directly to edges' [nodes]
                Graph[key] = [nodes['label'][i] for i in edges[key]['nodes']]
                return Graph
        else:
            Graph = {}
            for i in range(len(edges)):
                Graph[edges['id'][i]] = []
                for j in range(len(edges['hyperedge'][i])):
                    Graph[edges['id'][i]].append(nodes['label'][edges['hyperedge'][i][j]])
            return Graph

    
    def df_2_Hypergraph(self, data_new = False, nodes = None, edges = None):
        if data_new is False:
            Graph = self.df2hnx()
            return hnx.Hypergraph(Graph)
        else:
            Graph = self.df2hnx(data_new = True, nodes = nodes, edges = edges)
            return hnx.Hypergraph(Graph)
