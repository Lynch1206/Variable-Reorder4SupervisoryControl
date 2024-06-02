import os, sys
from pathlib import Path
from scipy.sparse import csgraph
import numpy as np
import array_to_latex as a2l
import hypernetx as hnx
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from prettytable import MARKDOWN
# from tkinter import filedialog
# from tkinter.filedialog import askdirectory
import pandas as pd
from Utility.Hypergraph_matrix import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from components_analysis import *
# from Hyper_plot import *
from Utility.Hygraph_json import Hyper_Read
from Utility.Hyper_read import Read_graph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_clusters(data, max_clusters):
    distortions = []  # 存儲每個群組數量對應的畸變程度
    silhouette_scores = []  # 存儲每個群組數量對應的輪廓分數

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # Inertia is the sum of squared distances to the nearest centroid
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    return distortions, silhouette_scores, kmeans

def Plot_KNN_evaluation(data, max_clusters):
    distortions, silhouette_scores, k_means = evaluate_clusters(data, max_clusters)
    # 視覺化結果
    plt.figure(figsize=(12, 3))

    # 畫肘部法則圖
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')

    # 畫輪廓分數圖
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.tight_layout()
    plt.show()
    return k_means

def Evaluate_KNN(data, max_clusters):
    SilScores, Elbows, _ = evaluate_clusters(data, max_clusters)
    Max_SilHouette = SilScores.index(max(SilScores))
    # check SilScores.index(sorted(SilScores)[-2]) exists
    if len(SilScores) > 1:
        Second_max_SilHouette = SilScores.index(sorted(SilScores)[-2])
    else:
        Second_max_SilHouette = Max_SilHouette
    if len(Elbows) > 1:
        Best_elbow = np.argmin(np.gradient(Elbows, 2))
    else:
        # return single cluster
        Best_elbow = 1
    # Best_elbow = np.argmin(np.gradient(Elbows, 2))
    # compare the 3 results and return the largest one
    result = max(Max_SilHouette, Second_max_SilHouette, Best_elbow)
    return result

def level_structure(graph, root):
    levels = {0: {root}}
    visited = {node: False for node in graph.nodes}
    visited[root] = True
    queue = [(root, 0)]

    while queue:
        current_node, current_level = queue.pop(0)
        next_level = current_level + 1
        if next_level not in levels:
            levels[next_level] = set()

        for neighbor in graph.neighbors(current_node):
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, next_level))
                levels[next_level].add(neighbor)

    return levels

def find_pseudo_peripheral_node(graph, G, subcluster):
    """
    graph: graph in which to find the pseudo-peripheral node
    DegreeArray: degree of each node in the whole graph
    G: whole graph
    subcluster: subgraph of the graph
    """
    # Step 1: Initialization
    # restrict the graph to subcluster
    # print(f"subcluster: {subcluster['cluster']}")

    if len(subcluster['label'].tolist()) == 1:
        return subcluster['label'].tolist()[0]
    if len(subcluster['label'].tolist()) == 2:
        subgraph = G.restrict_to_nodes(subcluster['label'].tolist())
        r = min(subgraph.nodes, key=subgraph.degree)
        return r
    else:
        # restrict the graph with nodes that contains in the cluster_nodes, e.g., interect of graph and cluster_nodes
        # ex: cluster_nodes = ['a', 'b', 'c']; list(G.nodes) = ['b', 'c', 'd', 'e'] then G.restrict_to_nodes(['b','c'])
        Intersect_set = list(set(list(G.nodes)) & set(subcluster['label'].tolist()))
        if len(Intersect_set) > 1:
            subgraph = G.restrict_to_nodes(Intersect_set)
            r = min(subgraph.nodes, key=subgraph.degree)
        else:
            subgraph = G
            r = min(subgraph.nodes, key=subgraph.degree)

        while True:
            # Step 2: Generation of level structure
            if r not in graph.nodes.labels:
                level_structure_inuse = level_structure(subgraph, root=r)
            else:
                level_structure_inuse = level_structure(graph, root=r)

            # Convert level structure to a list of levels
            level_structure_item = [set() for _ in range(max(level_structure_inuse.keys()) + 1)]
            for level, nodes in level_structure_inuse.items():
                level_structure_item[level] = nodes

            # Step 3: Sort the last level
            last_level = level_structure_item[-1]
            last_level_sorted = sorted(last_level, key=graph.degree)

            # Step 4: Test for termination
            for x in last_level_sorted:
                subgraph = graph.subgraph(level_structure_inuse[x])
                subgraph_level_structure_dict = level_structure_item(subgraph, root=x)
                subgraph_last_level = subgraph_level_structure_dict[max(subgraph_level_structure_dict.keys())]
                subgraph_last_level_sorted = sorted(subgraph_last_level, key=graph.degree)

                if subgraph_last_level_sorted > last_level_sorted:
                    r = x
                    break
            else:
                # Step 5: Exit
                return r

# make a function to return the dataframe of clusters
def ClusterToFile(nodes, edges, Adj, D_V, clusters_label, savefile= False, savepath = None, name = None):
    nodes_list = nodes['label'].tolist()
    df = pd.DataFrame({'label': nodes_list, f"{name}": clusters_label})
    # sort by cluster_label
    df_result = df.sort_values(by=[f"{name}"])
    # add new column named 'id' based on index
    df_result['id'] = df_result.index
    # Adj = Hyper_matrix(savefile, nodes, edges, w= None).AdjacencyMatrix_A()
    # D_V = Hyper_matrix(savefile, nodes, edges,w=None).Deg_Vertex(diag=False)
    df_result = SortTACreate_weight(df_result, Adj, D_V)
    if savefile is not True:
        return df_result
    else:
        df_result.to_csv(Path(savepath/f"{name}Clustering_result.csv"), index=True, header=True)
        return df_result
    
def Plot_2_clustering(node, clusters, s=20):
    # unzip the clusters to Affinity and spectral
    Affinity, Spectral = clusters 
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 3)
    ax[0].scatter(node['label'].tolist(),Affinity.labels_,c=Affinity.labels_, s=s, cmap='viridis')
    ax[0].set_title('AffinityPropagation')
    # legende on the right
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].scatter(node['label'].tolist(),Spectral.labels_,c=Spectral.labels_, s=s, cmap='viridis')
    ax[1].set_title('Spectral Clustering')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print(f"Affinity:{Affinity.labels_.max()+1} | Spectral: {Spectral.labels_.max()}")

def Plot_1_clustering(node, cluster, s=20, fig_size=(15, 3)):
    fig, ax = plt.subplots(1, 1)
    # set size
    fig.set_size_inches(fig_size)
    ax.scatter(node['label'].tolist(),cluster,c=cluster, s=s, cmap='viridis')
    ax.set_title('Clustering')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # compute the maximum of the cluster list
    print(f"Clustering result: {max(cluster)+1}")

def Subgraph_PPnode(cluster, file_directory, E, Graph):
    """\
        Cluster -> pd.Dataframe() : subgraph of the graph from the clustering result.
        Graph -> Hypergraph hnx: the whole graph
    """
    # Graph = Hyper_Read(file_directory).df_2_Hypergraph()
    # edges = Read_graph(file_directory).edges()

    Cluster_eges = []
    # E.387 go error because nothing in E.387?
    for i in E.keys():
        for j in cluster['id'].tolist():
            if j in E[i]['nodes']:
                Cluster_eges.append(i)
    return find_pseudo_peripheral_node(Graph.restrict_to_edges(Cluster_eges), G=Graph, subcluster = cluster)


def Target_clusters_table(data, no=None):
    """
    data: pd.DataFrame() of the cluster
    no: int, the number of the cluster
    """
    if no is not None:
        x = PrettyTable()
        x.field_names = [f"No.{no} Target cluster"]
        x.add_row([data])
        print(x)
    else:
        x = PrettyTable()
        x.field_names = [f"Target cluster"]
        x.add_row([data])
        print(x)

def SortTACreate_weight(df, Adj, Deg):
    """
    Sizes of df (number of clusters) should match the number of ppnodes
    """

    for i in df['id']:
        df.loc[i, 'Adj'] = Adj[i,0]
        df.loc[i, 'Degree'] = Deg[i]
    return df
