
import sys, os
import numpy as np
from scipy.sparse import csgraph
import networkx as nx
import hypernetx as hnx
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
import pandas as pd
# from make_nodes2 import make_node_pair_matrix
# from Hyper_read import Read_graph
# from Hygraph_json import Hyper_Read
from scipy import linalg
from Utility.Hygraph_json import Hyper_Read


"""
From the paper of 
[1] D. Zhou, J. Huang, and B. Schölkopf, 
“Learning with hypergraphs: Clustering, classification, and embedding,” 
in Advances in neural information processing systems, B. Schölkopf, J. Platt, and T. Hoffman, Eds., MIT Press, 2006. [Online].
 Available: https://proceedings.neurips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf
"""

# given a hypergraph G = (V, E, w)
# where w is weight of edges, deafault is 1 (or unity)
class Hyper_matrix():
    def __init__(self, file_dir, Vertex, Edges, w, data_new = None) -> None:
        weighted = None
        self.file_dir = file_dir
        self.Vertex = Vertex
        self.Edges = Edges
        self.w = w if not None else weighted
        self.n = len(Vertex)
        self.data_new = data_new
        if self.data_new is None:
            self.m = len(Edges.keys())
        else:
            self.m = len(Edges)

    def Graph_df2hnx(self):
        """Convert graph to hypergraph"""
        Graph = Hyper_Read(self.file_dir).df2hnx()
        return hnx.Hypergraph(Graph)
    
    def IncidenceMatrix_H(self):
        """
        H is an incidence matrix that can represent a hypergraph
        H = |V| x |E| matrix.
        H[n,m] = 1 if vertex n is in edge m
        # Parameters:
        ------------------------------------------
        @Returns : np.array (n x m)
        ------------------------------------------
        """
        H = np.zeros((self.n, self.m), dtype=int)

        if self.data_new is None:
            for i in range(self.n): # n is number of vertices
                for edge_id in self.Edges.keys(): # enumerate all edges
                    if i in self.Edges[edge_id]['nodes']:
                        # (example) self.Edges[edge_id]['nodes'] = [0, 2, 3], this edge has vertice:[0, 2, 3] 
                        H[i, int(edge_id)] = 1
            return H
        else:
            for i in range(self.n):
                for j in self.Edges['id']:
                    if i in self.Edges['hyperedge'][j]:
                        # if nodes item in edges['hyperedge'][j] then incidence matrix element count 1
                        H[i, j] = 1
            return H


    def AdjacencyMatrix_A(self, norm=False):
        """
        # Un-normalized adjacency matrix of a hypergraph G = (V, E)
        # Normalized ajdacency matrix of a hypergraph: “A = HW HT − Dv” ([Zhou et al., 2006, p. 3]
        # Parameters:
        ------------------------------------------
        `norm`: BOOL, switcher for normalization case. Default is False.
            1. False: Un-normalized adjacency matrix:
            “it is a square matrix which rows and columns are indexed by the vertices of H and for all x, y ∈ V , x = y the entry ax,y =|{e ∈ E : x, y ∈ e}| and ax,x = 0.” 
            Unormalized source: [Bretto, 2013, p. 8]
            2. False: Normalized adjacency matrix:
            
        @Returns : np.array (n x n)
        ------------------------------------------
        """
        if self.data_new is None:
            if norm is False:
            ## Un-normalized adjacency matrix of a hypergraph G = (V, E)
            # “it is a square matrix which rows and columns are indexed by the vertices of H and for all x, y ∈ V , x = y the entry ax,y =|{e ∈ E : x, y ∈ e}| and ax,x = 0.” 
                A = np.zeros((self.n, self.n), dtype=int) # dimension of A is |V| x |V|
                for i in range(self.n):
                    for j in range(self.n):
                        for edge in self.Edges.keys():
                            if i in self.Edges[edge]['nodes'] and j in self.Edges[edge]['nodes']:
                                A[i, j] += 1
                            if i == j:
                                A[i, j] = 0
                return A
            else:
                return self.IncidenceMatrix_H()@self.IncidenceMatrix_H().T - self.Deg_Vertex(norm=True, diag=True)
        else:
            if norm is False:
                A = np.zeros((self.n, self.n), dtype=int) # dimension of A is |V| x |V|
                for i in range(self.n):
                    for j in range(self.n):
                        for edge in self.Edges['id']:
                            if i in self.Edges['hyperedge'][edge] and j in self.Edges['hyperedge'][edge]:
                                A[i, j] += 1
                            if i == j:
                                A[i, j] = 0
                return A
            else:
                return self.IncidenceMatrix_H()@self.IncidenceMatrix_H().T - self.Deg_Vertex(norm=True, diag=True)
    
    def Deg_Edges(self, norm=False, diag=False):
        """
        Degree of edges: delta(e) = |e|
        """
        if self.data_new is None:
            if norm is False:
                A = self.AdjacencyMatrix_A()
            if diag is False:
                return A.sum(axis=0)
            else:
                return np.diag(A.sum(axis=0))
        else:
            A = self.AdjacencyMatrix_A()
            if diag is False:
                return A.sum(axis=0)
            else:
                return np.diag(A.sum(axis=0))
        

    def Deg_Vertex(self, norm=False, diag=False):
        """
        Unnormalized: Degree of vertex: D(x) = Σ_{y ∈ V} a_{x,y} - by n nodes
        Normalized: “d(x) = ∑_{e∈E} w(e)h(x, e)” ([Zhou et al., 2006, p. 3]
        # Parameters:
        ------------------------------------------
        `norm`: BOOL, switcher for normalization case. Default is False.
        `diag`: BOOL, switcher for return diagnol matrix case. Default is False.
        @Returns : np.array (n) or (n x n)
        ------------------------------------------
        """

        if norm is False:
            matrix = self.AdjacencyMatrix_A(norm=False)
        else:
            matrix = self.IncidenceMatrix_H()

        if norm is False:
            # matrix = self.AdjacencyMatrix_A(norm=False)
            if diag is False:
                return matrix.sum(axis=1)
            else:
                return np.diag(matrix.sum(axis=1))
        else:
            # “d(v) = ∑ e∈E w(e)h(v, e)” ([Zhou et al., 2006, p. 3](zotero://select/library/items/EE7FSPK4)) ([pdf](zotero://open-pdf/library/items/ZX3TGVRS?page=3&annotation=X488MU3D))
            # matrix = self.IncidenceMatrix_H()
            w = self.w_edges(input_list=self.w)
            # print(f"normalization weighted matrix: {w}")
            D = w@matrix # should be an 1- array
            if diag is False:
                return D
            else:
                return np.diag(D)
            # print(f"Dgree of vertex: \n{D}")
            # if diag is False:
            #     return D.sum(axis=1)
            # else:
            #     return np.diag(D.sum(axis=1))
        
    def w_edges(self, input_list):
        """
        “A weighted hypergraph is a hypergraph that has a positive number w(e) associated with each hyperedge e, 
        called the weight of hyperedge e.” [Zhou et al., 2006, p. 3]
        “W denote the diagonal matrix containing the weights of hyperedges.”
        """
        if input_list is not None:
            # receive as list and convert to diagonal np.array
            # convert input_list to np.array if itss not an array yet
            assert len(input_list) == len(self.Edges), f"input_list length: {len(input_list)} and Edges: {len(self.Edges)} are not equal."
            if type(input_list) != np.ndarray:
                input_list = np.array(input_list)
            return input_list
        else:
            # convert input_list to np.array if its not an array yet
            if type(input_list) != np.ndarray:
                input_list = np.array(self.w)
            return np.ones(len(self.Vertex))

    def LaplacianMatrix_L(self):
        """
        Laplacian matrix of a hypergraph G = (V, E)
        “L(H ) = D − A(H )” ([Bretto, 2013, p. 8]
        L(H ) = D − A(H ), where D = diag(D(x1), D(x2), . . . , D(xn)).
        """
        norm = False
        
        return self.Deg_Vertex(norm=norm, diag=True) - self.AdjacencyMatrix_A(norm=norm)

    
    def Normalized_LapaclianMatrix(self):
        """
        Normalized Laplacian matrix of a hypergraph G = (V, E, w)
        L(H ) = D − A(H ), where D = diag(D(x1), D(x2), . . . , D(xn)).
        “∆=I−1 2 D−1/2 HW HT D−1/2” [Zhou et al., 2006, p. 5]
        W : diagonal weighted matrix, default is 1
        I : identity matrix
        D : Inverted degree of vertex D_inv
        """
        # create an diagnol indentity matrix
        I = np.eye(self.n)
        H = self.IncidenceMatrix_H()
        A = self.AdjacencyMatrix_A()
        D = self.Deg_Vertex(norm=True,diag=True)
        # if H is None:
        #     H = self.IncidenceMatrix_H()
        # else:
        #     H = H.T
        # if Adj is not None:
        #     A = Adj
        #     D = self.Deg_Vertex(norm=True,diag=True, Adj=Adj)
        # else:
        #     A = self.AdjacencyMatrix_A()
        #     D = self.Deg_Vertex(norm=True,diag=True)

        # inverse D_v
        #check if D has inf or nan
        assert not np.any(np.isnan(D)), "D has nan"
        assert not np.any(np.isinf(D)), "D has inf"
        # inverse the elments in diagnol matrix D one by one
        # create a same size as diagnol matrix D
        D_inv = np.zeros((self.n, self.n))
        for i in range(self.n):
            if D[i,i] != 0:
                D_inv[i,i] = 1/np.sqrt(D[i,i])
            else:
                D_inv[i,i] = 0
        # weight of edges, need to convert to diagonal matrix while computing
        if self.data_new is None:
            # w = self.w_edges(input_list=self.w)
            w = np.ones(len(self.Edges))
        else:
            w = np.ones(len(self.Edges))
        return I - 0.5*D_inv@H@np.diag(w)@H.T@D_inv

    def S_int(self, S):
        """
        Convert provided S to a list of integer
        and wrap into a dataframe
        """
        # check id exist in self.Vertex
        assert all(element in [int(element) for element in list(self.Vertex['id'])] for element in S), "S contains label names not in V"
        df = pd.DataFrame()
        df['id'] = S
        df['label'] = self.Vertex['label'].iloc[self.Vertex['id'].iloc[S]]
        # Convert pd.Series to list 
        df['label'] = list(self.Vertex['label'].iloc[self.Vertex['id'].iloc[S]])
        return df
    def S_str(self, S):
        """
        Map provided S to a list of vertices
        and wrap into a dataframe with id
        """
        # check label names exist in self.Vertex
        assert all(element in list(self.Vertex['label']) for element in S), "S contains label names not in V"
        df = pd.DataFrame()
        df['id'] = list(self.Vertex['id'][self.Vertex['label'].isin(S)])
        df['label'] = S
        return df

    def S(self, S):
        """
        Cardinality of a subset of nodes.
        i.e., a vertices set S is a subset of V
        Input data type: should be either the id of the vertices or the label name of the vertices
        """

        assert len(S) <= len(self.Vertex), "Size of S is greater than V"
        # check mixture data
        first_el = S[0]
        for i in S[1:]:
            if type(i) != type(first_el):
                invalid_type = [type(j) for j in S]
                raise ValueError(f"Set has mixture data type, Type:{invalid_type}")
            
        # Try to convert S=['1', '2'] to S = [1, 2] case
        try:
            S = [int(element) for element in S]
            return self.S_int(S)
        except:
            pass
                

        # check elements in S is unique
        if pd.Series(S).is_unique:
            pass # no duplicate
        else:
            print(f"\nfind duiplicate in set.")
            S = pd.Series(S).unique()
            S = list(map(int, S)) # and convert to int
        

        if not all(isinstance(x, int) for x in S):
            # check if all instance of S is interger
            # may be float in list
            print(f"\nSet is not a list of integer id")
            if type(S[0]) == str:
                # the case when S is a list of string
                return self.S_str(S)
        else:
            try:
                S = list(map(int, S)) # convert S to integer
                if pd.Series(S).is_unique:
                    # check duplicate after convert
                    pass
                else:
                    print(f"\nfind duiplicate in set.")
                    S = pd.Series(S).unique()
                    S = list(map(int, S))
            except:
                raise ValueError("S is not a list of integer or convertable")
            pass
        
        try:
            # check element in the list can be converted to int
            S = list(map(int, S)) # convert a floatlist to intlist
            return self.S_int(S)
        except:
            case: type(S[0]) == str
            return self.S_str(S)
        
def is_positive_semidefinite(matrix):
    """
    Check if a matrix is positive-semidefinite.
    64-bit double precision, with an approximate absolute normalized range of 0 and 10 -308 to 10 308 and with a precision of about 16 decimal digits.
    Parameters:
    - matrix: numpy array, the input matrix to be checked.
    Returns:
    - True if the matrix is positive-semidefinite, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are non-negative
    # return all(eigenvalue >= 0 for eigenvalue in eigenvalues)
    # any eigenvalues in matrix is less than 1e-17, replace it with 0
    eigenvalues = [0 if (eigenvalue) < 1e-17 else eigenvalue for eigenvalue in eigenvalues]
    # check all eigenvalues are positive
    return all(eigenvalue >= 0 for eigenvalue in eigenvalues)

def round_eig(eigenvalues)-> np.array:
    """
    Due to machine precision, round up until 1e-17
    # Parameters:
    ------------------------------------------
    @ eigenvalues: np.array
    @ Returns : np.array
    ------------------------------------------
    """
    return np.array([0 if abs(eigenvalue) < 1e-15 else eigenvalue for eigenvalue in eigenvalues])

def eigen_dict(matrix, round=None):
    """
    Compose a dictionary of eigenvalues and eigenvectors of a matrix. <br/>
    Using np.linalg.eig from scipy.linalg package. (scipy.linalg.eigh is for "symmetric" Hermitian matrix) <br/>
    ----------
    Parameters:
    ----------
        @matrix: matrix to be decomposed
        @round: round the eigenvalues to a certain decimal place, default is True
        @return: eigen_dict{'eigenvalues', 'eigenvectors'} a dictionary of eigenvalues and eigenvectors
    ----------
    """
    eigenvalues, eigenvectors = linalg.eig(matrix) # return unsorted eigenvalues

    if round is not False or None:
        eigen_dict = {
            'eigenvalues': round_eig(eigenvalues),
            'eigenvectors': eigenvectors
        }
    else:
        eigen_dict = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }

    return eigen_dict