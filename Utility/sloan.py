from collections import  defaultdict
import pandas as pd

class rooted_level_structure(object):
    def __init__(self, graph, root) -> None:
        self.graph = graph
        # self.root is the smallest degree node
        # if graph is less than 2 nodes
        if root is None:
            if len(self.graph.nodes) < 2:
                self.root = list(self.graph.nodes)[0]
            else:
                # print(f"{len(self.graph.nodes)}")
                min_degree = min([self.graph.degree(i) for i in self.graph.nodes])
                self.root = [i for i in self.graph.nodes if self.graph.degree(i) == min_degree]
            # make self.root hashable
            self.root = self.root[0]
        else:
            self.root = root
    def level_structure(self):
        if type(self.root) is list:
            self.root = self.root[0]
        levels = {0: {(self.root)}}
        visited = {node: False for node in self.graph.nodes}
        visited[self.root[0]] = True
        queue = [(self.root, 0)]
        while queue:
            current_node, current_level = queue.pop(0)
            # make sure the current_node is a string
            if type(current_node) is not str:
                current_node = current_node[0]

            next_level = current_level + 1
            if next_level not in levels:
                levels[next_level] = set()
            for neighbor in self.graph.neighbors(current_node):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, next_level))
                    levels[next_level].add(neighbor)
        # # if empty levels, delete them
        for level in list(levels.keys()):
            if not levels[level]:
                del levels[level]    
        return levels
    
    def max_width(self):
        levels = self.level_structure()
        return max([len(levels[level]) for level in levels])
    def max_depth(self):
        levels = self.level_structure()
        return max(levels.keys())+1
    def level_depth(self, level):
        levels = self.level_structure()
        return len(levels)
    def level_width(self, level):
        levels = self.level_structure()
        return len(levels[level])
    
    def sort_and_shrink_last_level(self):
        levels = self.level_structure()
        last_level = levels[self.max_depth() - 1]
        last_level = list(last_level)
        last_level.sort(key=lambda node: self.graph.degree(node))
        degree_to_node = defaultdict(lambda: None)
        q = []
        for node in last_level:
            degree = self.graph.degree(node)
            if degree_to_node[degree] is None:
                q.append(node)
                degree_to_node[degree] = node
        return q
    
def find_sloan_ppnode(graph, root=None):
        """
        Parameters
        ----------
        graph : hypernextkx graph
        root (s) : the starting node, default is None. None would return the minimum degree node. <br/> 
        ex: 'sig'.
        Return : tuple of (s, e) <br/>
        -------
        """
        # Step 1 (First guess for starting node)
        if root is None:
             # return the minimum degree node
            if len(graph.nodes) < 2:
                root = list(graph.nodes)[0]
            else:
                min_degree = min([graph.degree(i) for i in graph.nodes])
                root = [i for i in graph.nodes if graph.degree(i) == min_degree]
        Ground_level = rooted_level_structure(graph=graph, root=root)
        
        # Below is a loop to find the pseudo-peripheral node
        # When inner loop is break, continue the loop
        # until both e and s are not None
        e = None
        # loop in the condition that if any root or e is None
        # while root is not None and e is None :

        while root is not None and e is None :
            # last_level = Ground_level.level_structure()[Ground_level.max_depth() - 1]
            q = Ground_level.sort_and_shrink_last_level()
            we = float('inf')
            for i in q:
                i_level = rooted_level_structure(graph=graph, root=i)
                if i_level is not None:
                    if i_level.max_width() > Ground_level.max_width():
                        root = i
                        Ground_level = i_level
                        q = Ground_level.sort_and_shrink_last_level()
                        continue
                    else:
                        if i_level.max_width() < we:
                            e = i
                            we = i_level.max_width()
                    break
        if type(root) is list:
            root = root[0]

        return tuple([root, e])


# sloan ordering
# step 1 : compute the distance of two nodes
class Sloan_distance_fnc(object):
    def __init__(self, graph) -> None:
        self.graph = graph

    def Sloan_distance(self, a,b):
        # if any of a or b is list, conver to string
        if isinstance(a, list):
            a = a[0]
        if isinstance(b, list):
            b = b[0]
        return int(self.graph.distance(a,b, s =1))

    # Step 2 : compute the distance of each node that from rooted level struc of e to the node e.
    def compute_distances(self, e):
        Rt_level = rooted_level_structure(self.graph, root=e)
        rls_e = Rt_level.level_structure()
        # rls_e is a dictionary structure, the key is the level, the value is the nodes in the level
        dd = {}
        for i in rls_e.keys():
            for j in rls_e[i]:
                # compute the distance of delta_j to e
                dd[j] = self.Sloan_distance(j, e)
        return dd

# Step 3 : compute the inac priority of each node
def Assign_inac_priority(delta, graph, W1=None, W2=None, node=None):
    # add ini_priority of each node to delta
    # degree of the node
    if W1 is None:
        W1 = 1
    if W2 is None:
        W2 = 2

    p = {}
    for i in delta.keys():
        # set inactive status
        p[i] = {}
        p[i]['status'] = 'inactive'
        p[i]['priority'] = W1*delta[i] - W2*(graph.degree(i)+1)
        # map key name to nodes id
    property = ['node', 'status', 'priority']
    p = pd.DataFrame(p).T
    # add column name 'node' to df first column
    p.insert(0, 'node', p.index)
    # drop the index column
    p.reset_index(drop=True, inplace=True)
    return p

# Step 4 : Initialize node count and priority queue
class Sloan_order_class(object):
    def __init__(self, p, graph, s) -> None:
        self.p = p # priority
        self.graph = graph
        self.s = s  # starting node

    # Step 4 : Initialize node count and priority queue : 
    # a. Initlize the resulting node order to empty list : next function
    # b. Assign node 's' to a preactive status
    # c. Let 'q' denote a priority queue of length n 
    # d. n = 0  and q[n] = s
    def Intialize_node_queue(self):
        # a. : see next function

        # b.
        # swhitch status of s node to preactive
        self.p.loc[self.p['node'] == self.s, 'status'] = 'preactive'
        # c. 
        # create an empty list q
        # get the index of s node
        s_index = self.p[self.p['node'] == self.s].index[0]
        print(s_index)
        q =[]           # initialize priority queue
        q.append(s_index)

        return q
    
    # Step 5
    def Sloan_order(self, W2=None):
        result = []
        q = self.Intialize_node_queue()
        if W2 is None:
            W2 = 2

        while len(q) > 0:
            # return the maximum priority of p with index
            max_v = self.p.loc[self.p.index[q], 'priority'].max()
            W2 = 2
            # find the second max priority of p with index
            # find the index of max_v in p
            if len(self.p[self.p['priority'] == max_v]) > 1:
                # use the first index of max_v in p
                m = self.p.index[self.p['priority'] == max_v][0]
            else:
                m = self.p.index[self.p['priority'] == max_v][0]
            # locate the node i by index of m in p
            if m in q:
                q_max = q.index(m)
            else:
                if len(q) == 1:
                    list_m_index = self.p.index[self.p['priority'] == max_v]
                    # return the list_m_index that is in q
                    # get the index of list_m_index that is equal to q
                    m = [j for j in list_m_index if j in q][0]

                else:

                    m = self.p.index[self.p['priority'] == max_v][1]
                    list_m_index = self.p.index[self.p['priority'] == max_v]
                    m = [j for j in list_m_index if j in q][0]
                q_max = q.index(m)
            # node i by index of m in p
            i = self.p.loc[m, 'node']
            n = len(q) - 1
            # swap the max priority with the last index of q
            q[q_max] = q[n]
            q.pop(n)
            # ==================== Step 7 : Update queue and priorities ====================
            if (self.p['status'][self.p['node'] == i] == 'preactive').any():
                for j in self.graph.neighbors(i):
                    self.p.loc[self.p['node'] == j, 'priority'] += W2
                    if  (self.p['status'][self.p['node'] == j] == 'inactive').any():
                        self.p.loc[self.p['node'] == j, 'status'] = 'preactive'
                        # get index list of j in p
                        j_index = self.p.index[self.p['node'] == j]
                        q.append(j_index.values[0])
            # ==================== Step 8 : Label the next node ====================
                        # index of i in p
            i_index = self.p.index[self.p['node'] == i]
            result.append(i_index.values[0])
            # print(f"result: {result}")
            # switch i status to postactive in p
            self.p.loc[self.p['node'] == i, 'status'] = 'postactive'
            # ==================== Step 9 : Update priorities and queue ====================
            for j in self.graph.neighbors(i):
                if (self.p['status'][self.p['node'] == j] == 'preactive').any():
                    # set j to active
                    self.p.loc[self.p['node'] == j, 'status'] = 'active'
                    # update priority of j
                    self.p.loc[self.p['node'] == j, 'priority'] += W2
                    for k in self.graph.neighbors(j):
                        if (self.p['status'][self.p['node'] == k] != 'postactive').any():
                            self.p.loc[self.p['node'] == k, 'priority'] += W2
                            if (self.p['status'][self.p['node'] == k] == 'inactive').any():
                                self.p.loc[self.p['node'] == k, 'status'] = 'preactive'
                                # get index list of k in p
                                k_index = self.p.index[self.p['node'] == k]
                                q.append(k_index.values[0])
            # ================== Step 10 : Exit with the new node order ====================
        return result
    
# Map a list of node index to nodes name and return with a pd.DataFrame

def lst_node(lst, nodes):
    df = pd.DataFrame()
    df['id'] = lst
    df['node'] = df['id'].map(nodes['label'])

    return df

def sloan_orderer4C(graph, df):
    """
    To execute the sloan ordering for clusters.
    graph: the partition graph
    cluster: the subgraph of the graph
    df is for id for the cluster
    """
    # Step1 : find the pseudo-peripheral node, starting and end point
    s, e = find_sloan_ppnode(graph)
    # R_e = rooted_level_structure(graph, root = e)
    # COmpute distance
    delta = Sloan_distance_fnc(graph).compute_distances(e)
    p = Assign_inac_priority(delta, graph)
    ordering_sequence = Sloan_order_class(p, graph, s).Sloan_order()

    return Sloan_Cluster_id(df, p, ordering_sequence)

def Sloan_Cluster_id(df, p, ordering_result):
    """
    # Solve
    This function aims to solve the id resetting when using sloan ordering.
    In short, the result of sloan ordering recounted from 0 to n-1. 
    The Sloan orering result returns with a list of index for cluster dataframe.
    The function check if the label name mathces the df label name, return the id that correspond to the name.
    Parameters
    ----------
    df : pd.DataFrame of the cluster input
    p : pd.DataFrame that is putted into sloan ordering
    ordering_result : list() the result of sloan ordering
    return : pd.DataFrame of the cluster input with id remap to the label name
    """
    Sloan_result = pd.DataFrame()
    Sloan_result['order'] = ordering_result
    Sloan_result['label'] = Sloan_result['order'].map(p['node'])

    id_list =[]
    # in the order of Sloan result.
    for i in ordering_result:
        for j in range(len(df)):
            # if the label name matches the df label name, return the id that correspond to the name.
            if p.iloc[i]['node'] == df.iloc[j]['label']:
                id_list.append(df.iloc[j]['id'])
    Sloan_result['id'] = id_list
    Sloan_result.drop(columns=['order'], inplace=True)
    return Sloan_result


