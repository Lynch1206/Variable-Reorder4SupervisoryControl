
import itertools as itt
import numpy as np

def make_node_pair_matrix(H):
    # Make H's nodes relationships in to pairs
    node_pairs = list(itt.combinations(H.nodes, 2))
    node_pair_counts = {}

    # pair frequency 
    for edge in H.edges:
        for pair in itt.combinations(H.edges[edge], 2):
            # Convert the tuple to a frozenset to make it hashable
            frozen_pair = frozenset(pair)
            # Check if the pair is already a key in the dictionary
            if frozen_pair in node_pair_counts:
                node_pair_counts[frozen_pair] += 1
            else:
                # If not, initialize the count for this pair
                node_pair_counts[frozen_pair] = 1

    # Initialize the matrix
    node_pair_matrix = np.zeros((len(H.nodes), len(H.nodes)))

    Nodes = list(H.nodes)

    # # Fill in the matrix
    for pair, count in node_pair_counts.items():
        # Get the indices of the nodes in the pair
        node1, node2 = pair
        node1_idx = Nodes.index(node1)
        node2_idx = Nodes.index(node2)
        # print(Nodes.index(node1), Nodes.index(node2), count)
        # Add the count to the matrix
        node_pair_matrix[node1_idx, node2_idx] = count
        node_pair_matrix[node2_idx, node1_idx] = count
    return node_pair_matrix