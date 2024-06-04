from pathlib import Path
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.sparse import csgraph



def subplot_variable(component, nodes = None,figsize=(15, 1.5), mark='x',title=None, save=False, save_path=None, dpi=300, picformat = 'png'):
    """plot a verticle plot to present the order and distribution of the variable nodes
    Parameters :
    ------------
    @ component: result of decomposition, see Utility/components_analysis.py ~ @component_eigenvalues
        0: II_mu
        1: II_mu_index
        2: II_mu_eigenvector
    @ nodes: nodes dataframe, with attributes 'label' and 'id'
    @ figsize: figure size(default: (15, 1.5))
    @ mark: mark for nodes(default: 'x')
    @ title: title of the figure(default: None)
    @ save: save the figure or not(default: False)
    @ save_path: path to save the figure(default: None)
    @ return: figure
    """
    save_path = str(save_path)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f"Hypergraph: {title}")
    fig.subplots_adjust(top=0.8)
    for i in range(len(component[2])):
        ax.scatter(component[2][i], 0, label = f"nodes in components {i}", marker = mark)
        ax.annotate(nodes['label'][i], (component[2][i], 0))
    if save is not False:
        # plt.savefig(save_folder+'/'+f"{Title}_Hypergraph.png", dpi=300)
        plt.savefig(save_path+'/'+f"{title}_VariableOrder.{picformat}", dpi=dpi)
    else:
        pass