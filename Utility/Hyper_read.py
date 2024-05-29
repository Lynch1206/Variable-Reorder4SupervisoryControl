from Utility.Hygraph_json import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class Read_graph():
    def __init__(self, file) -> None:
        self.file = file
    
    def read_json(self):
        """Read json file"""
        return Hyper_Read((self.file)).read_js()
    def return_graph_title(self):
        """return title of the graph"""
        df = self.read_json()
        return df['Title']
    def return_graph(self):
        """return graph"""
        return Hyper_Read((self.file)).df2hnx()
    def nodes(self):
        """return nodes"""
        return Hyper_Read((self.file)).js2df()[0]
    def edges(self):
        """return edges"""
        return Hyper_Read((self.file)).js2df()[1]