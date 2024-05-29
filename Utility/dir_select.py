from tkinter import filedialog
from tkinter.filedialog import askdirectory
from pathlib import Path
import os
import pandas as pd
Default_route = Path(__file__).parent.parent.parent
print(f"Default route: {Default_route}")

class SelectFiles:
    def __init__(self, default_path=Default_route):
        self.default_path = default_path
    
    def select_dir(self):
        folder_selected = askdirectory(initialdir=(self.default_path))
        return Path(folder_selected)
    def list_files_in_dir(self):
        files = os.listdir(self.default_path)
        return files
    def select_files(self):
        """
        default_path: the default path to open the file dialog \n
        allow multiple files to be selected
        """
        file_selected = list(filedialog.askopenfilename(initialdir=select_dir(self.default_path), multiple=True))
        filename = [Path(file).name for file in file_selected]
        return zip(file_selected,filename)
    def makedf(self, data):
        """Find all data, and read them a dataframe"""
        df = {}
        for file,name in data:
            df[name] = pd.read_csv(file, sep=',')
        return df