


import pandas as pd


def load_dataset():
    pass



def split_dataset_by_terms():
    """
    sql like split
    """
    
def split_dataset_by_indices():
    """
    random indices split, by cluster algorithm {kmeans, meanshift, dbscan} 
    """





class AsColumn():

    def histogram(self):
        pass

    def counter(self):
        pass

def histogram():
    pass

def column_counter():
    pass



"""
API
========================================================
"""

"""
USING DIR dirx
IMPORT DIR <dirpath>

USING TABLE tablex
LOAD/RELOAD <filename> FROM dirx

LS 

SET COLUMN NAME ["sex","age","score"]
SET COLUMN TYPE {"sex":"cat"}

HIST sex (n_bin=5) # n_bin>n_cat will be meaningless
HIST age (n_bin=5)

GROUPBY sex 
SPLIT age percent_range(0.1,0.8)



"""


